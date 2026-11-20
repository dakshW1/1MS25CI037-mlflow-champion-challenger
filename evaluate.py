"""Run Model A (Champion) and Model B (Challenger) on the same SST-2 subset and
log everything to MLflow.

MLflow layout (experiment: hf-sentiment-champion-challenger):
    compare_sentiment                 <- parent / summary run (decision lives here)
      |- A_sentiment_distilbert-sst2  <- nested run, Model A
      |- B_sentiment_bert-base-sst2   <- nested run, Model B

Usage:
    python evaluate.py              # full run (200 samples)
    python evaluate.py --limit 10   # quick debug run on the first 10 samples
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from huggingface_hub import hf_hub_download
from mlflow.models import infer_signature
from scipy.stats import binomtest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# MLflow prints emoji run links; make that safe on Windows consoles (cp1252)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as C  # noqa: E402
from sentiment_pyfunc import HFSentimentModel  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
EVAL_CSV = DATA_DIR / "eval_inputs.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_eval_set() -> pd.DataFrame:
    """Create the shared evaluation set once and reuse the saved CSV afterwards,
    so both models (and every rerun) see the same rows in the same order."""
    if EVAL_CSV.exists():
        return pd.read_csv(EVAL_CSV)
    path = hf_hub_download(
        C.DATASET_NAME,
        f"{C.DATASET_CONFIG}/{C.DATASET_SPLIT}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    full = pd.read_parquet(path)
    sample = full.sample(n=C.EVAL_SAMPLES, random_state=C.SEED).reset_index(drop=True)
    df = pd.DataFrame(
        {
            "row_id": range(len(sample)),
            "sst2_idx": sample["idx"],
            "text": sample["sentence"].str.strip(),
            "reference": [C.LABELS[i] for i in sample["label"]],
        }
    )
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(EVAL_CSV, index=False)
    return df


# ---------------------------------------------------------------------------
# Prediction + metrics
# ---------------------------------------------------------------------------
def predict(pipe, texts, label_map):
    """Run one example at a time so we get a clean per-example latency."""
    preds, scores, raw, lat_ms = [], [], [], []
    pipe(texts[0], truncation=True, max_length=C.MAX_LENGTH)  # warm-up, not timed
    for t in texts:
        t0 = time.perf_counter()
        out = pipe(t, truncation=True, max_length=C.MAX_LENGTH)[0]
        lat_ms.append((time.perf_counter() - t0) * 1000)
        raw.append(out["label"])
        if out["label"] not in label_map:
            raise ValueError(f"Unexpected label {out['label']!r}; update label_map in config.py")
        preds.append(label_map[out["label"]])
        scores.append(float(out["score"]))
    return preds, scores, raw, lat_ms


def compute_metrics(refs, preds, lat_ms):
    f1_per = f1_score(refs, preds, labels=C.LABELS, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(refs, preds),
        "macro_f1": f1_score(refs, preds, labels=C.LABELS, average="macro", zero_division=0),
        "precision_macro": precision_score(refs, preds, labels=C.LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(refs, preds, labels=C.LABELS, average="macro", zero_division=0),
        "f1_negative": float(f1_per[0]),
        "f1_positive": float(f1_per[1]),
        "n_errors": int(sum(r != p for r, p in zip(refs, preds))),
        "avg_latency_ms": float(np.mean(lat_ms)),
        "p95_latency_ms": float(np.percentile(lat_ms, 95)),
        "total_inference_s": float(np.sum(lat_ms) / 1000),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion(refs, preds, title):
    cm = confusion_matrix(refs, preds, labels=C.LABELS)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=C.LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_metric_bars(m_a, m_b):
    keys = ["accuracy", "macro_f1", "precision_macro", "recall_macro", "f1_negative", "f1_positive"]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    ax.bar(x - 0.2, [m_a[k] for k in keys], 0.4, label=f"A champion ({C.MODELS['A']['short_name']})", color="#4C72B0")
    ax.bar(x + 0.2, [m_b[k] for k in keys], 0.4, label=f"B challenger ({C.MODELS['B']['short_name']})", color="#DD8452")
    for i, k in enumerate(keys):
        ax.text(i - 0.2, m_a[k] + 0.003, f"{m_a[k]:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, m_b[k] + 0.003, f"{m_b[k]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, keys, rotation=15)
    lo = min(min(m_a[k], m_b[k]) for k in keys)
    ax.set_ylim(max(0, lo - 0.08), 1.0)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Quality metrics on the shared SST-2 subset")
    ax.legend(loc="lower right", fontsize=8)
    ax2 = axes[1]
    ax2.bar(["A", "B"], [m_a["avg_latency_ms"], m_b["avg_latency_ms"]], color=["#4C72B0", "#DD8452"])
    for i, v in enumerate([m_a["avg_latency_ms"], m_b["avg_latency_ms"]]):
        ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("avg latency per example, ms (lower is better)")
    ax2.set_title("CPU latency")
    fig.tight_layout()
    return fig


def plot_confidence(df_a, df_b):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, df, name in [(axes[0], df_a, "A champion"), (axes[1], df_b, "B challenger")]:
        bins = np.linspace(0.5, 1.0, 21)
        ax.hist(df.loc[df.correct, "score"], bins=bins, alpha=0.75, label="correct", color="#55A868")
        ax.hist(df.loc[~df.correct, "score"], bins=bins, alpha=0.9, label="wrong", color="#C44E52")
        ax.set_title(f"{name}: confidence of predicted label")
        ax.set_xlabel("pipeline score")
        ax.legend()
    axes[0].set_ylabel("# examples")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Champion rule (fixed before the run, see README section 3)
# ---------------------------------------------------------------------------
def decide(m_a, m_b):
    delta = m_b[C.PRIMARY_METRIC] - m_a[C.PRIMARY_METRIC]
    latency_ratio = m_b["avg_latency_ms"] / m_a["avg_latency_ms"]
    if latency_ratio > C.LATENCY_GUARDRAIL:
        return "A", delta, latency_ratio, (
            f"Guardrail: challenger latency is {latency_ratio:.2f}x the champion's (> {C.LATENCY_GUARDRAIL}x)."
        )
    if delta >= C.MARGIN:
        return "B", delta, latency_ratio, (
            f"Challenger improves {C.PRIMARY_METRIC} by {delta:+.4f} (>= margin {C.MARGIN})."
        )
    if delta <= -C.MARGIN:
        return "A", delta, latency_ratio, (
            f"Challenger is worse on {C.PRIMARY_METRIC} by {delta:+.4f} (beyond the margin); champion stays."
        )
    faster = "A" if m_a["avg_latency_ms"] <= m_b["avg_latency_ms"] else "B"
    return faster, delta, latency_ratio, (
        f"|delta| = {abs(delta):.4f} < margin {C.MARGIN}: tie-breaker picks the faster model ({faster})."
    )


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on the discordant pairs (informational only, not part of the rule)."""
    b_only = int(np.sum(~correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    n = a_only + b_only
    p = 1.0 if n == 0 else binomtest(b_only, n, 0.5).pvalue
    return a_only, b_only, float(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="debug: only use the first N eval rows")
    args = ap.parse_args()

    set_seed(C.SEED)
    OUT_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(C.TRACKING_URI)
    # debug runs go to a separate experiment so they never pollute the real comparison
    mlflow.set_experiment(C.EXPERIMENT_NAME + ("-debug" if args.limit else ""))

    eval_df = load_eval_set()
    if args.limit:
        eval_df = eval_df.head(args.limit)
    texts, refs = eval_df["text"].tolist(), eval_df["reference"].tolist()
    print(f"Evaluation set: {len(eval_df)} rows from {C.DATASET_NAME}/{C.DATASET_CONFIG}:{C.DATASET_SPLIT}")

    library_versions = (
        f"transformers={transformers.__version__}; torch={torch.__version__}; "
        f"mlflow={mlflow.__version__}; scikit-learn={sklearn.__version__}"
    )
    common_tags = {
        "student_usn": C.STUDENT_USN,
        "library_versions": library_versions,
        "dataset_name": f"{C.DATASET_NAME}/{C.DATASET_CONFIG}:{C.DATASET_SPLIT}",
        "task": C.TASK,
    }

    results = {}
    with mlflow.start_run(run_name=C.SUMMARY_RUN_NAME) as parent:
        mlflow.set_tags({**common_tags, "run_type": "summary"})

        for slot in ["A", "B"]:
            spec = C.MODELS[slot]
            name = C.run_name(slot)
            print(f"\n=== {name}  ({spec['hf_model_id']}) ===")
            with mlflow.start_run(run_name=name, nested=True) as run:
                tok = AutoTokenizer.from_pretrained(spec["hf_model_id"])
                mdl = AutoModelForSequenceClassification.from_pretrained(spec["hf_model_id"])
                mdl.eval()
                pipe = pipeline("text-classification", model=mdl, tokenizer=tok, device=-1)

                preds, scores, raw, lat = predict(pipe, texts, spec["label_map"])
                metrics = compute_metrics(refs, preds, lat)
                print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))

                mlflow.log_params(
                    {
                        "hf_model_id": spec["hf_model_id"],
                        "task": C.TASK,
                        "role": f"{spec['role']} candidate",
                        "slot": slot,
                        "max_length": C.MAX_LENGTH,
                        "batch_size": C.BATCH_SIZE,
                        "truncation": True,
                        "eval_samples": len(eval_df),
                        "seed": C.SEED,
                        "device": C.DEVICE,
                        "label_map": json.dumps(spec["label_map"]),
                        "params_millions": spec["params_millions"],
                    }
                )
                mlflow.set_tags({**common_tags, "run_type": "model", "role": spec["role"]})
                mlflow.log_metrics(metrics)

                # --- artifacts -------------------------------------------------
                short = spec["short_name"]
                pred_df = eval_df[["row_id", "sst2_idx", "text", "reference"]].copy()
                pred_df["raw_label"] = raw
                pred_df["prediction"] = preds
                pred_df["score"] = np.round(scores, 5)
                pred_df["correct"] = pred_df["reference"] == pred_df["prediction"]
                pred_df["latency_ms"] = np.round(lat, 2)
                pred_path = OUT_DIR / f"predictions_{short}.csv"
                pred_df.to_csv(pred_path, index=False)

                summary = {
                    "run_name": name,
                    "hf_model_id": spec["hf_model_id"],
                    "role": spec["role"],
                    "eval_samples": len(eval_df),
                    "metrics": metrics,
                    "directions": {
                        **{k: "higher_is_better" for k in
                           ["accuracy", "macro_f1", "precision_macro", "recall_macro", "f1_negative", "f1_positive"]},
                        **{k: "lower_is_better" for k in
                           ["n_errors", "avg_latency_ms", "p95_latency_ms", "total_inference_s"]},
                    },
                }
                ms_path = OUT_DIR / f"metrics_summary_{short}.json"
                ms_path.write_text(json.dumps(summary, indent=2))

                mlflow.log_artifact(str(pred_path))
                mlflow.log_artifact(str(EVAL_CSV), artifact_path="data")
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td) / "metrics_summary.json"
                    p.write_text(json.dumps(summary, indent=2))
                    mlflow.log_artifact(str(p))
                fig = plot_confusion(refs, preds, f"{slot}: {spec['hf_model_id']}")
                fig.savefig(OUT_DIR / f"confusion_matrix_{short}.png", dpi=120)
                mlflow.log_figure(fig, f"plots/confusion_matrix_{short}.png")
                plt.close(fig)
                mlflow.log_table(pred_df, artifact_file=f"tables/predictions_{short}.json")

                # --- log the model as a self-contained pyfunc -------------------
                export_dir = Path(tempfile.mkdtemp()) / "hf_model"
                mdl.save_pretrained(export_dir)
                tok.save_pretrained(export_dir)
                example = pd.DataFrame({"text": texts[:2]})
                sig = infer_signature(
                    example,
                    pd.DataFrame({"text": texts[:2], "label": preds[:2], "score": scores[:2]}),
                )
                mlflow.pyfunc.log_model(
                    name=C.MODEL_ARTIFACT_NAME,
                    python_model=HFSentimentModel(),
                    artifacts={"hf_model_dir": str(export_dir)},
                    model_config={
                        "hf_model_id": spec["hf_model_id"],
                        "label_map": spec["label_map"],
                        "max_length": C.MAX_LENGTH,
                    },
                    code_paths=[str(ROOT / "src" / "sentiment_pyfunc.py")],
                    signature=sig,
                    input_example=example,
                    pip_requirements=[
                        f"mlflow=={mlflow.__version__}",
                        f"transformers=={transformers.__version__}",
                        f"torch=={torch.__version__.split('+')[0]}",
                        f"pandas=={pd.__version__}",
                    ],
                )
                shutil.rmtree(export_dir.parent, ignore_errors=True)

                results[slot] = {"run_id": run.info.run_id, "metrics": metrics, "df": pred_df}

        # ------------------------- summary run ---------------------------------
        m_a, m_b = results["A"]["metrics"], results["B"]["metrics"]
        winner, delta, lat_ratio, reason = decide(m_a, m_b)
        a_only, b_only, p_val = mcnemar_exact(
            results["A"]["df"]["correct"].to_numpy(), results["B"]["df"]["correct"].to_numpy()
        )
        print(f"\nDecision: winner = {winner} ({C.MODELS[winner]['hf_model_id']})\n  {reason}")
        print(f"  McNemar exact p = {p_val:.3f} (A-only correct: {a_only}, B-only correct: {b_only})")

        mlflow.log_params(
            {
                "primary_metric": C.PRIMARY_METRIC,
                "primary_metric_direction": "higher_is_better",
                "margin": C.MARGIN,
                "latency_guardrail_x": C.LATENCY_GUARDRAIL,
                "tie_breaker": "faster model (lower avg_latency_ms) when |delta| < margin",
                "champion_hf_model_id": C.MODELS["A"]["hf_model_id"],
                "challenger_hf_model_id": C.MODELS["B"]["hf_model_id"],
                "eval_samples": len(eval_df),
                "seed": C.SEED,
            }
        )
        mlflow.set_tags(
            {
                "winner": f"{winner}:{C.MODELS[winner]['hf_model_id']}",
                "winner_slot": winner,
                "decision_reason": reason,
                "run_id_A": results["A"]["run_id"],
                "run_id_B": results["B"]["run_id"],
            }
        )
        mlflow.log_metrics(
            {
                "delta_primary_metric": delta,
                "delta_accuracy": m_b["accuracy"] - m_a["accuracy"],
                "latency_ratio_B_over_A": lat_ratio,
                "A_macro_f1": m_a["macro_f1"],
                "B_macro_f1": m_b["macro_f1"],
                "A_avg_latency_ms": m_a["avg_latency_ms"],
                "B_avg_latency_ms": m_b["avg_latency_ms"],
                "mcnemar_p_value": p_val,
                "n_discordant": a_only + b_only,
            }
        )

        # comparison table + combined metrics_summary.json
        keys = list(m_a.keys())
        comp = pd.DataFrame(
            {"metric": keys, "A_champion": [m_a[k] for k in keys], "B_challenger": [m_b[k] for k in keys]}
        )
        comp["delta_B_minus_A"] = comp["B_challenger"] - comp["A_champion"]
        comp.to_csv(OUT_DIR / "comparison_table.csv", index=False)
        combined = {
            "experiment": C.EXPERIMENT_NAME,
            "eval_samples": len(eval_df),
            "rule": {
                "primary_metric": C.PRIMARY_METRIC,
                "direction": "higher_is_better",
                "margin": C.MARGIN,
                "latency_guardrail_x": C.LATENCY_GUARDRAIL,
                "tie_breaker": "faster model when |delta| < margin",
            },
            "A": {"hf_model_id": C.MODELS["A"]["hf_model_id"], "run_name": C.run_name("A"), "metrics": m_a},
            "B": {"hf_model_id": C.MODELS["B"]["hf_model_id"], "run_name": C.run_name("B"), "metrics": m_b},
            "delta_primary_metric": delta,
            "latency_ratio_B_over_A": lat_ratio,
            "mcnemar": {"A_only_correct": a_only, "B_only_correct": b_only, "p_value": p_val},
            "winner": winner,
            "winner_hf_model_id": C.MODELS[winner]["hf_model_id"],
            "reason": reason,
        }
        (OUT_DIR / "metrics_summary.json").write_text(json.dumps(combined, indent=2))

        # disagreements between the two models
        da, db = results["A"]["df"], results["B"]["df"]
        dis = da[["row_id", "text", "reference"]].copy()
        dis["pred_A"], dis["score_A"] = da["prediction"], da["score"]
        dis["pred_B"], dis["score_B"] = db["prediction"], db["score"]
        dis = dis[dis["pred_A"] != dis["pred_B"]]
        dis["right_model"] = np.where(dis["pred_A"] == dis["reference"], "A", "B")
        dis.to_csv(OUT_DIR / "disagreements_all.csv", index=False)

        fig = plot_metric_bars(m_a, m_b)
        fig.savefig(OUT_DIR / "comparison_bar_chart.png", dpi=120)
        mlflow.log_figure(fig, "plots/comparison_bar_chart.png")
        plt.close(fig)
        fig = plot_confidence(da, db)
        fig.savefig(OUT_DIR / "confidence_histogram.png", dpi=120)
        mlflow.log_figure(fig, "plots/confidence_histogram.png")
        plt.close(fig)

        for f in ["comparison_table.csv", "metrics_summary.json", "disagreements_all.csv"]:
            mlflow.log_artifact(str(OUT_DIR / f))
        if (OUT_DIR / "disagreements_commented.csv").exists():
            mlflow.log_artifact(str(OUT_DIR / "disagreements_commented.csv"))
        mlflow.log_artifact(str(EVAL_CSV), artifact_path="data")
        mlflow.log_artifact(str(ROOT / "README.md"))
        mlflow.log_artifact(str(ROOT / "requirements.txt"))

        print(f"\nSummary run: {parent.info.run_id}  ->  {C.TRACKING_URI}")
        print(f"Next: python register_champion.py")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
