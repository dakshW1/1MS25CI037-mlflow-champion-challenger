"""Load the Champion from the MLflow Model Registry by ALIAS and predict on new data.

This script never imports or calls Hugging Face itself - all prediction goes through
mlflow.pyfunc (the logged wrapper uses transformers internally). It also does not know
which version or which Hugging Face model is the Champion: it reads that from the registry.

Usage:
    python predict_from_mlflow.py                       # champion only
    python predict_from_mlflow.py --alias challenger    # any alias
    python predict_from_mlflow.py --compare             # champion and challenger side by side
"""
import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as C  # noqa: E402  (only names: tracking URI + registered model name)


def describe_alias(client: MlflowClient, alias: str) -> dict:
    """Read version + HF model ID from the registry / source run, not from local config."""
    mv = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, alias)
    hf_id = mv.tags.get("hf_model_id") or client.get_run(mv.run_id).data.params.get("hf_model_id")
    return {"alias": alias, "version": mv.version, "hf_model_id": hf_id, "run_id": mv.run_id}


def predict_with_alias(client, alias, texts):
    info = describe_alias(client, alias)
    uri = f"models:/{C.REGISTERED_MODEL_NAME}@{alias}"
    model = mlflow.pyfunc.load_model(uri)
    print(f"Loaded {uri}")
    print(f"  -> registry version : {info['version']}")
    print(f"  -> hf_model_id      : {info['hf_model_id']}")
    print(f"  -> source run_id    : {info['run_id']}")
    preds = model.predict(pd.DataFrame({"text": texts}))
    return info, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--compare", action="store_true", help="also run the @challenger alias")
    ap.add_argument("--input", default=str(ROOT / "data" / "new_data.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "new_data_predictions.csv"))
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()
    new_df = pd.read_csv(args.input)
    texts = new_df["text"].tolist()

    aliases = [args.alias] + (["challenger"] if args.compare and args.alias != "challenger" else [])
    out = new_df[["id", "text"]].copy()
    for alias in aliases:
        info, preds = predict_with_alias(client, alias, texts)
        out[f"{alias}_label"] = preds["label"].values
        out[f"{alias}_score"] = preds["score"].round(4).values
        out[f"{alias}_version"] = info["version"]
        out[f"{alias}_hf_model_id"] = info["hf_model_id"]
    if "my_label" in new_df.columns:
        out["my_label"] = new_df["my_label"]

    show_cols = ["id", "text"] + [c for a in aliases for c in (f"{a}_label", f"{a}_score")]
    show_cols += ["my_label"] if "my_label" in out.columns else []
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print("\nPredictions on new data (not part of the evaluation set):")
        print(out[show_cols].to_string(index=False))

    Path(args.out).parent.mkdir(exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved -> {Path(args.out).relative_to(ROOT) if Path(args.out).is_relative_to(ROOT) else args.out}")


if __name__ == "__main__":
    main()
