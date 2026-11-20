"""Shared settings for the champion-challenger project.

Everything that both evaluate.py and register_champion.py need lives here, so
the scripts never hard-code run IDs or model versions.
"""
import os

STUDENT_USN = "1MS25CI037"

# MLflow
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
MODEL_ARTIFACT_NAME = "model"  # name used in mlflow.pyfunc.log_model(name=...)
TASK = "sentiment"
SUMMARY_RUN_NAME = f"compare_{TASK}"

# Data
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
DATASET_SPLIT = "validation"  # SST-2 test labels are hidden, validation is held out from training
EVAL_SAMPLES = 200
SEED = 42

# Inference settings (identical for both models)
MAX_LENGTH = 128
BATCH_SIZE = 1  # one example per call so per-example latency is meaningful
DEVICE = "cpu"

# Canonical labels used everywhere in this project
LABELS = ["negative", "positive"]  # index == SST-2 integer label

# Model A = current Champion, Model B = Challenger
MODELS = {
    "A": {
        "role": "champion",
        "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "short_name": "distilbert-sst2",
        "params_millions": 67,
        # the pipeline returns NEGATIVE / POSITIVE
        "label_map": {"NEGATIVE": "negative", "POSITIVE": "positive"},
    },
    "B": {
        "role": "challenger",
        "hf_model_id": "textattack/bert-base-uncased-SST-2",
        "short_name": "bert-base-sst2",
        "params_millions": 110,
        # this checkpoint has no id2label names, so the pipeline returns
        # LABEL_0 / LABEL_1. In SST-2, 0 = negative and 1 = positive.
        "label_map": {"LABEL_0": "negative", "LABEL_1": "positive"},
    },
}

# ---------------------------------------------------------------------------
# Champion rule (fixed BEFORE the final run - see README "Champion rule")
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "macro_f1"   # higher is better
MARGIN = 0.02                 # challenger must beat champion by >= 2 percentage points
LATENCY_GUARDRAIL = 3.0       # challenger rejected if avg latency > 3x champion's


def run_name(slot: str) -> str:
    """A_sentiment_distilbert-sst2 / B_sentiment_bert-base-sst2."""
    return f"{slot}_{TASK}_{MODELS[slot]['short_name']}"
