"""Custom MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (Lab 9 style).

The Hugging Face weights are stored inside the MLflow model as an artifact
("hf_model_dir"), so the model can be loaded from a runs:/ or models:/ URI on
any machine without knowing the Hugging Face model ID.

Input : DataFrame with a "text" column (or a list of strings)
Output: DataFrame with columns text, label (negative/positive), score
"""
import mlflow.pyfunc
import pandas as pd


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import pipeline  # imported here so callers never touch HF directly

        cfg = context.model_config or {}
        self.label_map = cfg.get("label_map", {})
        self.max_length = int(cfg.get("max_length", 128))
        self.hf_model_id = cfg.get("hf_model_id", "unknown")
        model_dir = context.artifacts["hf_model_dir"]
        self.pipe = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            device=-1,
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            col = "text" if "text" in model_input.columns else model_input.columns[0]
            texts = model_input[col].astype(str).tolist()
        else:
            texts = [str(t) for t in list(model_input)]

        outputs = self.pipe(texts, truncation=True, max_length=self.max_length)
        labels = [self.label_map.get(o["label"], o["label"]) for o in outputs]
        scores = [float(o["score"]) for o in outputs]
        return pd.DataFrame({"text": texts, "label": labels, "score": scores})
