# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student USN:** 1MS25CI037 · Skill Lab (MLOps with MLflow) mini project

> Work in progress. Results are added after the final run.

## 1. Task and models

| | |
|---|---|
| Task | Binary sentiment classification (negative / positive) |
| Dataset | GLUE **SST-2**, `validation` split (`nyu-mll/glue`, config `sst2`) |
| Subset | 200 sentences, random sample with seed 42, saved in `data/eval_inputs.csv` |
| Model A (current Champion) | `distilbert-base-uncased-finetuned-sst-2-english` (~67M params) |
| Model B (Challenger) | `textattack/bert-base-uncased-SST-2` (~110M params, outputs `LABEL_0/LABEL_1`, mapped to negative/positive) |

## 3. Champion rule (written before the final run)

This rule is fixed in `src/config.py` and in this section **before** the final evaluation run, and it is not changed after seeing results (see git history).

1. **Primary metric:** macro-F1 on the 200-sentence evaluation set. **Higher is better.**
2. **Minimum improvement margin:** the Challenger must beat the Champion by **at least 2 percentage points of macro-F1** (`delta = F1_B − F1_A ≥ 0.02`). Otherwise the Champion stays.
   *Why 2 points:* with 200 examples one sentence is worth 0.5 points, and the standard error of an accuracy near 90 % is about √(0.9·0.1/200) ≈ 2.1 points. A smaller margin would let noise decide the winner, and swapping a deployed model has a real cost (a bigger model, re-validation), so the Challenger has to show a gain about the size of one standard error.
3. **Guardrail:** the Challenger is rejected regardless of accuracy if its average latency per example is **more than 3× the Champion's**.
4. **Tie-breaker:** if `|delta| < 0.02` (within the margin), the **faster** model (lower `avg_latency_ms`) is the Champion. If the Challenger is worse by more than the margin, the Champion stays.

Metric directions used in this project:

| Metric | Direction |
|---|---|
| accuracy, macro_f1, precision_macro, recall_macro, f1_negative, f1_positive | higher is better |
| avg_latency_ms, p95_latency_ms, total_inference_s | lower is better |
| delta_primary_metric (B − A) | positive means the Challenger is better |
