"""Register Model A and Model B in the MLflow Model Registry and set the
`champion` / `challenger` aliases according to the decision logged by evaluate.py.

No run IDs are hard-coded: the script finds the latest finished `compare_sentiment`
summary run in the experiment and reads the child run IDs and the winner from its tags.

Usage:
    python register_champion.py                 # register + set aliases from the decision
    python register_champion.py --swap-aliases  # stretch goal: flip champion <-> challenger
"""
import argparse
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as C  # noqa: E402


def latest_summary_run(client: MlflowClient):
    exp = client.get_experiment_by_name(C.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment {C.EXPERIMENT_NAME!r} not found - run evaluate.py first.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.run_name = '{C.SUMMARY_RUN_NAME}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No finished compare_sentiment run found - run evaluate.py first.")
    return runs[0]


def get_or_create_version(client: MlflowClient, run_id: str, slot: str):
    """Reuse an existing registry version for this run (idempotent), otherwise register."""
    for mv in client.search_model_versions(f"name = '{C.REGISTERED_MODEL_NAME}'"):
        if mv.run_id == run_id:
            return mv
    spec = C.MODELS[slot]
    mv = mlflow.register_model(
        f"runs:/{run_id}/{C.MODEL_ARTIFACT_NAME}",
        C.REGISTERED_MODEL_NAME,
        tags={
            "hf_model_id": spec["hf_model_id"],
            "slot": slot,
            "short_name": spec["short_name"],
            "source_run_name": C.run_name(slot),
        },
    )
    return client.get_model_version(C.REGISTERED_MODEL_NAME, mv.version)


def show_aliases(client: MlflowClient):
    print(f"\nRegistered model: {C.REGISTERED_MODEL_NAME}")
    print(f"{'alias':<12}{'version':<9}{'hf_model_id':<52}run_id")
    for alias in ["champion", "challenger"]:
        try:
            mv = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, alias)
            print(f"{alias:<12}{mv.version:<9}{mv.tags.get('hf_model_id', '?'):<52}{mv.run_id}")
        except Exception:
            print(f"{alias:<12}(not set)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-aliases", action="store_true",
                    help="stretch goal: point champion at the current challenger version and vice versa")
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()

    if args.swap_aliases:
        champ = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, "champion")
        chall = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, "challenger")
        client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", chall.version)
        client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", champ.version)
        print(f"Swapped: champion -> v{chall.version}, challenger -> v{champ.version}")
        show_aliases(client)
        return

    summary = latest_summary_run(client)
    tags = summary.data.tags
    winner = tags["winner_slot"]
    loser = "B" if winner == "A" else "A"
    print(f"Summary run {summary.info.run_id} ({C.SUMMARY_RUN_NAME}): winner = {tags['winner']}")
    print(f"Reason: {tags['decision_reason']}")

    try:
        client.create_registered_model(
            C.REGISTERED_MODEL_NAME,
            description=f"SST-2 sentiment models compared in experiment {C.EXPERIMENT_NAME}. "
                        "Load with models:/sentiment-model@champion.",
            tags={"task": C.TASK, "student_usn": C.STUDENT_USN},
        )
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    versions = {}
    for slot in ["A", "B"]:
        mv = get_or_create_version(client, tags[f"run_id_{slot}"], slot)
        versions[slot] = mv
        print(f"Model {slot}: {C.MODELS[slot]['hf_model_id']} -> version {mv.version}")

    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", versions[winner].version)
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", versions[loser].version)
    for slot, role in [(winner, "champion"), (loser, "challenger")]:
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, versions[slot].version, "decided_role", role)
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, versions[slot].version,
                                     "decision_run_id", summary.info.run_id)
    show_aliases(client)


if __name__ == "__main__":
    main()
