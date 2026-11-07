"""Register T3 leakage pipelines or T6 evidence ablations, after smoke."""
import argparse
import json
import subprocess

from .common import ROOT, SEEDS, json_dump, sha256
from .fold_data import load_fold
from .missingness import SCENARIOS
from .remaining_queue import PRECHARGED, verified
from .train_variant_evidence import T3_VARIANTS, T6_VARIANTS, T6_SCENARIOS


def create(task, smoke=False):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # Reuse the already-frozen telemetry, span and context prerequisite closure.
    parent = json.loads((ROOT / "manifests/remaining/06_full_evidence_masks.json").read_text())
    sources = {name: digest for name, digest in parent["source_sha256"].items()}
    for name in ("exp_revision/train_variant_evidence.py", "exp_revision/variant_manifest.py"):
        sources[name] = sha256(ROOT / name)
    sources["manifests/remaining/06_full_evidence_masks.json"] = sha256(ROOT / "manifests/remaining/06_full_evidence_masks.json")
    settings = T3_VARIANTS if task == "T3" else T6_VARIANTS
    scenarios = [f"{s}_{p or 'none'}" for s, p in SCENARIOS] if task == "T3" else T6_SCENARIOS
    jobs = []
    for dataset in ("tt", "aiops"):
        for fold in ([0] if smoke else range(5)):
            data = load_fold(dataset, fold); ids = data.test_ids[:5] if smoke else data.test_ids
            for seed in ([42] if smoke else SEEDS):
                key = f"{'smoke' if smoke else 'full'}_{task.lower()}_evidence_{dataset}_fold{fold}_seed{seed}"
                command = ["-m", "exp_revision.train_variant_evidence", "--task", task, "--dataset", dataset,
                           "--fold", str(fold), "--seed", str(seed)]
                if smoke: command.append("--smoke")
                jobs.append({"id": key, "source_commit": commit, "resource": "cpu", "cores": 4,
                    "timeout_seconds": 600 if smoke else 1200, "max_attempts": 2, "command": command,
                    "expected_status": ["completed_variant_evidence"],
                    "expected_config": dict(task=task, dataset=dataset, fold=fold, seed=seed, smoke=smoke,
                        methods=list(settings), variants=settings, scenarios=scenarios, alpha=.75, split="kfold5",
                        preregistered=False if task == "T6" else None),
                    "test_ids": ids, "score_shape": [len(settings), len(scenarios), len(ids), len(data.services)]})
    if not smoke:
        state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
        for dataset in ("tt", "aiops"):
            row = state["jobs"][f"smoke_{task.lower()}_evidence_{dataset}_fold0_seed42"]
            if row["status"] != "completed" or not verified(ROOT / row["output"], row["spec"]):
                raise ValueError("both evidence variant smoke jobs must pass")
    manifest = dict(backend="local", phase=f"{'smoke' if smoke else 'full'}_{task.lower()}_evidence",
                    source_commit=commit, precharged_hours=PRECHARGED, source_sha256=sources, jobs=jobs)
    number = (7 if smoke else 8) if task == "T3" else (9 if smoke else 10)
    path = ROOT / f"manifests/remaining/{number:02d}_{'smoke' if smoke else 'full'}_{task.lower()}_evidence.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest: raise FileExistsError("immutable variant manifest differs")
    else: json_dump(path, manifest)
    return path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--task", choices=("T3", "T6"), required=True)
    p.add_argument("--smoke", action="store_true"); a = p.parse_args(); print(create(a.task, a.smoke))


if __name__ == "__main__": main()
