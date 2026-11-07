"""Freeze T2 model reuse and raw-recomputed multi-method score jobs."""
import argparse
import json
import subprocess

from .common import ROOT, SEEDS, json_dump, sha256
from .fold_data import load_fold
from .infer_evidence_masks import METHODS, completed_folder, context_scores
from .missingness import SCENARIOS
from .remaining_queue import PRECHARGED, verified


def create(smoke=False):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = [ROOT / name for name in (
        "exp_revision/evidence_masks_manifest.py", "exp_revision/infer_evidence_masks.py",
        "exp_revision/masked_evidence.py", "exp_revision/raw_spans_v2.py", "exp_revision/train_evidence.py",
        "exp_revision/remaining_queue.py", "exp_revision/common.py", "exp_revision/fold_data.py",
        "exp_revision/missingness.py", "exp_revision/disk_guard_local.py", "exp_revision/disk_guard.py",
        "scripts/care/evidence_reranker.py", "scripts/care/robust_evidence.py", "scripts/care/trace_residual.py")]
    jobs = []
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    for row in state["jobs"].values():
        if row["spec"]["id"].startswith("full_rpc_cache_"):
            if row["status"] != "completed": raise ValueError("raw span cache is incomplete")
            path = ROOT / row["output"] / "run_meta.json"
            receipt = json.loads(path.read_text()); sources.append(path)
            sources.extend(ROOT / entry["path"] for entry in receipt["incident_files"].values())
    sources.extend([ROOT / "manifests/remaining/04_full_span_cache.json", ROOT / "cache/raw_metric_tt.npz"])
    for dataset in ("tt", "aiops"):
        sources.extend([ROOT / f"splits/{dataset}_kfold5.json", ROOT / f"splits/{dataset}_unlabeled_events.json"])
        for fold in range(5):
            sources.extend(ROOT / f"cache/{kind}/{dataset}/fold{fold}/{name}" for kind in ("fold_preprocessing", "fold_evidence_v2")
                           for name in ("features.npz", "metadata.json"))
            sources.append(ROOT / f"splits/{dataset}_training_labels/fold_{fold}.json")
        sources.extend(ROOT / f"masks/{dataset}/{s}_{p or 'none'}/seed{seed}.npz" for s, p in SCENARIOS for seed in SEEDS)
        for fold in ([0] if smoke else range(5)):
            data = load_fold(dataset, fold); ids = data.test_ids[:5] if smoke else data.test_ids
            for seed in ([42] if smoke else SEEDS):
                evidence = completed_folder("results_v2/T1_evidence_full", dataset, fold, seed)
                et = completed_folder("results_v2/T4_et_spec_full", dataset, fold, seed)
                _, context = context_scores(dataset, fold, seed, ids)
                sources.extend(evidence / name for name in ("specific.joblib", "generic.joblib", "run_meta.json", "scores.npz"))
                sources.extend(et / name for name in ("model.joblib", "run_meta.json", "scores.npz"))
                sources.extend(context / name for name in ("run_meta.json", "scores.npz"))
                sources.append(ROOT / f"results_v2/T1_arise_s0/{dataset}/{dataset}_fold{fold}_seed{seed}/scores.npz")
                key = f"{'smoke' if smoke else 'full'}_evidence_masks_{dataset}_fold{fold}_seed{seed}"
                command = ["-m", "exp_revision.infer_evidence_masks", "--dataset", dataset, "--fold", str(fold), "--seed", str(seed)]
                if smoke: command.append("--smoke")
                jobs.append({"id": key, "source_commit": commit, "resource": "cpu", "cores": 4,
                    "timeout_seconds": 600, "max_attempts": 2, "command": command,
                    "expected_status": ["completed_T2_evidence_sanity"],
                    "expected_config": {"task": "T2", "dataset": dataset, "fold": fold, "seed": seed, "smoke": smoke,
                                        "methods": METHODS, "pipeline": "family_gated", "alpha": .75, "split": "kfold5"},
                    "test_ids": ids, "score_shape": [len(METHODS), len(SCENARIOS), len(ids), len(data.services)]})
    if not smoke:
        for dataset in ("tt", "aiops"):
            row = state["jobs"][f"smoke_evidence_masks_{dataset}_fold0_seed42"]
            if row["status"] != "completed" or not verified(ROOT / row["output"], row["spec"]):
                raise ValueError("both T2 raw-evidence smoke jobs must pass")
    manifest = {"backend": "local", "phase": "smoke_evidence_masks" if smoke else "full_evidence_masks",
        "source_commit": commit, "precharged_hours": PRECHARGED,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in dict.fromkeys(sources)}, "jobs": jobs}
    path = ROOT / f"manifests/remaining/{'05_smoke_evidence_masks' if smoke else '06_full_evidence_masks'}.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest: raise FileExistsError("immutable evidence manifest differs")
    else: json_dump(path, manifest)
    return path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); print(create(a.smoke))


if __name__ == "__main__": main()
