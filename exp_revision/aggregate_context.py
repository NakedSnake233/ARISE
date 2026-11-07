"""Evaluate and aggregate the complete context-only outer-fold matrix.

This reads held-out labels only after all training outputs have been verified.
It deliberately names the product context-only, not full ARISE/T1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import ROOT, json_dump, load_bundle, sha256
from .context_queue import FOLDS, _verified_output, verify_manifest
from .evaluate_context import evaluate
from .fold_data import load_fold
from .common import SEEDS


def aggregate(dataset: str, manifest_path: Path,
              root: Path = ROOT / "results_v2/T1_context_full") -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"] != dataset:
        raise ValueError("dataset differs from the frozen queue manifest")
    folds = verify_manifest(manifest)
    folder = root / dataset
    state = json.loads((folder / "queue_state.json").read_text())
    if state["status"] != "completed" or state["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("full context queue is incomplete or belongs to another manifest")
    expected_jobs = {f"{dataset}_fold{fold}_seed{seed}" for fold in FOLDS for seed in SEEDS}
    if set(state["jobs"]) != expected_jobs:
        raise ValueError("queue does not contain the exact 5-fold by 5-seed matrix")
    frames = []
    for fold in FOLDS:
        for seed in SEEDS:
            job = state["jobs"][f"{dataset}_fold{fold}_seed{seed}"]
            if job["status"] != "completed":
                raise ValueError(f"fold {fold}, seed {seed} is incomplete")
            output = ROOT / job["output"]
            if not _verified_output(output, manifest, fold, seed, folds[fold]):
                raise ValueError(f"fold {fold}, seed {seed} failed output verification")
            evaluate(dataset, fold, seed, output=output)
            frame = pd.read_parquet(output / "predictions.parquet")
            if len(frame) != len(folds[fold].test_ids):
                raise ValueError(f"fold {fold}, seed {seed} has missing predictions")
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    bundle = load_bundle(dataset)
    expected = Counter((row["incident_id"], seed)
                       for row in bundle.rows for seed in SEEDS)
    observed = Counter(zip(result["incident_id"], result["seed"]))
    if observed != expected:
        raise ValueError("events are missing, duplicated, or outside the frozen cohort")
    if set(result["method"]) != {"context-only"} or set(result["scenario"]) != {"S0"}:
        raise ValueError("the context-only product contains another method or scenario")
    failed = int((result["status"] != "ok").sum())
    if failed / len(result) > 0.1:
        raise RuntimeError("more than 10% of context-only predictions failed")
    destination = folder / "predictions.parquet"
    if destination.exists():
        old = pd.read_parquet(destination)
        if not old.equals(result):
            raise FileExistsError("refusing to replace a different aggregate")
    else:
        result.to_parquet(destination, index=False)
    summary = {"status": "partial_T1_context_only", "dataset": dataset,
               "events": len(bundle.rows), "seeds": list(SEEDS), "rows": len(result),
               "failed": failed, "manifest_sha256": sha256(manifest_path),
               "predictions_sha256": sha256(destination),
               "charged_core_or_gpu_hours": state["charged_core_or_gpu_hours"]}
    json_dump(folder / "aggregate_meta.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "results_v2/T1_context_full")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.dataset, args.manifest, args.root)))


if __name__ == "__main__":
    main()
