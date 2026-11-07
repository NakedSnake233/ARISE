"""Evaluate the complete saved context-only T2 scenario matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, SEEDS, json_dump, load_bundle, sha256
from .fold_data import load_fold
from .missingness import SCENARIOS


def aggregate(dataset: str) -> dict:
    folder = ROOT / "results_v2/T2_context_full" / dataset
    folds = {fold: load_fold(dataset, fold, smoke=False) for fold in range(5)}
    expected_keys = [f"{scenario}_{param or 'none'}" for scenario, param in SCENARIOS]
    with (ROOT / "results_v2/T2/realized_rates.csv").open(newline="") as stream:
        rate_rows = [row for row in csv.DictReader(stream) if row["dataset"] == dataset]
    rates = {(row["incident_id"], int(row["seed"]), row["scenario"],
              row["scenario_param"]): row for row in rate_rows}
    expected_rate_count = sum(len(fold.test_ids) for fold in folds.values()) * len(SEEDS) * len(SCENARIOS)
    if len(rates) != expected_rate_count or len(rate_rows) != expected_rate_count:
        raise ValueError("actual deletion-rate record is incomplete or duplicated")
    for seed in SEEDS:
        for scenario, param in SCENARIOS:
            mask_path = ROOT / "masks" / dataset / f"{scenario}_{param or 'none'}" / f"seed{seed}.npz"
            with np.load(mask_path) as archive:
                if set(archive.files) != set(incident_id for fold in folds.values()
                                              for incident_id in fold.test_ids):
                    raise ValueError("shared mask does not cover exact evaluation cohort")
                for fold in folds.values():
                    for offset, incident_id in enumerate(fold.test_ids):
                        a0 = fold.availability[len(fold.train_ids) + offset]
                        mask = archive[incident_id]
                        removed = int(((a0 == 1) & (mask == 0)).sum())
                        rate = rates[(incident_id, seed, scenario, param)]
                        if (int(rate["deleted"]) != removed
                                or int(rate["original_available"]) != int(a0.sum())
                                or not np.isclose(float(rate["realized_rate"]),
                                                  removed / max(int(a0.sum()), 1))):
                            raise ValueError("reported deletion rate differs from shared mask")
    # Verify the exact 5×5 score matrix before opening any held-out labels.
    verified = {}
    for fold, data in folds.items():
        for seed in SEEDS:
            job_id = f"{dataset}_fold{fold}_seed{seed}"
            output = folder / job_id
            meta = json.loads((output / "run_meta.json").read_text())
            if (meta["status"] != "partial_T2_context_only"
                    or (meta["config"]["dataset"], meta["config"]["fold"],
                        meta["config"]["seed"], meta["config"]["smoke"])
                    != (dataset, fold, seed, False)
                    or meta["scores_sha256"] != sha256(output / "scores.npz")
                    or meta["baseline_s0_max_abs_diff"] > 1e-5):
                raise ValueError(f"invalid context missingness metadata: {job_id}")
            for relative, digest in {**meta["source_sha256"],
                                     **meta["mask_sha256"]}.items():
                if sha256(ROOT / relative) != digest:
                    raise ValueError(f"changed source or mask for {job_id}: {relative}")
            with np.load(output / "scores.npz") as archive:
                if (set(archive.files) != {"incident_ids", "scenario_keys", "probabilities"}
                        or archive["incident_ids"].tolist() != data.test_ids
                        or archive["scenario_keys"].tolist() != expected_keys):
                    raise ValueError(f"context masked score schema differs: {job_id}")
                scores = archive["probabilities"]
            if (scores.shape != (len(SCENARIOS), len(data.test_ids), len(data.services))
                    or not np.isfinite(scores).all() or (scores < 0).any()
                    or not np.allclose(scores.sum(axis=-1), 1, atol=1e-5)):
                raise ValueError(f"invalid context masked scores: {job_id}")
            verified[(fold, seed)] = scores
    bundle = load_bundle(dataset)
    root_by_id = {row["incident_id"]: int(bundle.labels[i])
                  for i, row in enumerate(bundle.rows)}
    fault_by_id = {row["incident_id"]: row["fault_type"] for row in bundle.rows}
    if len(root_by_id) != len(bundle.rows):
        raise ValueError("duplicate evaluation incident ID")
    predictions = []
    for fold, data in folds.items():
        if data.services != bundle.services:
            raise ValueError("evaluation service order differs from fold")
        for seed in SEEDS:
            scores = verified[(fold, seed)]
            for scenario_index, (scenario, param) in enumerate(SCENARIOS):
                for event_index, incident_id in enumerate(data.test_ids):
                    rate = rates[(incident_id, seed, scenario, param)]
                    root = root_by_id[incident_id]
                    score = scores[scenario_index, event_index]
                    order = np.argsort(-score, kind="stable")
                    rank = float(np.flatnonzero(order == root)[0] + 1)
                    predictions.append({"task": "T2", "method": "context-only",
                                        "dataset": dataset, "split": "kfold5",
                                        "fold": fold, "seed": seed,
                                        "scenario": scenario, "scenario_param": param,
                                        "realized_rate": float(rate["realized_rate"]),
                                        "incident_id": incident_id,
                                        "fault_type": fault_by_id[incident_id],
                                        "root_true": data.services[root],
                                        "rank_of_root": rank,
                                        "top5": json.dumps([data.services[int(j)] for j in order[:5]],
                                                           ensure_ascii=False),
                                        "status": "ok", "invalid_by_design": False,
                                        "runtime_ms": None})
    result = pd.DataFrame(predictions)
    expected = Counter((row["incident_id"], seed, scenario, param)
                       for row in bundle.rows for seed in SEEDS
                       for scenario, param in SCENARIOS)
    actual = Counter(zip(result["incident_id"], result["seed"],
                         result["scenario"], result["scenario_param"]))
    if actual != expected:
        raise ValueError("context T2 scenario matrix is incomplete")
    destination = folder / "predictions.parquet"
    if destination.exists():
        if not pd.read_parquet(destination).equals(result):
            raise FileExistsError("refusing to replace different T2 predictions")
    else:
        result.to_parquet(destination, index=False)
    summary = {"status": "partial_T2_context_only", "dataset": dataset,
               "events": len(bundle.rows), "seeds": list(SEEDS),
               "scenarios": [list(item) for item in SCENARIOS],
               "prediction_rows": len(result), "failed": 0,
               "predictions_sha256": sha256(destination)}
    json_dump(folder / "aggregate_meta.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("tt", "aiops"))
    args = parser.parse_args()
    print(json.dumps(aggregate(args.dataset)))


if __name__ == "__main__":
    main()
