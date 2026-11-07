"""Verify Appendix D against retained fixed-split prediction arrays.

This is an artifact-level reproduction check. It does not claim fresh model
training; that distinction is recorded in the CSV and run metadata.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .common import RESULT_ROOT, ROOT, SEEDS, config_snapshot, json_dump
from .metrics import ranking_metrics


TARGETS = {
    "tt": {
        "ARISE": (94.67, 2.98, 96.83, 1.60),
        "context-only": (71.33, 9.60, 80.83, 7.79),
        "ET-spec": (98.00, 1.83, 99.00, .91),
        "Eadro": (90.00, 5.27, 94.89, 2.56),
        "MULAN": (70.00, 5.27, 77.57, 2.78),
    },
    "aiops": {
        "ARISE": (92.31, 2.72, 95.60, 1.75),
        "context-only": (78.46, 7.98, 83.64, 4.46),
        "ET-spec": (92.31, 0.00, 96.15, 0.00),
        "Eadro": (81.54, 3.22, 87.91, 1.31),
        "MULAN": (64.62, 5.01, 76.85, 2.57),
    },
}

SOURCE_DATASET = {"tt": "trainticket", "aiops": "aiops2022"}
SOURCE_KEY = {"ARISE": "final", "context-only": "base", "ET-spec": "standard_evidence"}


def _rank(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or len(scores) != len(labels):
        raise ValueError("prediction/label shape mismatch")
    order = np.argsort(-scores, axis=1, kind="stable")
    matches = order == labels[:, None]
    if not matches.any(axis=1).all():
        raise ValueError("root absent from full ranking")
    return matches.argmax(axis=1) + 1


def verify_dataset(dataset: str) -> list[dict]:
    name = SOURCE_DATASET[dataset]
    rows = []
    for method, target in TARGETS[dataset].items():
        metrics = []
        source_paths = []
        for seed in SEEDS:
            primary = ROOT / "artifacts/care/review-response-v1" / name / f"seed-{seed}" / "predictions.npz"
            archive = np.load(primary)
            labels = archive["labels"]
            source_paths.append(str(primary.relative_to(ROOT)))
            if method in SOURCE_KEY:
                scores = archive[SOURCE_KEY[method]]
            else:
                baseline = ROOT / "artifacts/care/fse2027-baselines" / name / method / f"seed-{seed}" / "scores.npz"
                other = np.load(baseline)
                if not np.array_equal(archive["indices"], other["test_indices"]):
                    raise ValueError("baseline test IDs do not align")
                if not np.array_equal(labels, other["labels"][archive["indices"]]):
                    raise ValueError("baseline labels do not align")
                scores = other["test_scores"]
                source_paths.append(str(baseline.relative_to(ROOT)))
            metrics.append(ranking_metrics(_rank(scores, labels)))
        values = []
        for metric in ("PR@1", "MAP@5"):
            series = np.asarray([m[metric] for m in metrics]) * 100
            values.extend([float(series.mean()), float(series.std(ddof=1))])
        row = {
            "dataset": dataset, "method": method, "status": "artifact_verified",
            "n_seeds": len(SEEDS), "n_test_events": len(labels),
            "pr1_mean": values[0], "pr1_sd": values[1],
            "map5_mean": values[2], "map5_sd": values[3],
            "target_pr1_mean": target[0], "target_pr1_sd": target[1],
            "target_map5_mean": target[2], "target_map5_sd": target[3],
            "within_rounding_tolerance": all(abs(a - b) <= .011 for a, b in zip(values, target)),
            "source_predictions": json.dumps(source_paths),
        }
        rows.append(row)
    return rows


def write_reproduction(dataset: str) -> list[dict]:
    rows = verify_dataset(dataset)
    path = RESULT_ROOT / "T0_reproduction.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["dataset"] != dataset]
    all_rows = existing + rows
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    json_dump(RESULT_ROOT / "T0" / dataset / "run_meta.json", {
        **config_snapshot(f"python -m exp_revision.run --task T0 --dataset {dataset}", {"seeds": SEEDS, "source": "retained_predictions"}),
        "start_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "end_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verification_kind": "saved-prediction recalculation; no fresh training",
    })
    if not all(r["within_rounding_tolerance"] for r in rows):
        raise RuntimeError(f"T0 Appendix D mismatch for {dataset}; inspect results_v2/T0_reproduction.csv")
    return rows
