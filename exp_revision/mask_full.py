"""Freeze shared held-out missingness masks from raw, fold-aligned A0.

This is a mask audit, not a robustness result. Only the S3 family reads held-out
root labels, as required to identify (or avoid) the root service for that mask.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .common import ROOT, SEEDS, dataset_dir, json_dump, sha256
from .fold_data import load_fold
from .missingness import SCENARIOS, make_mask


def _write_npz(path: Path, masks: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as old:
            if set(old.files) != set(masks) or any(
                not np.array_equal(old[key], value) for key, value in masks.items()
            ):
                raise FileExistsError(f"refusing to replace a different mask: {path}")
        return
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **masks)
    temporary.replace(path)


def _roots(dataset: str, services: list[str]) -> dict[str, int]:
    source = dataset_dir(dataset)
    incidents = json.loads((source / "incidents.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["services"] != services:
        raise ValueError("root label service order differs from fold")
    with np.load(source / "features.npz") as archive:
        labels = archive["labels"].astype(int)
    if len(incidents) != len(labels):
        raise ValueError("root label count differs from incidents")
    roots = {row["incident_id"]: int(label)
             for row, label in zip(incidents, labels)}
    if len(roots) != len(incidents):
        raise ValueError("duplicate root-label incident ID")
    return roots


def generate(dataset: str, *, smoke: bool = False,
             output_root: Path = ROOT / "masks",
             report_root: Path = ROOT / "results_v2/T2") -> dict:
    folds = [load_fold(dataset, fold, smoke=False) for fold in range(5)]
    services = folds[0].services
    if any(fold.services != services for fold in folds):
        raise ValueError("fold service orders differ")
    cases: list[tuple[int, str, np.ndarray]] = []
    for fold in folds:
        for offset, incident_id in enumerate(fold.test_ids):
            cases.append((fold.fold, incident_id,
                          fold.availability[len(fold.train_ids) + offset].astype(np.uint8)))
    counts = Counter(incident_id for _, incident_id, _ in cases)
    cohort = json.loads((ROOT / f"splits/{dataset}_unlabeled_events.json").read_text())
    if counts != Counter(row["incident_id"] for row in cohort):
        raise ValueError("held-out mask cases are missing, repeated, or outside cohort")
    if smoke:
        cases = [row for row in cases if row[0] == 0][:5]
        scenarios, seeds = SCENARIOS, SEEDS[:1]
        output_root = ROOT / "results_v2/T2_mask_smoke_full/masks"
        report_root = ROOT / "results_v2/T2_mask_smoke_full"
    else:
        scenarios, seeds = SCENARIOS, SEEDS
    roots = None
    detail: list[dict] = []
    summary: list[dict] = []
    for scenario, param in scenarios:
        if scenario in {"S3", "S3c"} and roots is None:
            roots = _roots(dataset, services)
        folder = output_root / dataset / f"{scenario}_{param or 'none'}"
        for seed in seeds:
            masks: dict[str, np.ndarray] = {}
            for fold, incident_id, a0 in cases:
                root = roots[incident_id] if scenario in {"S3", "S3c"} else None
                k, rate = make_mask({"dataset": dataset, "incident_id": incident_id},
                                    a0, scenario, param, seed, root=root)
                available = int(a0.sum())
                deleted = int(((a0 == 1) & (k == 0)).sum())
                if not np.isclose(rate, deleted / max(available, 1)):
                    raise ValueError("realized deletion rate differs from mask")
                if scenario == "S2" and np.any((a0.sum(axis=1) > 0)
                                                & ((a0 * k).sum(axis=1) == 0)):
                    raise ValueError("S2 erased every originally available modality")
                if scenario == "S3c" and not np.array_equal(k[root], np.ones(3, dtype=np.uint8)):
                    raise ValueError("S3c changed the root service")
                masks[incident_id] = k
                detail.append({"dataset": dataset, "fold": fold, "seed": seed,
                               "scenario": scenario, "scenario_param": param,
                               "requested_p": float(param) if scenario == "S2" else "",
                               "incident_id": incident_id,
                               "original_available": available, "deleted": deleted,
                               "realized_rate": rate})
            destination = folder / f"seed{seed}.npz"
            _write_npz(destination, masks)
            rows = detail[-len(cases):]
            original = sum(row["original_available"] for row in rows)
            removed = sum(row["deleted"] for row in rows)
            summary.append({"dataset": dataset, "seed": seed,
                            "scenario": scenario, "scenario_param": param,
                            "requested_p": float(param) if scenario == "S2" else "",
                            "events": len(cases), "original_available": original,
                            "deleted": removed,
                            "realized_rate": removed / max(original, 1),
                            "mask_sha256": sha256(destination)})
    report_root.mkdir(parents=True, exist_ok=True)
    with (report_root / ".mask_report.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, new_rows in (("realized_rates.csv", detail),
                               ("realized_rates_summary.csv", summary)):
            path = report_root / name
            existing = []
            if path.exists():
                with path.open(newline="") as stream:
                    existing = [row for row in csv.DictReader(stream)
                                if row["dataset"] != dataset]
            rows = existing + new_rows
            sort_keys = ("dataset", "scenario", "scenario_param", "seed", "fold",
                         "incident_id") if name == "realized_rates.csv" else (
                         "dataset", "scenario", "scenario_param", "seed")
            rows.sort(key=lambda row: tuple(str(row[key]) for key in sort_keys))
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
    meta = {"status": "mask_audit_only", "dataset": dataset,
            "smoke": smoke, "events": len(cases), "seeds": list(seeds),
            "scenarios": [list(item) for item in scenarios],
            "detail_rows": len(detail), "summary_rows": len(summary),
            "source": "fold_preprocessing raw availability (A0)",
            "mask_code_sha256": sha256(ROOT / "exp_revision/missingness.py"),
            "generator_sha256": sha256(ROOT / "exp_revision/mask_full.py"),
            "split_sha256": sha256(ROOT / f"splits/{dataset}_kfold5.json"),
            "fold_availability_sha256": {
                str(fold.fold): sha256(ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold.fold}/features.npz")
                for fold in folds}}
    json_dump(report_root / f"{dataset}_masks_meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("tt", "aiops"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.dataset, smoke=args.smoke)))


if __name__ == "__main__":
    main()
