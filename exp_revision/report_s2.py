"""Freeze S2 masks from raw fold availability and report actual deletion rates."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np

from .common import ROOT, config_snapshot, json_dump, sha256
from .fold_data import load_fold
from .missingness import make_mask


RATES = (0.10, 0.25, 0.50, 0.75)


def report(dataset: str, fold: int = 0, seed: int = 42, *, smoke: bool = False) -> dict:
    data = load_fold(dataset, fold, smoke=smoke)
    result_dir = ROOT / "results_v2" / ("T2_mask_smoke" if smoke else "T2_masks") / dataset / f"fold{fold}_seed{seed}"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing mask audit: {result_dir}")
    result_dir.mkdir(parents=True)
    mask_root = ROOT / ("masks_smoke" if smoke else "masks") / dataset
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    event_rows = []
    summaries = []
    for p in RATES:
        param = f"{p:.2f}"
        folder = mask_root / f"S2_{param}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"fold{fold}_seed{seed}.npz"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite frozen S2 masks: {target}")
        masks = {}
        deleted_total = available_total = 0
        expectation = 0.0
        for offset, incident_id in enumerate(data.test_ids, start=len(data.train_ids)):
            a0 = data.availability[offset]
            k, realized = make_mask({"dataset": dataset, "incident_id": incident_id}, a0,
                                    "S2", param, seed)
            if np.any((a0.sum(-1) > 0) & ((a0 & k).sum(-1) == 0)):
                raise AssertionError("S2 recovery failed for an available service")
            available = int(a0.sum())
            deleted = int(np.sum((a0 == 1) & (k == 0)))
            if not np.isclose(realized, deleted / max(available, 1)):
                raise AssertionError("realized rate does not match saved mask")
            expectation += sum(float(m * p - p**m) for m in a0.sum(-1) if m)
            deleted_total += deleted
            available_total += available
            masks[incident_id] = k
            event_rows.append({"dataset": dataset, "fold": fold, "seed": seed,
                               "scenario": "S2", "requested_rate": p,
                               "incident_id": incident_id, "available_cells": available,
                               "deleted_cells": deleted, "realized_rate": realized})
        np.savez_compressed(target, **masks)
        summaries.append({"dataset": dataset, "fold": fold, "seed": seed,
                          "scenario": "S2", "requested_rate": p,
                          "events": len(data.test_ids), "available_cells": available_total,
                          "deleted_cells": deleted_total,
                          "realized_rate": deleted_total / max(available_total, 1),
                          "expected_rate_under_restoration": expectation / max(available_total, 1),
                          "mask_sha256": sha256(target)})
    for name, rows in (("realized_rates.csv", event_rows), ("summary.csv", summaries)):
        with (result_dir / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    feature_path = ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}{'_smoke' if smoke else ''}/features.npz"
    meta = config_snapshot(
        f"python -m exp_revision.report_s2 --dataset {dataset} --fold {fold} --seed {seed}{' --smoke' if smoke else ''}",
        {"task": "T2", "component": "S2_mask_realized_rate", "dataset": dataset,
         "fold": fold, "seed": seed, "smoke": smoke, "requested_rates": list(RATES),
         "restoration": "uniformly restore one originally available modality when a service becomes empty"},
    )
    meta.update(start_time=started, end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                status="partial_T2_S2_masks_only", feature_sha256=sha256(feature_path),
                events=len(data.test_ids))
    json_dump(result_dir / "run_meta.json", meta)
    return {"dataset": dataset, "fold": fold, "seed": seed,
            "status": "partial_T2_S2_masks_only", "summary": summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(report(args.dataset, args.fold, args.seed, smoke=args.smoke))


if __name__ == "__main__":
    main()
