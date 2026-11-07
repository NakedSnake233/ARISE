"""Evaluate saved context-only scores after model training has finished."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, config_snapshot, json_dump, load_bundle, sha256, write_rows
from .fold_data import load_fold
from .metrics import ranking_metrics


def evaluate(dataset: str, fold: int, seed: int, *, smoke: bool = False,
             output: Path | None = None) -> dict:
    if output is None:
        suffix = "T1_context_smoke" if smoke else "T1_context"
        output = Path("results_v2") / suffix / dataset / f"fold{fold}_seed{seed}"
    run_meta = json.loads((output / "run_meta.json").read_text())
    if run_meta["status"] != "partial_T1_context_only":
        raise ValueError("missing completed context-only training metadata")
    if sha256(output / "scores.npz") != run_meta["scores_sha256"]:
        raise ValueError("saved scores differ from the recorded training output")
    for relative, expected_hash in run_meta["input_sha256"].items():
        path = ROOT / relative
        if sha256(path) != expected_hash:
            raise ValueError(f"training input differs from evaluated input: {relative}")
    fold_data = load_fold(dataset, fold, smoke=smoke)
    expected_ids = fold_data.train_ids + fold_data.test_ids
    with np.load(output / "scores.npz") as archive:
        ids = archive["incident_ids"].tolist()
        scores = archive["probabilities"]
    if ids != expected_ids or scores.shape != (len(ids), len(fold_data.services)):
        raise ValueError("score IDs, order, or service dimension differs from frozen fold")

    # This is the first point in the workflow that reads held-out root and
    # fault-type labels. Training and checkpoint selection use FoldData only.
    bundle = load_bundle(dataset)
    by_id = {row["incident_id"]: i for i, row in enumerate(bundle.rows)}
    if fold_data.services != bundle.services:
        raise ValueError("service order differs from evaluation metadata")
    rows = []
    for offset, incident_id in enumerate(fold_data.test_ids, start=len(fold_data.train_ids)):
        incident = bundle.rows[by_id[incident_id]]
        root_index = int(bundle.labels[by_id[incident_id]])
        score = scores[offset]
        valid = bool(np.isfinite(score).all() and np.all(score >= 0)
                     and np.isclose(score.sum(), 1.0, atol=1e-4))
        if valid:
            order = np.argsort(-score, kind="stable")
            rank = float(np.flatnonzero(order == root_index)[0] + 1)
            top5 = [fold_data.services[int(i)] for i in order[:5]]
        else:
            rank, top5 = float("inf"), []
        rows.append({
            "task": "T1_smoke" if smoke else "T1", "method": "context-only",
            "dataset": dataset, "split": "kfold5", "fold": fold, "seed": seed,
            "scenario": "S0", "scenario_param": "", "realized_rate": 0.0,
            "incident_id": incident_id, "fault_type": incident["fault_type"],
            "root_true": fold_data.services[root_index], "rank_of_root": rank,
            "top5": json.dumps(top5, ensure_ascii=False),
            "status": "ok" if valid else "failed", "invalid_by_design": False,
            "runtime_ms": None,
        })
    failed = sum(row["status"] == "failed" for row in rows)
    metrics = ranking_metrics(np.asarray([row["rank_of_root"] for row in rows]))
    predictions_path = output / "predictions.parquet"
    if predictions_path.exists():
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(predictions_path),
                                          pd.DataFrame(rows), check_dtype=False)
        except AssertionError as exc:
            raise FileExistsError("refusing to replace different saved predictions") from exc
    else:
        write_rows(predictions_path, rows)
    summary_path = output / "summary.csv"
    summary = {"dataset": dataset, "method": "context-only", "fold": fold,
               "seed": seed, "n": len(rows), "failed": failed,
               **{key: round(100 * value, 2) for key, value in metrics.items()}}
    if not summary_path.exists():
        with summary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)
    else:
        with summary_path.open(newline="") as file:
            previous = next(csv.DictReader(file))
        if previous != {key: str(value) for key, value in summary.items()}:
            raise FileExistsError("refusing to replace different fold summary")
    meta = config_snapshot(
        f"python -m exp_revision.evaluate_context --dataset {dataset} --fold {fold} --seed {seed}{' --smoke' if smoke else ''}",
        {"task": "T1", "method": "context-only", "dataset": dataset,
         "fold": fold, "seed": seed, "smoke": smoke},
    )
    meta.update(start_time=run_meta["end_time"],
                end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                status="partial_T1_context_only", prediction_rows=len(rows),
                failed=failed, failed_rate=failed / len(rows))
    meta["predictions_sha256"] = sha256(predictions_path)
    evaluation_meta = output / "evaluation_meta.json"
    if evaluation_meta.exists():
        previous = json.loads(evaluation_meta.read_text())
        if (previous.get("predictions_sha256") != meta["predictions_sha256"]
                or previous.get("prediction_rows") != len(rows)):
            raise FileExistsError("refusing to replace different evaluation metadata")
    else:
        json_dump(evaluation_meta, meta)
    if failed / len(rows) > 0.1:
        raise RuntimeError(f"failed prediction rate {failed / len(rows):.1%} exceeds 10%")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(evaluate(args.dataset, args.fold, args.seed, smoke=args.smoke, output=args.output))


if __name__ == "__main__":
    main()
