"""Label-safe Random and RootFreqPrior on the frozen grouped folds."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .common import RESULT_ROOT, SEEDS, config_snapshot, json_dump, load_bundle, stable_seed, write_rows
from .metrics import ranking_metrics
from .splits import check_splits, ids_to_indices, load_or_make


def _prediction(bundle, incident_idx: int, order: np.ndarray, fold: int, seed: int, method: str) -> dict:
    # The root and fault-type fields are read only after the complete ranking
    # has been constructed, and are used solely for metrics/output provenance.
    root_idx = int(bundle.labels[incident_idx])
    rank = int(np.flatnonzero(order == root_idx)[0] + 1)
    row = bundle.rows[incident_idx]
    return {
        "task": "T4_sanity", "method": method, "dataset": bundle.dataset,
        "split": "kfold5", "fold": fold, "seed": seed,
        "scenario": "S0", "scenario_param": "", "realized_rate": 0.0,
        "incident_id": row["incident_id"], "fault_type": row["fault_type"],
        "root_true": bundle.services[root_idx], "rank_of_root": float(rank),
        "top5": json.dumps([bundle.services[j] for j in order[:5]], ensure_ascii=False),
        "status": "ok", "invalid_by_design": False, "runtime_ms": None,
    }


def run(dataset: str, smoke: bool = False) -> dict:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    bundle = load_bundle(dataset)
    split = load_or_make(bundle, Path("splits") / f"{dataset}_kfold5.json")
    check_splits(bundle, split)
    rows = []
    fold_items = sorted(split.items())[:1 if smoke else 5]
    seeds = SEEDS[:1 if smoke else len(SEEDS)]
    for fold_name, part in fold_items:
        fold = int(fold_name.split("_")[-1])
        train = ids_to_indices(bundle, part["train"])
        test = ids_to_indices(bundle, part["test"])
        if smoke:
            test = test[:5]
        root_counts = np.bincount(bundle.labels[train], minlength=bundle.n_services)
        prior_order = np.argsort(-root_counts, kind="stable")
        for seed in seeds:
            for index in test:
                rows.append(_prediction(bundle, int(index), prior_order, fold, seed, "RootFreqPrior"))
                rng = np.random.default_rng(stable_seed(dataset, bundle.rows[index]["incident_id"], "Random", seed))
                order = rng.permutation(bundle.n_services)
                rows.append(_prediction(bundle, int(index), order, fold, seed, "Random"))
    out = RESULT_ROOT / "T4_sanity" / dataset
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / ("predictions_smoke.parquet" if smoke else "predictions.parquet"), rows)
    summary = []
    for method in ("RootFreqPrior", "Random"):
        for seed in seeds:
            ranks = np.asarray([r["rank_of_root"] for r in rows if r["method"] == method and r["seed"] == seed])
            if len(ranks):
                summary.append({"dataset": dataset, "method": method, "seed": seed, "n": len(ranks), **ranking_metrics(ranks)})
    # The analytic random reference is 1/S. The Monte Carlo check uses 100
    # permutations of every test incident and is kept separate from predictions.
    random_hits = []
    for fold_name, part in sorted(split.items()):
        for index in ids_to_indices(bundle, part["test"]):
            root = int(bundle.labels[index])
            incident_id = bundle.rows[index]["incident_id"]
            for repetition in range(100):
                order = np.random.default_rng(stable_seed(dataset, incident_id, "RandomCheck", repetition)).permutation(bundle.n_services)
                random_hits.append(int(order[0] == root))
    json_dump(out / "random_check.json", {"analytic_PR@1": 1 / bundle.n_services,
                                           "monte_carlo_100x_PR@1": float(np.mean(random_hits)),
                                           "n_permutations": len(random_hits)})
    json_dump(out / "run_meta.json", {**config_snapshot(f"python -m exp_revision.run --task T4 --dataset {dataset}{' --smoke' if smoke else ''}",
                                                        {"methods": ["RootFreqPrior", "Random"], "seeds": seeds, "scenario": "S0"}),
                                      "start_time": started, "end_time": dt.datetime.now(dt.timezone.utc).isoformat(),
                                      "status": "partial_T4_sanity_only", "prediction_rows": len(rows)})
    with (out / ("summary_smoke.csv" if smoke else "summary.csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    return {"dataset": dataset, "smoke": smoke, "rows": len(rows), "status": "partial_T4_sanity_only"}
