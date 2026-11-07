"""Train fold-isolated S0 ARISE evidence experts and select beta on training data.

This is the evidence branch only. Full ARISE results require fusing its saved
scores with the independently trained context branch at fixed alpha=0.75.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from .common import ROOT, config_snapshot, json_dump, sha256
from .fold_data import load_fold
from .metrics import ranking_metrics

sys.path.insert(0, str(ROOT / "scripts/care"))
from evidence_reranker import fit, model_for, predict  # noqa: E402
from robust_evidence import (  # noqa: E402
    _augmentation_availabilities, aggregate_type_agnostic,
    apply_availability, fault_type_holdout, fuse_all,
)


def _ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.asarray([int(np.flatnonzero(order[i] == label)[0] + 1)
                       for i, label in enumerate(labels)])


def _fit_one(x: np.ndarray, names: list[str], labels: np.ndarray,
             availability: np.ndarray, indices: np.ndarray, seed: int,
             *, type_agnostic: bool) -> tuple[object, list[str]]:
    if type_agnostic:
        x, names = aggregate_type_agnostic(x, names)
    train_x, a0 = x[indices], availability[indices]
    views = _augmentation_availabilities(train_x.shape, seed, (0.25, 0.5, 0.75))
    augmented = np.concatenate([apply_availability(
        train_x, names, a0 * view, append_flags=True) for view in views])
    augmented_labels = np.tile(labels[indices], len(views))
    model = model_for(None)
    model.set_params(random_state=seed, n_estimators=256, min_samples_leaf=3,
                     max_features=0.7, class_weight="balanced", n_jobs=4)
    fit(model, augmented, augmented_labels)
    return model, names


def _predict_one(model: object, x: np.ndarray, names: list[str],
                 availability: np.ndarray, *, type_agnostic: bool) -> np.ndarray:
    if type_agnostic:
        x, names = aggregate_type_agnostic(x, names)
    return predict(model, apply_availability(x, names, availability, append_flags=True))


def train(dataset: str, fold: int, seed: int, *, smoke: bool = False,
          output: Path | None = None) -> Path:
    started = time.perf_counter()
    start_time = dt.datetime.now(dt.timezone.utc).isoformat()
    data = load_fold(dataset, fold, smoke=False)
    evidence = ROOT / f"cache/fold_evidence_v2/{dataset}/fold{fold}"
    evidence_meta = json.loads((evidence / "metadata.json").read_text())
    with np.load(evidence / "features.npz") as archive:
        x = archive["x"]
        ids = archive["incident_ids"].tolist()
    all_ids = data.train_ids + data.test_ids
    if ids != all_ids or evidence_meta["train_ids"] != data.train_ids or evidence_meta["test_ids"] != data.test_ids:
        raise ValueError("evidence cache does not match the frozen fold")
    for relative, digest in evidence_meta["input_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"evidence source changed: {relative}")
    names = evidence_meta["feature_names"]
    if x.shape != (len(all_ids), len(data.services), len(names)) or not np.isfinite(x).all():
        raise ValueError("invalid S0 evidence tensor")
    train_count = len(data.train_ids)
    labels = data.train_labels
    if len(labels) != train_count:
        raise ValueError("training labels do not align with evidence rows")
    if output is None:
        stage = "T1_evidence_smoke" if smoke else "T1_evidence_full"
        output = ROOT / "results_v2" / stage / dataset / f"fold{fold}_seed{seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite evidence run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    rows = [{"incident_id": incident_id,
             "fault_type": data.train_fault_types[i],
             "overlap_group": data.train_groups[i]}
            for i, incident_id in enumerate(data.train_ids)]
    train_indices = np.arange(train_count)
    availability = data.availability.astype(np.float32)
    held_specific, held_generic, held_labels = [], [], []
    inner = []
    for fault_type in sorted(set(data.train_fault_types)):
        fit_indices, held_indices, purged, groups = fault_type_holdout(
            rows, train_indices, fault_type)
        if not len(fit_indices) or not len(held_indices):
            raise ValueError(f"empty inner fault-type holdout: {fault_type}")
        specific, _ = _fit_one(x, names, labels, availability, fit_indices, seed,
                               type_agnostic=False)
        generic, _ = _fit_one(x, names, labels, availability, fit_indices, seed,
                              type_agnostic=True)
        held_specific.append(_predict_one(specific, x[held_indices], names,
                                          availability[held_indices], type_agnostic=False))
        held_generic.append(_predict_one(generic, x[held_indices], names,
                                         availability[held_indices], type_agnostic=True))
        held_labels.append(labels[held_indices])
        inner.append({"fault_type": fault_type, "fit_ids": [data.train_ids[i] for i in fit_indices],
                      "held_ids": [data.train_ids[i] for i in held_indices],
                      "purged_ids": [data.train_ids[i] for i in purged],
                      "held_groups": groups})
    specific_cv = np.concatenate(held_specific)
    generic_cv = np.concatenate(held_generic)
    labels_cv = np.concatenate(held_labels)
    grid = []
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = fuse_all((1 - beta, specific_cv), (beta, generic_cv))
        metrics = ranking_metrics(_ranks(scores, labels_cv))
        grid.append({"beta": beta, "metrics": metrics})
    selected = max(grid, key=lambda row: (row["metrics"]["PR@1"],
                                          row["metrics"]["MAP@5"], -row["beta"]))
    beta = float(selected["beta"])
    specific, _ = _fit_one(x, names, labels, availability, train_indices, seed,
                           type_agnostic=False)
    generic, _ = _fit_one(x, names, labels, availability, train_indices, seed,
                          type_agnostic=True)
    indices = data.test_indices if not smoke else data.test_indices[:5]
    specific_scores = _predict_one(specific, x[indices], names,
                                   availability[indices], type_agnostic=False)
    generic_scores = _predict_one(generic, x[indices], names,
                                  availability[indices], type_agnostic=True)
    evidence_scores = fuse_all((1 - beta, specific_scores), (beta, generic_scores))
    if not np.isfinite(evidence_scores).all() or np.any(evidence_scores < 0):
        raise ValueError("invalid evidence scores")
    joblib.dump(specific, output / "specific.joblib", compress=3)
    joblib.dump(generic, output / "generic.joblib", compress=3)
    np.savez_compressed(output / "scores.npz", incident_ids=np.asarray([all_ids[i] for i in indices]),
                        specific=specific_scores, generic=generic_scores,
                        probabilities=evidence_scores)
    json_dump(output / "inner_beta.json", {"grid": grid, "selected_beta": beta,
                                           "holdouts": inner})
    meta = config_snapshot(
        f"python -m exp_revision.train_evidence --dataset {dataset} --fold {fold} --seed {seed}{' --smoke' if smoke else ''}",
        {"method": "ARISE-evidence", "dataset": dataset, "fold": fold,
         "seed": seed, "smoke": smoke, "beta_candidates": [0, 0.25, 0.5, 0.75, 1],
         "alpha": 0.75, "trees": 256, "min_samples_leaf": 3,
         "max_features": 0.7, "max_depth": None,
         "class_weight": "balanced", "threads": 4,
         "partial_training_rates": [0.25, 0.5, 0.75],
         "append_availability_flags": True,
         "availability_policy": "raw_A0_intersect_augmentation_view"},
    )
    source_paths = [ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}/features.npz",
                    ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}/metadata.json",
                    evidence / "features.npz", evidence / "metadata.json",
                    ROOT / f"splits/{dataset}_training_labels/fold_{fold}.json",
                    ROOT / f"splits/{dataset}_kfold5.json",
                    ROOT / "exp_revision/train_evidence.py", ROOT / "exp_revision/fold_data.py",
                    ROOT / "scripts/care/robust_evidence.py", ROOT / "scripts/care/evidence_reranker.py",
                    ROOT / "configs/care/review-response-v1.json"]
    meta.update(status="partial_T1_evidence_S0", start_time=start_time,
                selected_beta=beta, fit_count=train_count, test_count=len(indices),
                elapsed_seconds=time.perf_counter()-started,
                end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
                scores_sha256=sha256(output / "scores.npz"),
                inner_beta_sha256=sha256(output / "inner_beta.json"),
                specific_model_sha256=sha256(output / "specific.joblib"),
                generic_model_sha256=sha256(output / "generic.joblib"))
    json_dump(output / "run_meta.json", meta)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(train(args.dataset, args.fold, args.seed, smoke=args.smoke, output=args.output))


if __name__ == "__main__":
    main()
