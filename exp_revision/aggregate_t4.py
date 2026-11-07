"""Combine verified S0 sanity baselines on the shared grouped folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, SEEDS, json_dump, load_bundle, sha256, stable_seed
from .evidence_queue import _verified as verify_evidence, verify_manifest
from .et_spec_matrix import _verified as verify_et, verify_manifest as verify_et_manifest


def _evidence_config(dataset: str, fold: int, seed: int) -> dict:
    return {"alpha": 0.75, "append_availability_flags": True,
            "availability_policy": "raw_A0_intersect_augmentation_view",
            "beta_candidates": [0, 0.25, 0.5, 0.75, 1],
            "class_weight": "balanced", "dataset": dataset, "fold": fold,
            "max_depth": None, "max_features": 0.7,
            "method": "ARISE-evidence", "min_samples_leaf": 3,
            "partial_training_rates": [0.25, 0.5, 0.75],
            "seed": seed, "smoke": False, "threads": 4, "trees": 256}


def _validate_sanity(dataset: str, frame: pd.DataFrame, meta: dict,
                     folds: dict, bundle) -> None:
    config = {"methods": ["RootFreqPrior", "Random"],
              "scenario": "S0", "seeds": list(SEEDS)}
    if (meta["status"] != "partial_T4_sanity_only" or meta["config"] != config
            or meta["config_sha256"] != hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest()
            or meta["command"] != f"python -m exp_revision.run --task T4 --dataset {dataset}"
            or meta["prediction_rows"] != 2 * len(bundle.rows) * len(SEEDS)):
        raise ValueError("Random/RootFreqPrior run is not the full frozen S0 run")
    # The legacy sanity metadata has no source hashes. Verify its recorded
    # revision, then reconstruct every deterministic ranking below.
    for relative in ("exp_revision/sanity_baselines.py", "exp_revision/common.py",
                     "exp_revision/splits.py", "exp_revision/metrics.py",
                     "exp_revision/runner.py"):
        saved_source = subprocess.check_output(
            ["git", "show", f"{meta['git_commit']}:{relative}"], cwd=ROOT)
        if saved_source != (ROOT / relative).read_bytes():
            raise ValueError(f"sanity source differs from recorded revision: {relative}")
    id_to_index = {row["incident_id"]: index for index, row in enumerate(bundle.rows)}
    test_fold = {incident_id: fold for fold, data in folds.items()
                 for incident_id in data.test_ids}
    priors = {}
    for fold, data in folds.items():
        labels = bundle.labels[[id_to_index[incident_id] for incident_id in data.train_ids]]
        if not np.array_equal(labels, data.train_labels):
            raise ValueError("sanity training labels differ from frozen fold")
        priors[fold] = np.argsort(-np.bincount(labels, minlength=bundle.n_services),
                                  kind="stable")
    expected = Counter((method, row["incident_id"], seed)
                       for method in ("RootFreqPrior", "Random")
                       for row in bundle.rows for seed in SEEDS)
    if Counter(zip(frame["method"], frame["incident_id"], frame["seed"])) != expected:
        raise ValueError("Random/RootFreqPrior matrix is incomplete or duplicated")
    for row in frame.itertuples(index=False):
        fold = test_fold[row.incident_id]
        if (row.task != "T4_sanity" or row.dataset != dataset or row.split != "kfold5"
                or row.fold != fold or row.scenario != "S0" or row.scenario_param != ""
                or row.realized_rate != 0 or row.status != "ok"
                or row.invalid_by_design or not pd.isna(row.runtime_ms)):
            raise ValueError(f"invalid sanity row provenance: {row.incident_id}")
        order = (priors[fold] if row.method == "RootFreqPrior" else
                 np.random.default_rng(stable_seed(dataset, row.incident_id,
                                                   "Random", row.seed)).permutation(bundle.n_services))
        index = id_to_index[row.incident_id]
        root = int(bundle.labels[index])
        rank = float(np.flatnonzero(order == root)[0] + 1)
        top5 = [bundle.services[int(service)] for service in order[:5]]
        if (row.root_true != bundle.services[root]
                or row.fault_type != bundle.rows[index]["fault_type"]
                or row.rank_of_root != rank or json.loads(row.top5) != top5):
            raise ValueError(f"sanity ranking or ground truth differs: {row.incident_id}")


def aggregate(dataset: str) -> dict:
    manifest_path = ROOT / f"manifests/{dataset}_evidence_full.json"
    manifest = json.loads(manifest_path.read_text())
    folds = verify_manifest(manifest)
    evidence_folder = ROOT / "results_v2/T1_evidence_full" / dataset
    state = json.loads((evidence_folder / "queue_state.json").read_text())
    et_manifest_path = ROOT / f"manifests/{dataset}_et_spec_full.json"
    et_manifest = json.loads(et_manifest_path.read_text())
    et_folds = verify_et_manifest(et_manifest)
    et_state = json.loads((ROOT / f"results_v2/T4_et_spec_full/{dataset}/queue_state.json").read_text())
    job_ids = {f"{dataset}_fold{fold}_seed{seed}" for fold in range(5) for seed in SEEDS}
    if (state["status"] != "completed" or state["manifest_sha256"] != sha256(manifest_path)
            or set(state["jobs"]) != job_ids
            or any(row["status"] != "completed" for row in state["jobs"].values())):
        raise ValueError("evidence expert queue is incomplete")
    if (et_state["status"] != "completed" or et_state["manifest_sha256"] != sha256(et_manifest_path)
            or set(et_state["jobs"]) != job_ids
            or any(row["status"] != "completed" for row in et_state["jobs"].values())):
        raise ValueError("ET-spec queue is incomplete")
    verified: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for fold in range(5):
        data = folds[fold]
        if data.test_ids != et_folds[fold].test_ids or data.services != et_folds[fold].services:
            raise ValueError("ET-spec and gated experts use different outer folds")
        for seed in SEEDS:
            job_id = f"{dataset}_fold{fold}_seed{seed}"
            et_folder = ROOT / et_state["jobs"][job_id]["output"]
            if not verify_et(et_folder, et_manifest, fold, seed, data):
                raise ValueError(f"invalid ET-spec output: {job_id}")
            et_meta = json.loads((et_folder / "run_meta.json").read_text())
            if (et_meta["status"] != "partial_T4_ET_spec_S0"
                    or (et_meta["config"]["dataset"], et_meta["config"]["fold"],
                        et_meta["config"]["seed"], et_meta["config"]["smoke"])
                    != (dataset, fold, seed, False)
                    or et_meta["scores_sha256"] != sha256(et_folder / "scores.npz")
                    or et_meta["model_sha256"] != sha256(et_folder / "model.joblib")):
                raise ValueError(f"invalid ET-spec run: {job_id}")
            if any(sha256(ROOT / relative) != digest
                   for relative, digest in et_meta["input_sha256"].items()):
                raise ValueError(f"ET-spec source changed: {job_id}")
            with np.load(et_folder / "scores.npz") as archive:
                if set(archive.files) != {"incident_ids", "probabilities"} or archive["incident_ids"].tolist() != data.test_ids:
                    raise ValueError(f"ET-spec score order differs: {job_id}")
                et_scores = archive["probabilities"]
            evidence_output = ROOT / state["jobs"][job_id]["output"]
            if not verify_evidence(evidence_output, manifest, fold, seed, data):
                raise ValueError(f"invalid gated expert output: {job_id}")
            evidence_meta = json.loads((evidence_output / "run_meta.json").read_text())
            if evidence_meta["config"] != _evidence_config(dataset, fold, seed):
                raise ValueError(f"evidence configuration does not support gate+aug: {job_id}")
            with np.load(evidence_output / "scores.npz") as archive:
                gated_scores = archive["specific"]
            for name, score in (("ET-spec", et_scores),
                                ("ET-spec+gate+aug", gated_scores)):
                if (score.shape != (len(data.test_ids), len(data.services))
                        or not np.isfinite(score).all() or (score < 0).any()
                        or not np.allclose(score.sum(axis=1), 1, atol=1e-5)):
                    raise ValueError(f"invalid {name} full-service score: {job_id}")
            verified[(fold, seed)] = (et_scores, gated_scores)
    bundle = load_bundle(dataset)
    if len({row["incident_id"] for row in bundle.rows}) != len(bundle.rows):
        raise ValueError("duplicate evaluation incident ID")
    if any(fold.services != bundle.services for fold in folds.values()):
        raise ValueError("service order differs from evaluation labels")
    root_by_id = {row["incident_id"]: int(bundle.labels[i]) for i, row in enumerate(bundle.rows)}
    fault_by_id = {row["incident_id"]: row["fault_type"] for row in bundle.rows}
    rows = []
    for fold, data in folds.items():
        for seed in SEEDS:
            for method, scores in zip(("ET-spec", "ET-spec+gate+aug"),
                                      verified[(fold, seed)]):
                for index, incident_id in enumerate(data.test_ids):
                    root = root_by_id[incident_id]
                    order = np.argsort(-scores[index], kind="stable")
                    rows.append({"task": "T4", "method": method,
                                 "dataset": dataset, "split": "kfold5",
                                 "fold": fold, "seed": seed, "scenario": "S0",
                                 "scenario_param": "", "realized_rate": 0.0,
                                 "incident_id": incident_id,
                                 "fault_type": fault_by_id[incident_id],
                                 "root_true": data.services[root],
                                 "rank_of_root": float(np.flatnonzero(order == root)[0] + 1),
                                 "top5": json.dumps([data.services[int(j)] for j in order[:5]],
                                                    ensure_ascii=False),
                                 "status": "ok", "invalid_by_design": False,
                                 "runtime_ms": None})
    sanity_folder = ROOT / f"results_v2/T4_sanity/{dataset}"
    sanity = pd.read_parquet(sanity_folder / "predictions.parquet")
    _validate_sanity(dataset, sanity, json.loads((sanity_folder / "run_meta.json").read_text()),
                     folds, bundle)
    sanity = sanity.copy()
    sanity["task"] = "T4"
    result = pd.concat((sanity, pd.DataFrame(rows)), ignore_index=True)
    methods = ("Random", "RootFreqPrior", "ET-spec", "ET-spec+gate+aug")
    expected = Counter((method, row["incident_id"], seed)
                       for method in methods for row in bundle.rows for seed in SEEDS)
    observed = Counter(zip(result["method"], result["incident_id"], result["seed"]))
    if observed != expected:
        raise ValueError("T4 sanity result matrix is incomplete")
    destination = ROOT / "results_v2/T4_full" / dataset
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "predictions.parquet"
    if path.exists():
        if not pd.read_parquet(path).equals(result):
            raise FileExistsError("refusing to replace different T4 predictions")
    else:
        temporary = path.with_suffix(".tmp.parquet")
        result.to_parquet(temporary, index=False)
        temporary.replace(path)
    meta = {"status": "partial_T4_S0_only", "dataset": dataset,
            "methods": list(methods), "events": len(bundle.rows),
            "rows": len(result), "failed": int((result["status"] != "ok").sum()),
            "predictions_sha256": sha256(path)}
    meta.update(evidence_manifest_sha256=sha256(manifest_path),
                et_spec_manifest_sha256=sha256(et_manifest_path),
                random_prior_predictions_sha256=sha256(sanity_folder / "predictions.parquet"),
                random_prior_meta_sha256=sha256(sanity_folder / "run_meta.json"))
    json_dump(destination / "aggregate_meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("tt", "aiops"))
    args = parser.parse_args()
    print(json.dumps(aggregate(args.dataset)))


if __name__ == "__main__":
    main()
