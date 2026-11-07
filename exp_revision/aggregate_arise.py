"""Fuse verified context and evidence scores into full-service S0 ARISE ranks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, SEEDS, json_dump, load_bundle, sha256
from .context_queue import FOLDS, _verified_output as verify_context, verify_manifest as context_manifest
from .evidence_queue import _verified as verify_evidence, verify_manifest as evidence_manifest
from .fold_data import load_fold


def _read_state(dataset: str, path: Path, manifest_path: Path) -> dict:
    state = json.loads(path.read_text())
    expected_jobs = {f"{dataset}_fold{fold}_seed{seed}" for fold in FOLDS for seed in SEEDS}
    if (state["status"] != "completed" or state["manifest_sha256"] != sha256(manifest_path)
            or set(state["jobs"]) != expected_jobs
            or any(job["status"] != "completed" for job in state["jobs"].values())):
        raise ValueError(f"incomplete or mismatched queue state: {path}")
    return state


def _fuse(context: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    if context.shape != evidence.shape or context.ndim != 2:
        raise ValueError("context and evidence score shapes differ")
    if not (np.isfinite(context).all() and np.isfinite(evidence).all()
            and (context >= 0).all() and (evidence >= 0).all()):
        raise ValueError("invalid branch score")
    from .train_evidence import fuse_all
    return fuse_all((0.25, context), (0.75, evidence))


def aggregate(dataset: str, context_manifest_path: Path,
              evidence_manifest_path: Path,
              context_root: Path = ROOT / "results_v2/T1_context_full") -> dict:
    context = json.loads(context_manifest_path.read_text())
    evidence = json.loads(evidence_manifest_path.read_text())
    if context["dataset"] != dataset or evidence["dataset"] != dataset:
        raise ValueError("fusion manifests refer to another dataset")
    context_folds = context_manifest(context)
    evidence_folds = evidence_manifest(evidence)
    context_state = _read_state(
        dataset, context_root / dataset / "queue_state.json",
        context_manifest_path)
    evidence_state = _read_state(
        dataset, ROOT / f"results_v2/T1_evidence_full/{dataset}/queue_state.json",
        evidence_manifest_path)
    bundle = load_bundle(dataset)  # first point that reads held-out root/fault labels
    root_by_id = {row["incident_id"]: int(bundle.labels[i])
                  for i, row in enumerate(bundle.rows)}
    fault_by_id = {row["incident_id"]: row["fault_type"] for row in bundle.rows}
    if len(root_by_id) != len(bundle.rows):
        raise ValueError("duplicate evaluation incident ID")
    rows = []
    for fold in FOLDS:
        data = load_fold(dataset, fold, smoke=False)
        if data.test_ids != context_folds[fold].test_ids or data.test_ids != evidence_folds[fold].test_ids:
            raise ValueError("branch test folds differ")
        if data.services != bundle.services:
            raise ValueError("service order differs from evaluation cohort")
        for seed in SEEDS:
            job_id = f"{dataset}_fold{fold}_seed{seed}"
            context_output = ROOT / context_state["jobs"][job_id]["output"]
            evidence_output = ROOT / evidence_state["jobs"][job_id]["output"]
            if not verify_context(context_output, context, fold, seed, data):
                raise ValueError(f"invalid saved context branch: {job_id}")
            if not verify_evidence(evidence_output, evidence, fold, seed, data):
                raise ValueError(f"invalid saved evidence branch: {job_id}")
            with np.load(context_output / "scores.npz") as archive:
                if archive["incident_ids"].tolist() != data.train_ids + data.test_ids:
                    raise ValueError("context incident order differs from fold")
                base_scores = archive["probabilities"][len(data.train_ids):]
            with np.load(evidence_output / "scores.npz") as archive:
                if archive["incident_ids"].tolist() != data.test_ids:
                    raise ValueError("evidence scores are not test-only")
                evidence_scores = archive["probabilities"]
            fused = _fuse(base_scores, evidence_scores)
            if fused.shape != (len(data.test_ids), len(data.services)):
                raise ValueError("fusion shape mismatch")
            destination = ROOT / f"results_v2/T1_arise_s0/{dataset}/{job_id}"
            destination.mkdir(parents=True, exist_ok=True)
            score_path = destination / "scores.npz"
            if score_path.exists():
                with np.load(score_path) as archive:
                    if (archive["incident_ids"].tolist() != data.test_ids
                            or not np.array_equal(archive["probabilities"], fused)):
                        raise FileExistsError("refusing to replace a different fused score")
            else:
                temporary = score_path.with_name(f".scores.tmp-{os.getpid()}.npz")
                np.savez_compressed(temporary, incident_ids=np.asarray(data.test_ids),
                                    probabilities=fused)
                temporary.replace(score_path)
            beta = json.loads((evidence_output / "run_meta.json").read_text())["selected_beta"]
            meta_path = destination / "run_meta.json"
            meta = {"task": "T1", "method": "ARISE", "dataset": dataset,
                    "fold": fold, "seed": seed, "scenario": "S0",
                    "alpha_evidence": 0.75, "beta": beta, "status": "completed",
                    "created": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "context_source_commit": context["source_commit"],
                    "evidence_source_commit": evidence["source_commit"],
                    "context_scores_sha256": sha256(context_output / "scores.npz"),
                    "evidence_scores_sha256": sha256(evidence_output / "scores.npz"),
                    "scores_sha256": sha256(score_path)}
            if meta_path.exists():
                old = json.loads(meta_path.read_text())
                if any(old.get(key) != value for key, value in meta.items() if key != "created"):
                    raise FileExistsError("refusing to replace a different fusion record")
            else:
                json_dump(meta_path, meta)
            for i, incident_id in enumerate(data.test_ids):
                score = fused[i]
                valid = (np.isfinite(score).all() and (score >= 0).all()
                         and np.isclose(score.sum(), 1.0, atol=1e-6))
                if valid:
                    order = np.argsort(-score, kind="stable")
                    rank = float(np.flatnonzero(order == root_by_id[incident_id])[0] + 1)
                    top5 = [data.services[int(j)] for j in order[:5]]
                else:
                    rank, top5 = float("inf"), []
                rows.append({"task": "T1", "method": "ARISE", "dataset": dataset,
                             "split": "kfold5", "fold": fold, "seed": seed,
                             "scenario": "S0", "scenario_param": "",
                             "realized_rate": 0.0, "incident_id": incident_id,
                             "fault_type": fault_by_id[incident_id],
                             "root_true": data.services[root_by_id[incident_id]],
                             "rank_of_root": rank,
                             "top5": json.dumps(top5, ensure_ascii=False),
                             "status": "ok" if valid else "failed",
                             "invalid_by_design": False, "runtime_ms": None})
    frame = pd.DataFrame(rows)
    expected = Counter((row["incident_id"], seed)
                       for row in bundle.rows for seed in SEEDS)
    observed = Counter(zip(frame["incident_id"], frame["seed"]))
    if observed != expected:
        raise ValueError("ARISE test events are missing or duplicated")
    failed = int((frame["status"] != "ok").sum())
    if failed / len(frame) > 0.1:
        raise RuntimeError("ARISE failure rate exceeds 10%")
    destination = ROOT / f"results_v2/T1_arise_s0/{dataset}/predictions.parquet"
    if destination.exists():
        if not pd.read_parquet(destination).equals(frame):
            raise FileExistsError("refusing to replace different ARISE predictions")
    else:
        temporary = destination.with_name(f".predictions.tmp-{os.getpid()}.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    summary = {"status": "completed_T1_arise_S0",
               "method": "ARISE", "dataset": dataset, "scenario": "S0",
               "rows": len(frame), "events": len(bundle.rows), "failed": failed,
               "predictions_sha256": sha256(destination),
               "context_manifest_sha256": sha256(context_manifest_path),
               "evidence_manifest_sha256": sha256(evidence_manifest_path)}
    json_dump(ROOT / f"results_v2/T1_arise_s0/{dataset}/aggregate_meta.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, default=ROOT / "results_v2/T1_context_full")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.dataset, args.context_manifest,
                               args.evidence_manifest, args.context_root)))


if __name__ == "__main__":
    main()
