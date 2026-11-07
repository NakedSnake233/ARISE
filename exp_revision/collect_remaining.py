"""Accounted, guarded collectors for the complete reviewed evaluation matrices.

The top-level scores.npz is a row-count completion witness, never RCA scores.
Actual predictions and their receipts live in the products subdirectory.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from .common import ROOT, config_snapshot, json_dump, sha256
from . import remaining_queue as q
from .score_receipts import check_hashes

PHASES = {
    "main": ["02_full_baselines.json", "06_full_evidence_masks.json"],
    "T3": ["08_full_t3_evidence.json"],
    "T5": ["25_full_t5_composition.json"],
    "T6": ["10_full_t6_evidence.json", "13_full_context_variants.json", "23_full_t6_reuse.json"],
}
TARGETS = {"main": ["T1", "T2"], "T3": ["T3"], "T5": ["T5a", "T5b"], "T6": ["T6"]}


def publish(folder, task):
    canonical = ROOT / "results_v2" / task; canonical.mkdir(exist_ok=True)
    for name in ("predictions.parquet", "evaluation_meta.json"):
        target = canonical / name
        if target.exists():
            if sha256(target) != sha256(folder / name): raise FileExistsError("different canonical predictions already exist")
        else: os.link(folder / name, target)


def collect(kind, output):
    from .evaluate_score_batches import evaluate
    from .merge_main import merge
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable collection output exists")
    began, clock = q.now(), time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    products = output / "products"; products.mkdir(); receipts = {}
    if kind == "main":
        merge(products, publish=True)
    elif kind in ("T3", "T5"):
        manifest = ROOT / "manifests/remaining" / PHASES[kind][0]
        source = json.loads(manifest.read_text())
        for task in TARGETS[kind]:
            subset = {**source, "jobs": [j for j in source["jobs"] if j["expected_config"]["task"] == task]}
            path = products / f"{task}_evaluation_manifest.json"; json_dump(path, subset)
            folder = products / task; evaluate(path, folder); publish(folder, task)
    else:
        parts = []
        for i, name in enumerate(PHASES[kind]):
            folder = products / f"part{i}"; evaluate(ROOT / "manifests/remaining" / name, folder)
            parts.append(pd.read_parquet(folder / "predictions.parquet"))
        frame = pd.concat(parts, ignore_index=True)
        keys = ["dataset", "method", "incident_id", "seed", "scenario", "scenario_param"]
        if frame.duplicated(keys).any() or frame.method.nunique() != 15: raise ValueError("T6 ablation matrix duplicated or missing methods")
        from .train_variant_evidence import T6_SCENARIOS
        for dataset, n in (("tt", 150), ("aiops", 128)):
            for method in frame.method.unique():
                for seed in (42, 43, 44, 45, 46):
                    current = frame[(frame.dataset == dataset) & (frame.method == method) & (frame.seed == seed)]
                    if len(current) != n * len(T6_SCENARIOS): raise ValueError("T6 ablation denominator differs")
        folder = products / "T6"; folder.mkdir(); path = folder / "predictions.parquet"
        frame.sort_values(keys, kind="stable").reset_index(drop=True).to_parquet(path, index=False)
        json_dump(folder / "evaluation_meta.json", dict(task="T6", rows=len(frame), failed=int((frame.status == "failed").sum()),
            predictions_sha256=sha256(path), preregistered=False, part_receipts={str((products / f"part{i}/evaluation_meta.json").relative_to(ROOT)):
                sha256(products / f"part{i}/evaluation_meta.json") for i in range(3)}))
        publish(folder, "T6")
    for task in TARGETS[kind]:
        folder = products / task; receipt = json.loads((folder / "evaluation_meta.json").read_text())
        receipts[task] = receipt
    np.savez_compressed(output / "scores.npz", scores=np.asarray([[receipts[t]["rows"] for t in TARGETS[kind]]], dtype=float), incident_ids=TARGETS[kind])
    config = dict(task="T10", component="artifact_collection", collection=kind, targets=TARGETS[kind], witness="prediction row counts; not RCA scores")
    meta = config_snapshot("python -m exp_revision.collect_remaining", config)
    inputs = {str(p.relative_to(ROOT)): sha256(p) for task in TARGETS[kind] for p in (products / task / "predictions.parquet", products / task / "evaluation_meta.json")}
    meta.update(status="completed_artifact_collection", errors=[], start_time=began, end_time=q.now(), elapsed_seconds=time.perf_counter()-clock,
        products=receipts, input_sha256=inputs, source_sha256={"exp_revision/collect_remaining.py": sha256(Path(__file__))},
        scores_sha256=sha256(output / "scores.npz"))
    json_dump(output / "run_meta.json", meta)


def create(kind):
    state = json.loads((q.FOLDER / "queue_state.json").read_text()); sources = {}; paths = [ROOT / "manifests/remaining" / p for p in PHASES[kind]]
    for path in paths:
        manifest = json.loads(path.read_text()); check_hashes(manifest["source_sha256"])
        if any(state["jobs"][j["id"]]["status"] != "completed" for j in manifest["jobs"]): raise ValueError("collection prerequisites incomplete")
        sources.update(manifest["source_sha256"]); sources[str(path.relative_to(ROOT))] = sha256(path)
    for name in ("exp_revision/collect_remaining.py", "exp_revision/merge_main.py", "exp_revision/score_receipts.py", "exp_revision/evaluate_score_batches.py"):
        sources[name] = sha256(ROOT / name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = dict(task="T10", component="artifact_collection", collection=kind, targets=TARGETS[kind], witness="prediction row counts; not RCA scores")
    job = dict(id=f"collect_{kind}", source_commit=commit, resource="cpu", cores=1, timeout_seconds=1800, max_attempts=2,
        command=["-m", "exp_revision.collect_remaining", "--collection", kind], expected_status=["completed_artifact_collection"],
        expected_config=config, test_ids=TARGETS[kind], score_shape=[1, len(TARGETS[kind])])
    path = ROOT / f"manifests/remaining/30_collect_{kind}.json"
    if path.exists(): raise FileExistsError("immutable collection manifest exists")
    json_dump(path, dict(backend="local", phase=f"collect_{kind}", source_commit=commit, precharged_hours=q.PRECHARGED, source_sha256=sources, jobs=[job]))
    return path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--create", action="store_true"); p.add_argument("--collection", choices=PHASES, required=True)
    p.add_argument("--output", type=Path); a = p.parse_args()
    if a.create: print(create(a.collection))
    else: collect(a.collection, a.output)


if __name__ == "__main__": main()
