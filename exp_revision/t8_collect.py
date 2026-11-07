"""Guarded complete-group collection; original results and collectors stay fixed."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
from .common import ROOT, config_snapshot, dataset_dir, json_dump, sha256
from . import remaining_queue as q
from . import t8_admission as admission
from . import t8_manifest as original
from . import re2_data as data
from . import t8_group_evaluate as evaluation


def kind(manifest):
    components = {j["expected_config"]["component"] for j in manifest["jobs"]}
    if components == {"official_baseline"}: return "official"
    if components == {"llm_baseline"}: return "llm"
    return "models"


def create(source):
    path = q.MANIFESTS / ("60_collect_" + Path(source).stem + ".json")
    if path.exists() or not original.complete(source): return None
    manifest = json.loads((q.MANIFESTS / source).read_text())
    if any(j["expected_config"].get("smoke") for j in manifest["jobs"]): return None
    group_kind = kind(manifest)
    cfg = dict(task="T8", component="extension_collection", source_manifest=source, kind=group_kind, smoke=False)
    job = dict(id="collect_t8_" + Path(source).stem, resource="cpu", cores=1, max_attempts=2, timeout_seconds=300,
        source_commit=admission.commit(), expected_config=cfg,
        command=["-m", "exp_revision.t8_collect", "--manifest", source], test_ids=["evaluated_rows"],
        score_shape=[1, 1, 1], expected_status=["completed_t8_collection"])
    sources = dict(manifest["source_sha256"]); current = admission.state()
    for job_spec in manifest["jobs"]:
        folder = ROOT / current["jobs"][job_spec["id"]]["output"]
        for name in ("scores.npz", "run_meta.json"): sources[str((folder / name).relative_to(ROOT))] = sha256(folder / name)
    for name in ("exp_revision/t8_collect.py", "exp_revision/t8_group_evaluate.py", "exp_revision/t8_evaluate.py", "exp_revision/metrics.py"):
        sources[name] = sha256(ROOT / name)
    sources[str((q.MANIFESTS / source).relative_to(ROOT))] = sha256(q.MANIFESTS / source)
    for ds in {j["expected_config"]["dataset"] for j in manifest["jobs"]}:
        paths = [ROOT / f"splits/{ds}_evaluation_labels.json", ROOT / "cache/rcaeval_re2/cases.parquet"] if ds in data.DATASETS else [dataset_dir(ds) / "incidents.json", dataset_dir(ds) / "metadata.json"]
        for p in paths: sources[str(p.relative_to(ROOT))] = sha256(p)
    for p in (ROOT / "results_v2/T2/realized_rates.csv", ROOT / "results_v2/T8_inputs/realized_rates.csv"):
        sources[str(p.relative_to(ROOT))] = sha256(p)
    return admission.write_manifest(path, "collect_t8_" + Path(source).stem, [job], sources)


def run(source, output):
    start = dt.datetime.now(dt.timezone.utc).isoformat(); clock = time.perf_counter()
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((q.MANIFESTS / source).read_text()); group_kind = kind(manifest)
    evaluation.evaluate(group_kind, output / "products", source)
    audit = json.loads((output / "products/evaluation_meta.json").read_text())
    np.savez_compressed(output / "scores.npz", scores=np.asarray([[[float(audit["rows"])]]]), incident_ids=["evaluated_rows"])
    meta = config_snapshot("python -m exp_revision.t8_collect", dict(task="T8", component="extension_collection",
        source_manifest=source, kind=group_kind, smoke=False))
    meta.update(status="completed_t8_collection", errors=[], start_time=start,
        end_time=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-clock,
        scores_sha256=sha256(output / "scores.npz"), artifact_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (output / "products").iterdir() if p.is_file()},
        witness_type="evaluated-row count, not an RCA score")
    json_dump(output / "run_meta.json", meta)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__": main()
