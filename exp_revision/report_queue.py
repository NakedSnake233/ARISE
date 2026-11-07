"""Register versioned, budget-accounted statistical report jobs.

The queue completion archive records row counts, not diagnostic service scores.
"""
from __future__ import annotations
import argparse
import hashlib
import fcntl
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import ROOT, config_snapshot, json_dump, sha256
from . import remaining_queue as q
from . import report
from .score_receipts import check_hashes


def create(smoke=False):
    q.FOLDER.mkdir(parents=True, exist_ok=True)
    with (q.FOLDER / ".report_manifest.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return None
        return _create(smoke)


def _create(smoke=False):
    state = json.loads((q.FOLDER / "queue_state.json").read_text())
    reports = [(name, row) for name, row in state["jobs"].items() if row["spec"]["expected_config"].get("component") == "statistical_report"]
    if any(row["status"] in ("pending", "running") for _, row in reports): return None
    # A published manifest may not yet have been adopted by the scheduler.
    for path in q.MANIFESTS.glob("31_report_*.json"):
        if any(job["id"] not in state["jobs"] for job in json.loads(path.read_text())["jobs"]): return None
    if not smoke and not any(row["status"] == "completed" and row["spec"]["expected_config"]["smoke"] and q.verified(ROOT / row["output"], row["spec"]) for _, row in reports):
        raise ValueError("report smoke must pass before full statistics")
    tasks, inputs = [], {}
    for task in report.TASKS:
        key = "collect_" + report.COLLECTION[task]; row = state["jobs"].get(key)
        if row is None or row["status"] != "completed": continue
        if not q.verified(ROOT / row["output"], row["spec"]): raise ValueError("report collection receipt unverified")
        folder = ROOT / "results_v2" / task
        paths = [folder / "predictions.parquet", folder / "evaluation_meta.json",
            ROOT / row["output"] / "scores.npz", ROOT / row["output"] / "run_meta.json"]
        inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in paths}); tasks.append(task)
    if not {"T1", "T2"}.issubset(tasks): raise ValueError("main results not collected")
    main = state["jobs"]["collect_main"]
    for path in (ROOT / main["output"] / "products/run_meta.json", ROOT / "results_v2/T2/realized_rates.csv"):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    key = f"report_{'smoke' if smoke else 'full'}_{signature[:16]}"
    if key in state["jobs"]: return None
    path = ROOT / f"manifests/remaining/31_{key}.json"
    if path.exists(): return None
    sources = {**inputs, **{name: sha256(ROOT / name) for name in ("exp_revision/remaining_queue.py", "exp_revision/disk_guard.py",
        "exp_revision/disk_guard_local.py", "exp_revision/common.py", "exp_revision/score_receipts.py", "exp_revision/report.py",
        "exp_revision/report_queue.py", "exp_revision/report_env_spec.json")}}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = dict(task="T10", component="statistical_report", smoke=smoke, reported_tasks=tasks, input_signature=signature,
        witness="prediction row counts; not RCA scores")
    job = dict(id=key, source_commit=commit, resource="cpu", cores=1, timeout_seconds=600 if smoke else 1800, max_attempts=2,
        command=["-m", "exp_revision.report_queue", "--manifest", str(path.relative_to(ROOT))],
        expected_config=config, expected_status=["completed_statistical_report"], test_ids=tasks, score_shape=[1, len(tasks)])
    json_dump(path, dict(backend="local", phase=f"{'smoke' if smoke else 'full'}_statistical_report", source_commit=commit,
        source_sha256=sources, precharged_hours=q.PRECHARGED, jobs=[job], input_sha256=inputs))
    return path


def run(manifest_path, output):
    manifest = json.loads(manifest_path.read_text()); job = manifest["jobs"][0]; config = job["expected_config"]
    check_hashes(manifest["source_sha256"])
    spec = json.loads((ROOT / "exp_revision/report_env_spec.json").read_text())
    for package, version in spec["packages"].items():
        if importlib.metadata.version(package) != version: raise ValueError(f"report environment changed: {package}")
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable statistical report output exists")
    clock, began = time.perf_counter(), q.now(); output.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    products = output / f"report_{stamp}"
    result = report.generate(products, publish=False, smoke=config["smoke"], tasks=config["reported_tasks"])
    counts = []
    import pandas as pd
    for task in config["reported_tasks"]:
        frame = pd.read_parquet(ROOT / f"results_v2/{task}/predictions.parquet")
        counts.append(len(frame))
    np.savez_compressed(output / "scores.npz", scores=np.asarray([counts], dtype=float), incident_ids=config["reported_tasks"])
    files = {str(p.relative_to(ROOT)): sha256(p) for p in products.rglob("*") if p.is_file()}
    meta = config_snapshot("python -m exp_revision.report_queue", config)
    meta.update(status="prepared_statistical_report", errors=[], start_time=began, end_time=q.now(), elapsed_seconds=time.perf_counter()-clock,
        report_folder=str(products.relative_to(ROOT)), input_sha256=files,
        source_sha256={"exp_revision/report_queue.py": sha256(Path(__file__)), "exp_revision/report.py": sha256(ROOT / "exp_revision/report.py")},
        report_status=result["status"], scores_sha256=sha256(output / "scores.npz"))
    json_dump(output / "run_meta.json", meta)
    provisional = {**job, "expected_status": ["prepared_statistical_report"]}
    if not q.verified(output, provisional): raise ValueError("prepared statistical report failed verification")
    check_hashes(files)
    if not config["smoke"]: report.publish_report(products, result)
    meta.update(status="completed_statistical_report", end_time=q.now(), elapsed_seconds=time.perf_counter()-clock,
        publication="smoke archive only" if config["smoke"] else "verified artifacts published; latest pointer committed")
    json_dump(output / "run_meta.json", meta)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--create", action="store_true"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--manifest", type=Path); p.add_argument("--output", type=Path); a = p.parse_args()
    if a.create: print(create(a.smoke))
    else: run(ROOT / a.manifest, a.output)


if __name__ == "__main__": main()
