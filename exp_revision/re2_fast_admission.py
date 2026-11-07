"""Independent dataset preprocessing and explicit exact-parser recovery."""
from __future__ import annotations
import json

from .common import ROOT, sha256
from . import remaining_queue as q
from . import re2_data as data
from . import re2_manifest
from . import t8_admission as admission
from . import re2_fast_fold as fast


def closure():
    sources = re2_manifest.closure()
    for name in ("exp_revision/re2_fast_fold.py", "exp_revision/re2_fast_admission.py",
            "exp_revision/drain_bitset_compact.py", "exp_revision/drain_bitset_compact_spec.json",
            "exp_revision/re2_preprocessing_runtime.py", "exp_revision/extended_queue_v2.py",
            "results_v2/T8_drain_bitset_compact_witness/witness.json",
            "results_v2/T8_drain_bitset_compact_followdoc/witness.json"):
        sources[name] = sha256(ROOT / name)
    return sources


def create_recovery():
    path = q.MANIFESTS / "56_re2_exact_drain_recovery.json"
    if path.exists(): return None
    fast.ready(); current = admission.state(); name = "recovered_re2_fold_re2_ss_f0"
    row = current["jobs"].get(name)
    if not row or row["status"] == "completed" or not row["attempts"]: return None
    if any("Traceback (most recent call last)" in q.log_tail(ROOT / a["log"]) for a in row["attempts"]):
        raise ValueError("cannot treat a preprocessing code error as exact-parser recovery")
    if row["status"] == "running":
        if not q.alive(row["attempts"][-1]): return None
    elif row["status"] not in ("pending", "stuck"): return None
    spec = row["spec"]; cfg = spec["expected_config"]
    job = {**spec, "id": "fast_" + name, "source_commit": admission.commit(),
        "command": ["-m", "exp_revision.re2_fast_fold", "--dataset", cfg["dataset"],
            "--fold", str(cfg["fold"]), "--original-id", name]}
    sources = closure()
    for p in q.MANIFESTS.glob("*.json"):
        m = json.loads(p.read_text())
        if any(j["id"] == name for j in m["jobs"]):
            sources.update(m["source_sha256"]); sources[str(p.relative_to(ROOT))] = sha256(p)
    recipes = {name: dict(original_spec=spec, recovery_job=job["id"],
        preserved_attempts=row["attempts"], reason="full SS parser fit stalled; independently verified exact acceleration")}
    # Recovery changes execution speed only. Its deadline remains the existing
    # 7200 seconds, and all original attempts retain their actual compute cost.
    from .common import json_dump
    json_dump(path, dict(backend="local", phase="re2_exact_drain_recovery", source_commit=admission.commit(),
        precharged_hours=q.PRECHARGED, source_sha256=sources, jobs=[job], recovery_recipes=recipes))
    return path


def create_dataset(dataset):
    path = q.MANIFESTS / f"45_full_re2_fold_{dataset}.json"
    if path.exists() or not admission.verified_job(f"re2_fold_{dataset}_f0"): return None
    fast.ready(); current = admission.state(); completed = current["jobs"][f"re2_fold_{dataset}_f0"]
    pilot = completed["spec"]; receipt = json.loads((ROOT / completed["output"] / "run_meta.json").read_text())
    actual_job = current["jobs"][completed.get("completion_job", pilot["id"])]
    elapsed = max(receipt["elapsed_seconds"], actual_job["attempts"][-1]["elapsed_seconds"])
    # Do not extrapolate the small independent witness speedup. Use the full
    # fold measurement conservatively, even when its parser was slower.
    estimate = elapsed * 1.25 / 3600
    projection = dict(kind="preprocessing", forecast_hours=dict(cpu=4*estimate, gpu=0.),
        details=[dict(dataset=dataset, component="fold_preprocessing", per_job_hours=estimate, smoke_seconds=elapsed)],
        basis="same-system full fold-0 preprocessing; four additional folds plus25% margin, no extrapolated speedup")
    admission.admit_forecast(projection, "preprocessing_" + dataset)
    parts = json.loads((ROOT / f"splits/{dataset}_kfold5.json").read_text()); jobs = [pilot]
    for fold in range(1, 5):
        jobs.append({**pilot, "id": f"re2_fold_{dataset}_f{fold}", "source_commit": admission.commit(),
            "timeout_seconds": max(900, min(7200, int(elapsed*2)+60)),
            "expected_config": {**pilot["expected_config"], "fold": fold},
            "command": ["-m", "exp_revision.re2_fast_fold", "--dataset", dataset, "--fold", str(fold)],
            "test_ids": parts[f"fold_{fold}"]["test"],
            "score_shape": [1, len(parts[f"fold_{fold}"]["test"]), pilot["score_shape"][-1]]})
    return admission.write_manifest(path, "full_re2_fold_" + dataset, jobs, closure(), projection)


def create_index():
    path = q.MANIFESTS / "45_full_re2_fold.json"
    parts = [q.MANIFESTS / f"45_full_re2_fold_{ds}.json" for ds in data.DATASETS]
    if path.exists() or not all(p.exists() for p in parts): return None
    sources = {}; jobs = []
    for p in parts:
        m = json.loads(p.read_text()); sources.update(m["source_sha256"]); jobs += m["jobs"]
        sources[str(p.relative_to(ROOT))] = sha256(p)
    return admission.write_manifest(path, "full_re2_fold", jobs, sources)
