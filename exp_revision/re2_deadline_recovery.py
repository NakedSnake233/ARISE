"""Explicit, scientifically unchanged recovery of timed-out RE2 preprocessing."""
from __future__ import annotations
import json
import subprocess

from .common import ROOT, json_dump, sha256
from . import remaining_queue as queue


def create():
    path = queue.MANIFESTS / "54_re2_preprocessing_deadline_recovery.json"
    if path.exists(): return None
    state = json.loads((queue.FOLDER / "queue_state.json").read_text())
    jobs, recipes, sources = [], {}, {}
    for name, row in state["jobs"].items():
        spec = row["spec"]; cfg = spec["expected_config"]
        if cfg.get("component") != "fold_preprocessing" or cfg.get("task") != "T8": continue
        attempts = row["attempts"]
        if not attempts or row["status"] not in ("pending", "stuck"): continue
        if any(a.get("exit_code") != 124 or a.get("status") != "interrupted" for a in attempts): continue
        if any("Traceback (most recent call last)" in queue.log_tail(ROOT / a["log"]) for a in attempts): continue
        replacement = "recovered_" + name
        jobs.append({**spec, "id": replacement, "timeout_seconds": 7200,
            "command": ["-m", "exp_revision.re2_preprocessing_runtime", "--dataset", cfg["dataset"],
                "--fold", str(cfg["fold"]), "--original-id", name]})
        recipes[name] = dict(original_spec=spec, recovery_job=replacement,
            preserved_attempts=attempts, reason="observed preprocessing timeout, unchanged scientific protocol")
    if not jobs: return None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for job in jobs: job["source_commit"] = commit
    for manifest in queue.MANIFESTS.glob("*.json"):
        original = json.loads(manifest.read_text())
        if any(job["id"] in recipes for job in original["jobs"]):
            sources.update(original["source_sha256"])
            sources[str(manifest.relative_to(ROOT))] = sha256(manifest)
    for name in ("exp_revision/re2_deadline_recovery.py", "exp_revision/re2_preprocessing_runtime.py"):
        sources[name] = sha256(ROOT / name)
    json_dump(path, dict(backend="local", phase="re2_preprocessing_deadline_recovery", source_commit=commit,
        precharged_hours=queue.PRECHARGED, source_sha256=sources, jobs=jobs, recovery_recipes=recipes))
    return path


if __name__ == "__main__": print(create())
