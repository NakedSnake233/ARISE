"""Prioritize ready collectors/reports using the frozen executor's pilot list.

These manifests repeat existing job IDs and specifications exactly. The word
pilot is an operational priority marker, not a scientific smoke/pilot run.
The executor deduplicates by ID, preserving attempts and compute accounting.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q


def create():
    q.FOLDER.mkdir(parents=True, exist_ok=True)
    with (q.FOLDER / ".artifact_admission.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        state = json.loads((q.FOLDER / "queue_state.json").read_text())
        created = []
        paths = sorted([*q.MANIFESTS.glob("30_collect_*.json"), *q.MANIFESTS.glob("31_report_*.json")])
        for original in paths:
            manifest = json.loads(original.read_text())
            if len(manifest["jobs"]) != 1:
                raise ValueError("artifact admission requires a single original job")
            job = manifest["jobs"][0]
            if job["expected_config"].get("component") not in ("artifact_collection", "statistical_report"):
                raise ValueError("only collectors and statistical reports can be prioritized")
            if job["resource"] != "cpu" or job["cores"] != 1:
                raise ValueError("artifact processing must retain its single CPU allocation")
            row = state["jobs"].get(job["id"])
            if row is not None:
                if row["spec"] != job:
                    raise ValueError("original artifact job specification changed")
                if row["status"] != "pending":
                    continue
            target = q.MANIFESTS / f"32_pilot_{job['id']}.json"
            if target.exists():
                continue
            sources = {**manifest["source_sha256"], str(original.relative_to(ROOT)): sha256(original),
                "exp_revision/artifact_admission.py": sha256(Path(__file__))}
            json_dump(target, dict(backend=manifest["backend"], phase="pilot_artifact_admission",
                source_commit=manifest["source_commit"], precharged_hours=manifest["precharged_hours"],
                source_sha256=sources, jobs=[job], original_manifest=str(original.relative_to(ROOT)),
                reason="prioritize already-ready artifact processing; same job ID, configuration, attempts and accounting"))
            created.append(target)
        return created
