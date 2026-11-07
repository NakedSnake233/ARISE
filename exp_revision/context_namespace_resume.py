"""Run the frozen local context scheduler with a PID namespace storage guard.

The scheduler and original guard are source inputs of completed ET-spec runs,
so this adapter preserves them and records its additional reviewed entry point.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path

from . import context_local_resume as scheduler
from . import disk_guard_local as storage
from .common import ROOT, json_dump, sha256


SCHEDULER_SHA256 = "517ce0a511cb445ada82632695e4cd5dd88858e2d8c477e97bb885199fe13e49"


def run(manifest: Path, wrapper_sha256: str, local_guard_sha256: str) -> dict:
    def verify_local_sources() -> None:
        for path, digest in ((Path(__file__), wrapper_sha256),
                             (ROOT / "exp_revision/disk_guard_local.py", local_guard_sha256),
                             (ROOT / "exp_revision/context_local_resume.py", SCHEDULER_SHA256),
                             (ROOT / "exp_revision/disk_guard.py", storage.BASE_SHA256)):
            if sha256(path) != digest:
                raise ValueError(f"reviewed local execution source changed: {path.name}")

    verify_local_sources()
    folder = ROOT / "results_v2/T1_context_local_full/tt"
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / ".namespace_scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Reject a sandbox invocation before it can touch the training queue state.
    storage.DiskGuard(ROOT).poll()
    path = folder / "namespace_execution.json"
    receipt = {"wrapper_sha256": wrapper_sha256,
               "local_disk_guard_sha256": local_guard_sha256,
               "original_scheduler_sha256": SCHEDULER_SHA256,
               "original_disk_guard_sha256": storage.BASE_SHA256,
               "policy": storage.POLICY, "backend": "local",
               "manifest_sha256": sha256(manifest)}
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(key) != value for key, value in receipt.items()):
            raise ValueError("namespace scheduler differs from existing execution receipt")
    else:
        json_dump(path, {**receipt, "created": dt.datetime.now(dt.timezone.utc).isoformat()})

    original_verify = scheduler._verify_files_unchanged

    def verify_inputs(frozen: dict) -> None:
        verify_local_sources()
        original_verify(frozen)

    scheduler.DiskGuard = storage.DiskGuard
    scheduler.POLICY = storage.POLICY
    scheduler.guarded_command = storage.guarded_command
    scheduler._verify_files_unchanged = verify_inputs
    return scheduler.run(manifest, SCHEDULER_SHA256, storage.BASE_SHA256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    parser.add_argument("--local-disk-guard-sha256", required=True)
    args = parser.parse_args()
    state = run(args.manifest, args.wrapper_sha256, args.local_disk_guard_sha256)
    print(json.dumps({"status": state["status"],
                      "charged_hours": state["charged_core_or_gpu_hours"]}))


if __name__ == "__main__":
    main()
