"""Use the reviewed storage guard within one shared PID namespace.

The original guard remains frozen because completed ET-spec manifests bind it.
This local entry point rejects another process namespace before inspecting or
pruning leases. All active local queues and watchdogs use this entry point.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from . import disk_guard as base


BASE_SHA256 = "a0d5cd9a1979ec917504f88084e9f3eb6a210975186fb08a1990e099a0ab577e"
POLICY = {**base.POLICY, "pid_namespace": os.readlink("/proc/self/ns/pid")}
InsufficientSpace = base.InsufficientSpace


class DiskGuard(base.DiskGuard):
    def __init__(self, root: Path):
        if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA256:
            raise ValueError("the original reviewed storage guard changed")
        super().__init__(root)

    @contextmanager
    def _locked(self):
        with (self.folder / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                "policy": POLICY, "leases": {}}
            # Check this before process_identity: PIDs from another namespace
            # must never be mistaken for dead queue owners or writers.
            if state["policy"] != POLICY:
                raise ValueError("disk policy or PID namespace differs from shared ledger; "
                                 "run in its original host namespace")
            for key, row in list(state["leases"].items()):
                owner_alive = base.process_identity(row["owner_pid"]) == row["owner_identity"]
                child_alive = (row.get("child_identity") is not None and
                               base.process_identity(row.get("child_pid")) == row["child_identity"])
                if not owner_alive and not child_alive:
                    del state["leases"][key]
            yield state
            self._write(self.path, state)


def guarded_command(root: Path, key: str, command: list[str]) -> list[str]:
    result = base.guarded_command(root, key, command)
    result[2] = "exp_revision.disk_guard_local"
    return result


def main() -> None:
    # This assignment is confined to the separate watchdog process.
    base.DiskGuard = DiskGuard
    base.main()


if __name__ == "__main__":
    main()
