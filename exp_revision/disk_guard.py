"""Shared disk reservations and monitoring for experiment subprocesses.

Reservations cover outputs, logs and a task-owned temporary directory. No
existing result or cache is removed. Queues stop launching before the free-space
floor, and terminate their own writers if a critical floor or job limit is hit.
"""

from __future__ import annotations

import datetime as dt
import argparse
import ctypes
import fcntl
import json
import os
import resource
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


GIB = 1024 ** 3
MIB = 1024 ** 2
JOB_BYTES = 512 * MIB
POLICY = {"min_free_bytes": 20 * GIB, "min_free_fraction": 0.10,
          "critical_free_bytes": 10 * GIB, "critical_free_fraction": 0.05,
          "max_experiment_bytes": 20 * GIB, "job_bytes": JOB_BYTES,
          "min_free_inodes": 50_000, "job_inodes": 256,
          "snapshot_interval_seconds": 30}


class InsufficientSpace(RuntimeError):
    pass


def process_identity(pid: int | None) -> str | None:
    if not pid:
        return None
    try:
        # stat fields after the final ')' start with field 3 (state).
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{boot}:{fields[19]}"
    except (OSError, IndexError):
        return None


def usage_bytes(path: Path) -> int:
    """Bound logical and allocated size; do not follow cache/data symlinks."""
    if not path.exists() or path.is_symlink():
        return 0
    if path.is_file():
        stat = path.stat()
        return max(stat.st_size, stat.st_blocks * 512)
    total = 0
    for directory, _, files in os.walk(path, followlinks=False):
        for name in files:
            item = Path(directory) / name
            if item.is_symlink():
                continue
            try:
                stat = item.stat()
                total += max(stat.st_size, stat.st_blocks * 512)
            except FileNotFoundError:
                continue
    return total


class DiskGuard:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.folder = self.root / "results_v2/.disk_guard"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "reservations.json"
        self.temporary = self.root / "cache/job_tmp"
        self.temporary.mkdir(parents=True, exist_ok=True)
        self.monitored = [self.root / name for name in ("results_v2", "cache", "logs")]
        self.owner_pid = os.getpid()
        self.owner_identity = process_identity(self.owner_pid)
        self.last_snapshot = 0.0

    @contextmanager
    def _locked(self):
        with (self.folder / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                "policy": POLICY, "leases": {}}
            if state["policy"] != POLICY:
                raise ValueError("disk policy differs from existing reservation ledger")
            for key, row in list(state["leases"].items()):
                owner_alive = process_identity(row["owner_pid"]) == row["owner_identity"]
                child_alive = (row.get("child_identity") is not None and
                               process_identity(row.get("child_pid")) == row["child_identity"])
                if not owner_alive and not child_alive:
                    del state["leases"][key]
            yield state
            self._write(self.path, state)

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        temp.replace(path)

    def _snapshot(self, state: dict, additional_jobs: int = 0) -> dict:
        occupied = {key: sum(usage_bytes(Path(p)) for p in row["paths"])
                    for key, row in state["leases"].items()}
        remaining = sum(max(0, JOB_BYTES - occupied[key]) for key in occupied)
        remaining += additional_jobs * JOB_BYTES
        experiment_bytes = sum(usage_bytes(p) for p in self.monitored)
        volumes = {}
        for path in (self.root, self.temporary):
            stat = path.stat()
            if str(stat.st_dev) in volumes:
                continue
            fs = os.statvfs(path)
            total, free = fs.f_blocks * fs.f_frsize, fs.f_bavail * fs.f_frsize
            floor = max(POLICY["min_free_bytes"], int(total * POLICY["min_free_fraction"]))
            critical = max(POLICY["critical_free_bytes"], int(total * POLICY["critical_free_fraction"]))
            volumes[str(stat.st_dev)] = {
                "path": str(path), "total_bytes": total, "free_bytes": free,
                "launch_floor_bytes": floor, "critical_floor_bytes": critical,
                "free_inodes": fs.f_favail}
        inode_reserve = (len(occupied) + additional_jobs) * POLICY["job_inodes"]
        safe = (experiment_bytes + remaining <= POLICY["max_experiment_bytes"]
                and all(v["free_bytes"] >= v["launch_floor_bytes"] + remaining
                        and v["free_inodes"] >= POLICY["min_free_inodes"] + inode_reserve
                        for v in volumes.values()))
        critical = (experiment_bytes > POLICY["max_experiment_bytes"]
                    or any(v["free_bytes"] < v["critical_floor_bytes"]
                           or v["free_inodes"] < POLICY["min_free_inodes"]
                           for v in volumes.values()))
        # Every scheduler/watchdog receives the pressure signal, then acts only
        # on its own job keys. This also works after the original owner dies.
        abort = sorted(occupied if critical else
                       {key for key in occupied if occupied[key] > JOB_BYTES})
        return {"time": dt.datetime.now(dt.timezone.utc).isoformat(),
                "safe_to_launch": safe, "abort_job_keys": abort,
                "experiment_bytes": experiment_bytes, "reserved_remaining_bytes": remaining,
                "jobs": len(occupied), "job_bytes": occupied, "volumes": volumes}

    def poll(self) -> dict:
        with self._locked() as state:
            snapshot = self._snapshot(state)
            if time.monotonic() - self.last_snapshot >= POLICY["snapshot_interval_seconds"]:
                self._write(self.folder / "latest.json", snapshot)
                with (self.folder / "history.jsonl").open("a") as log:
                    log.write(json.dumps(snapshot, sort_keys=True) + "\n")
                self.last_snapshot = time.monotonic()
            return snapshot

    def reserve(self, key: str, output: Path, log: Path, *, child_pid: int | None = None) -> dict[str, str]:
        if not key or "/" in key or ".." in key:
            raise ValueError("invalid disk reservation name")
        temp = self.temporary / key
        paths = [str(p.resolve()) for p in (output, log, temp)]
        if any(not Path(p).is_relative_to(self.root) for p in paths):
            raise ValueError("job writes must stay inside its experiment workspace")
        with self._locked() as state:
            old = state["leases"].get(key)
            if old:
                if old["paths"] != paths:
                    raise ValueError("disk reservation output changed")
                if child_pid is None:
                    raise ValueError("disk reservation already exists; adopt its child explicitly")
            elif child_pid is None and not self._snapshot(state, additional_jobs=1)["safe_to_launch"]:
                raise InsufficientSpace("disk space/inodes or experiment quota cannot cover another job")
            state["leases"][key] = {
                "owner_pid": self.owner_pid, "owner_identity": self.owner_identity,
                "child_pid": child_pid, "child_identity": process_identity(child_pid),
                "paths": paths, "reserved_bytes": JOB_BYTES}
        temp.mkdir(parents=True, exist_ok=True)
        return {"TMPDIR": str(temp), "JOBLIB_TEMP_FOLDER": str(temp),
                "XDG_CACHE_HOME": str(temp / "xdg")}

    def attach(self, key: str, pid: int) -> None:
        with self._locked() as state:
            state["leases"][key].update(child_pid=pid, child_identity=process_identity(pid))

    def release(self, key: str) -> None:
        with self._locked() as state:
            row = state["leases"].get(key)
            if row and row.get("child_identity") is not None and process_identity(row["child_pid"]) == row["child_identity"]:
                raise RuntimeError("cannot release the reservation of a running writer")
            state["leases"].pop(key, None)


def guarded_command(root: Path, key: str, command: list[str]) -> list[str]:
    return [sys.executable, "-m", "exp_revision.disk_guard", "watch",
            "--root", str(root), "--key", key, "--", *command]


def watch(root: Path, key: str, command: list[str]) -> int:
    """Monitor one writer independently of the queue, even if it crashes."""
    guard = DiskGuard(root)
    deadline_pid = os.getppid()
    deadline_identity = process_identity(deadline_pid)
    if key not in guard.poll()["job_bytes"]:
        raise ValueError("writer has no durable disk reservation")
    stop = False

    def stopping(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    parent_pid = os.getpid()

    def child_limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (JOB_BYTES, JOB_BYTES))
        # Linux parent-death signal closes the remaining watchdog-crash window.
        # This stdlib-only watchdog is single-threaded before this fork.
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)

    child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                             start_new_session=True, preexec_fn=child_limits)
    try:
        while child.poll() is None:
            snapshot = guard.poll()
            deadline_lost = (os.getppid() != deadline_pid or
                             process_identity(deadline_pid) != deadline_identity)
            if (stop or deadline_lost or key not in snapshot["job_bytes"]
                    or key in snapshot["abort_job_keys"]):
                reason = ("disk pressure" if key in snapshot["abort_job_keys"] else
                          "deadline lost" if deadline_lost else
                          "lease lost" if key not in snapshot["job_bytes"] else "signal")
                print(json.dumps({"disk_guard_stopped": key, "stop_signal": stop,
                                  "reason": reason,
                                  "deadline_lost": deadline_lost,
                                  "space": snapshot["volumes"]}), flush=True)
                return 75
            time.sleep(2)
        return child.returncode if child.returncode >= 0 else 128 - child.returncode
    finally:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("watch",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key", required=True)
    tokens = sys.argv[1:]
    separator = tokens.index("--") if "--" in tokens else len(tokens)
    args = parser.parse_args(tokens[:separator])
    command = tokens[separator + 1:]
    if not command:
        parser.error("watch requires a writer command")
    raise SystemExit(watch(args.root, args.key, command))


if __name__ == "__main__":
    main()
