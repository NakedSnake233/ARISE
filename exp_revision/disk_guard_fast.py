"""Exact disk-accounting accelerator; policies, leases and watchdogs unchanged."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import stat

from . import disk_guard as base
from .disk_guard import POLICY, JOB_BYTES
from .disk_guard_local import DiskGuard as LocalDiskGuard, InsufficientSpace
from .disk_guard_local import guarded_command as local_command


def usage_bytes(path: Path) -> int:
    """Same logical/allocated max and symlink rules, using cached DirEntry types."""
    try:
        root_stat = os.lstat(path)
    except FileNotFoundError:
        return 0
    if stat.S_ISLNK(root_stat.st_mode):
        return 0
    if stat.S_ISREG(root_stat.st_mode):
        return max(root_stat.st_size, root_stat.st_blocks * 512)
    if not stat.S_ISDIR(root_stat.st_mode):
        return 0
    total = 0
    pending = [os.fspath(path)]
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            # os.walk likewise skips directories it cannot enumerate.
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
                    else:
                        info = entry.stat(follow_symlinks=False)
                        total += max(info.st_size, info.st_blocks * 512)
                except FileNotFoundError:
                    # Active writers may atomically replace their .tmp files.
                    continue
    return total


class DiskGuard(LocalDiskGuard):
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



def guarded_command(root: Path, key: str, command: list[str]) -> list[str]:
    result = local_command(root, key, command)
    result[2] = "exp_revision.disk_guard_fast"
    return result


def main():
    # Only this watchdog process uses the accelerated class. Base source stays frozen.
    base.DiskGuard = DiskGuard
    base.main()


if __name__ == "__main__": main()
