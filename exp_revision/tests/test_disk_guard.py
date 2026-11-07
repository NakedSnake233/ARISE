from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp_revision import disk_guard as storage


def fake_volume(monkeypatch, free_bytes):
    monkeypatch.setattr(storage.os, "statvfs", lambda _: SimpleNamespace(
        f_blocks=100 * storage.GIB // 4096, f_frsize=4096,
        f_bavail=free_bytes // 4096, f_favail=100_000))


def test_concurrent_queues_share_remaining_space(tmp_path, monkeypatch):
    fake_volume(monkeypatch, 20 * storage.GIB + storage.JOB_BYTES + storage.MIB)
    first, second = storage.DiskGuard(tmp_path), storage.DiskGuard(tmp_path)
    first.reserve("first", tmp_path / "results_v2/first", tmp_path / "logs/first.log")
    with pytest.raises(storage.InsufficientSpace):
        second.reserve("second", tmp_path / "results_v2/second", tmp_path / "logs/second.log")
    first.release("first")
    second.reserve("second", tmp_path / "results_v2/second", tmp_path / "logs/second.log")
    assert second.poll()["jobs"] == 1


def test_critical_pressure_stops_owned_writer(tmp_path, monkeypatch):
    guard = storage.DiskGuard(tmp_path)
    guard.reserve("writer", tmp_path / "results_v2/job", tmp_path / "logs/job.log")
    fake_volume(monkeypatch, 9 * storage.GIB)
    snapshot = guard.poll()
    assert not snapshot["safe_to_launch"]
    assert snapshot["abort_job_keys"] == ["writer"]


def test_runaway_output_stops_without_deleting_it(tmp_path):
    output = tmp_path / "results_v2/job"
    guard = storage.DiskGuard(tmp_path)
    guard.reserve("writer", output, tmp_path / "logs/job.log")
    output.mkdir()
    large = output / "unexpected.bin"
    # Sparse file exercises the job's size bound without filling the test host.
    with large.open("wb") as file:
        file.truncate(storage.JOB_BYTES + 1)
    assert guard.poll()["abort_job_keys"] == ["writer"]
    assert large.stat().st_size == storage.JOB_BYTES + 1


def test_restart_keeps_live_child_reservation_and_preserves_results(tmp_path):
    output = tmp_path / "results_v2/job"
    guard = storage.DiskGuard(tmp_path)
    env = guard.reserve("child", output, tmp_path / "logs/job.log")
    assert env["TMPDIR"].startswith(str(tmp_path))
    output.mkdir()
    result = output / "old.txt"
    result.write_text("preserve")
    child = subprocess.Popen(["sleep", "30"])
    try:
        guard.attach("child", child.pid)
        ledger = json.loads(guard.path.read_text())
        ledger["leases"]["child"]["owner_identity"] = "previous-owner"
        guard.path.write_text(json.dumps(ledger))
        resumed = storage.DiskGuard(tmp_path)
        assert resumed.poll()["jobs"] == 1
        with pytest.raises(RuntimeError):
            resumed.release("child")
        resumed.reserve("child", output, tmp_path / "logs/job.log", child_pid=child.pid)
        child.terminate()
        child.wait(timeout=5)
        resumed.release("child")
        assert resumed.poll()["jobs"] == 0
        assert result.read_text() == "preserve"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_stale_reservation_recovers_after_environment_restart(tmp_path):
    guard = storage.DiskGuard(tmp_path)
    guard.reserve("old", tmp_path / "results_v2/old", tmp_path / "logs/old.log")
    ledger = json.loads(guard.path.read_text())
    ledger["leases"]["old"]["owner_identity"] = "old-boot-or-pid"
    guard.path.write_text(json.dumps(ledger))
    assert storage.DiskGuard(tmp_path).poll()["jobs"] == 0


def watch_process(root, key, command):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(storage.__file__).resolve().parents[1])
    return subprocess.Popen(storage.guarded_command(root, key, command), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def test_watchdog_enforces_total_size_after_queue_owner_dies(tmp_path):
    guard = storage.DiskGuard(tmp_path)
    output = tmp_path / "results_v2/job"
    output.mkdir()
    guard.reserve("writer", output, tmp_path / "logs/job.log")
    script = ("from pathlib import Path; import time; "
              f"p=Path({str(output)!r}); "
              f"[(p/str(i)).open('wb').truncate({storage.JOB_BYTES // 2 + 4096}) for i in range(2)]; "
              "time.sleep(30)")
    process = watch_process(tmp_path, "writer", [sys.executable, "-c", script])
    try:
        guard.attach("writer", process.pid)
        with guard._locked() as ledger:
            ledger["leases"]["writer"]["owner_identity"] = "previous-queue-owner"
        assert process.wait(timeout=10) == 75, process.stderr.read()
        assert len(list(output.iterdir())) == 2
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_writer_exits_when_watchdog_is_killed(tmp_path):
    guard = storage.DiskGuard(tmp_path)
    output = tmp_path / "results_v2/job"
    output.mkdir()
    guard.reserve("writer", output, tmp_path / "logs/job.log")
    pid_path = output / "writer.pid"
    script = f"import os,time; open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(30)"
    process = watch_process(tmp_path, "writer", [sys.executable, "-c", script])
    try:
        guard.attach("writer", process.pid)
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_path.exists(), process.stderr.read() if process.poll() is not None else "writer did not start"
        child_pid = int(pid_path.read_text())
        process.kill()
        process.wait(timeout=5)
        deadline = time.monotonic() + 3
        while storage.process_identity(child_pid) is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert storage.process_identity(child_pid) is None
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
