from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from exp_revision import local_monitor as monitor
from exp_revision.common import json_dump
from exp_revision.disk_guard_local import DiskGuard


def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    monkeypatch.setattr(monitor, "QUEUE", tmp_path / "results_v2/T1_context_local_full/tt")
    monkeypatch.setattr(monitor, "MONITOR", tmp_path / "results_v2/local_monitor")
    monitor.QUEUE.mkdir(parents=True)
    config = {"backend": "local", "interval_minutes": 20, "source_sha256": {}}
    queue = {"status": "running", "charged_core_or_gpu_hours": 1,
             "jobs": {"already_done": {"status": "completed", "output": "preserve"},
                      "unfinished": {"status": "running", "attempts": [
                          {"number": 1, "log": "logs/unfinished.log"}]}}}
    json_dump(monitor.QUEUE / "queue_state.json", queue)
    original_run = monitor.subprocess.run

    def cpu_only_check(args, **kwargs):
        if args[0] == "/usr/bin/nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="test GPU", stderr="")
        return original_run(args, **kwargs)

    monkeypatch.setattr(monitor.subprocess, "run", cpu_only_check)
    return config


def test_stopped_queue_restarts_once_and_preserves_completed_outputs(tmp_path, monkeypatch):
    config = workspace(tmp_path, monkeypatch)
    script = tmp_path / "fake_scheduler.py"
    script.write_text("import time; time.sleep(10)\n")
    monkeypatch.setattr(monitor, "_scheduler_command", lambda _: [sys.executable, str(script)])
    before = (monitor.QUEUE / "queue_state.json").read_bytes()
    first = monitor.check(config)
    child = first["restart"]
    try:
        assert first["action"] == "resume_frozen_local_queue"
        second = monitor.check(config)
        assert second["scheduler_alive"]
        assert second["action"] == "none"
        assert second["restart"]["pid"] == child["pid"]
        assert (monitor.QUEUE / "queue_state.json").read_bytes() == before
    finally:
        import os, signal
        os.kill(child["pid"], signal.SIGTERM)
        # Reap our directly launched child after killing the test scheduler.
        os.waitpid(child["pid"], 0)


def test_existing_queue_lock_prevents_duplicate_launch(tmp_path, monkeypatch):
    config = workspace(tmp_path, monkeypatch)
    with (monitor.QUEUE / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = monitor.check(config)
        assert result["scheduler_alive"]
        assert "restart" not in result


def test_disk_pressure_and_compute_limit_never_restart(tmp_path, monkeypatch):
    config = workspace(tmp_path, monkeypatch)
    original_poll = DiskGuard.poll
    monkeypatch.setattr(DiskGuard, "poll", lambda self: {
        **original_poll(self), "safe_to_launch": False})
    assert monitor.check(config)["action"] == "wait_for_safe_storage"
    monkeypatch.setattr(DiskGuard, "poll", original_poll)
    queue_path = monitor.QUEUE / "queue_state.json"
    queue = json.loads(queue_path.read_text())
    queue["status"] = "budget_limited"
    json_dump(queue_path, queue)
    result = monitor.check(config)
    assert result["action"] == "respect_compute_limit"
    assert "restart" not in result


def test_collection_launch_record_survives_monitor_interruption(tmp_path, monkeypatch):
    config = workspace(tmp_path, monkeypatch)
    monitor.MONITOR.mkdir()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        from exp_revision.disk_guard import process_identity
        record = {"phase": "context", "number": 1, "pid": process.pid,
                  "identity": process_identity(process.pid), "status": "running",
                  "receipt": "unused.json"}
        json_dump(monitor.MONITOR / "launching_context.json", record)
        # Only the other phase may launch; the surviving context writer is adopted.
        launches = []
        monkeypatch.setattr(monitor, "_start_collection", lambda cfg, phase, number, guard:
                            launches.append(phase) or {"phase": phase, "number": number})
        result = monitor._collect(config, {}, DiskGuard(tmp_path))
        assert result["context"] == [record]
        assert launches == ["arise"]
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_crash_during_collection_spawn_does_not_strand_attempt_path(tmp_path, monkeypatch):
    config = workspace(tmp_path, monkeypatch)
    monitor.MONITOR.mkdir()
    (tmp_path / "logs").mkdir()
    guard = DiskGuard(tmp_path)

    def interrupted_spawn(*args, **kwargs):
        raise RuntimeError("simulated monitor termination during Popen")

    monkeypatch.setattr(monitor.subprocess, "Popen", interrupted_spawn)
    with pytest.raises(RuntimeError, match="simulated monitor termination"):
        monitor._start_collection(config, "context", 1, guard)
    launch = json.loads((monitor.MONITOR / "launching_context.json").read_text())
    assert launch["status"] == "launching" and launch["pid"] is None
    original_log = tmp_path / launch["log"]
    assert original_log.exists()
    launches = []
    monkeypatch.setattr(monitor, "_start_collection", lambda cfg, phase, number, guard:
                        launches.append((phase, number)) or {"phase": phase, "number": number})
    recovered = monitor._collect(config, {}, guard)
    assert recovered["context"][0]["status"] == "interrupted"
    assert ("context", 2) in launches
    assert original_log.exists()
    guard.release(launch["lease"])
