from __future__ import annotations

import json
import subprocess
import sys

import pytest

from exp_revision import disk_guard as original
from exp_revision import disk_guard_local as local


def test_other_namespace_cannot_prune_live_reservation(tmp_path, monkeypatch):
    guard = local.DiskGuard(tmp_path)
    guard.reserve("live", tmp_path / "results_v2/live", tmp_path / "logs/live.log")
    before = guard.path.read_bytes()
    monkeypatch.setattr(local, "POLICY", {**local.POLICY, "pid_namespace": "other-namespace"})
    monkeypatch.setattr(original, "process_identity", lambda pid: None)
    with pytest.raises(ValueError, match="PID namespace"):
        local.DiskGuard(tmp_path).poll()
    assert guard.path.read_bytes() == before
    # A legacy guard also fails before pruning a namespace-bound ledger.
    with pytest.raises(ValueError, match="disk policy"):
        original.DiskGuard(tmp_path).poll()
    assert guard.path.read_bytes() == before


def test_local_watchdog_shares_namespace_bound_ledger(tmp_path):
    guard = local.DiskGuard(tmp_path)
    output = tmp_path / "results_v2/job"
    output.mkdir()
    guard.reserve("writer", output, tmp_path / "logs/job.log")
    command = [sys.executable, "-c", f"open({str(output / 'done')!r},'w').write('done')"]
    process = subprocess.Popen(local.guarded_command(tmp_path, "writer", command),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    guard.attach("writer", process.pid)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stdout + stderr
    assert (output / "done").read_text() == "done"
    guard.release("writer")
    assert json.loads(guard.path.read_text())["leases"] == {}
