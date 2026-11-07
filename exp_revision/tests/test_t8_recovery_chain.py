import copy
import json
import pytest

from exp_revision import extended_queue_v2 as queue
from exp_revision import re2_fast_fold as fast


def chain(tmp_path, monkeypatch, ancestors):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue.q, "verified", lambda *args: True)
    folder = tmp_path / "artifact"; folder.mkdir()
    (folder / "run_meta.json").write_text(json.dumps(dict(completed_recovery_receipt=True,
        recovery=dict(original_id="slow", original_ancestors=ancestors))))
    state = dict(jobs={name: dict(status="pending", spec={}, attempts=[]) for name in ("original", "slow")})
    state["jobs"]["fast"] = dict(status="completed", spec={}, output="artifact", attempts=[])
    return state


def test_recovery_chain_retains_attempts_and_resolves_directly(tmp_path, monkeypatch):
    state = chain(tmp_path, monkeypatch, ["original"])
    state["jobs"]["original"]["attempts"] = [dict(status="interrupted", elapsed_seconds=900)]
    original_attempts = copy.deepcopy(state["jobs"]["original"]["attempts"])
    queue.reconcile(state, dict(original="slow", slow="fast"))
    for name in ("original", "slow"):
        assert state["jobs"][name]["status"] == "completed"
        assert state["jobs"][name]["completion_job"] == "fast"
    assert state["jobs"]["original"]["attempts"] == original_attempts


@pytest.mark.parametrize("ancestors", [[], ["unrelated"], ["original", "unrelated"]])
def test_recovery_rejects_incomplete_or_forged_ancestry(tmp_path, monkeypatch, ancestors):
    state = chain(tmp_path, monkeypatch, ancestors)
    with pytest.raises(ValueError, match="provenance"):
        queue.reconcile(state, dict(original="slow", slow="fast"))


def test_recovery_cycle_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="cyclic"):
        queue.reconcile(dict(jobs={}), dict(original="slow", slow="original"))


def test_timeout_scope_requires_all_owned_dead_timeout_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue.q, "alive", lambda a: False)
    (tmp_path / "log").write_text("algorithm in progress\n")
    row = dict(status="stuck", spec=dict(expected_config=dict(component="official_baseline", smoke=True), max_attempts=2),
        attempts=[dict(status="interrupted", exit_code=124, log="log") for _ in range(2)])
    assert queue.isolatable_timeout(row)
    row["spec"]["expected_config"]["smoke"] = False
    assert not queue.isolatable_timeout(row)
    row["spec"]["expected_config"]["smoke"] = True
    row["attempts"][0]["exit_code"] = 1
    assert not queue.isolatable_timeout(row)
    row["attempts"][0]["exit_code"] = 124
    (tmp_path / "log").write_text("Traceback (most recent call last)\n")
    assert not queue.isolatable_timeout(row)


@pytest.mark.parametrize("recovery", [False, True])
def test_fast_wrapper_receipt_is_atomic(tmp_path, monkeypatch, recovery):
    monkeypatch.setattr(fast, "ready", lambda: None)
    monkeypatch.setattr(fast, "ancestry", lambda original: ["original"])
    snapshots = []
    original_dump = fast.runtime.json_dump

    def record(path, value):
        if path == tmp_path / "run_meta.json": snapshots.append(copy.deepcopy(value))
        return original_dump(path, value)

    monkeypatch.setattr(fast.data, "json_dump", record)
    monkeypatch.setattr(fast.runtime, "json_dump", record)
    monkeypatch.setattr(fast.data, "prepare_fold", lambda ds, fold, output:
        fast.data.json_dump(output / "run_meta.json", dict(status="completed_re2_fold")))
    fast.prepare("re2_ss", 0, tmp_path, "slow" if recovery else None)
    assert snapshots[0]["status"] in ("pending_accelerator_receipt", "pending_recovery_receipt")
    final = snapshots[-1]
    assert final["status"] == "completed_re2_fold"
    assert final["acceleration"]["algorithm_changed"] is False
    if recovery:
        assert final["recovery"]["original_id"] == "slow"
        assert final["recovery"]["original_ancestors"] == ["original"]
    assert fast.data.json_dump is record and fast.runtime.json_dump is record
