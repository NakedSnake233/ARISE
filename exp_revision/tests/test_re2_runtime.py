import copy
import json
from pathlib import Path

from exp_revision import extended_queue as queue
from exp_revision import re2_preprocessing_runtime as runtime


def test_only_actual_official_algorithm_failure_is_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    folder = tmp_path / "failed"; folder.mkdir()
    cfg = dict(component="official_baseline", dataset="aiops", method="CIRCA", task="T8", smoke=True)
    row = dict(status="stuck", spec=dict(expected_config=cfg), attempts=[dict(output="failed")])
    state = dict(jobs=dict(circa=row))
    before = copy.deepcopy(state)
    meta = dict(status="blocked_algorithm_failure_rate", errors=[dict(error="singular matrix")],
        config=cfg, failure_rates=dict(S0_none=1., S1_metric=0.))
    (folder / "run_meta.json").write_text(json.dumps(meta))
    assert queue.scientific_blocks(state) == {("aiops", "CIRCA")}
    assert queue.in_blocked_scope(row, queue.scientific_blocks(state))
    assert not queue.in_blocked_scope(dict(spec=dict(expected_config={**cfg, "method": "BARO"})), queue.scientific_blocks(state))
    assert state == before  # Failure is never relabeled complete.
    meta["status"] = "python_error"
    (folder / "run_meta.json").write_text(json.dumps(meta))
    assert queue.scientific_blocks(state) == set()
    assert not queue.isolatable_failure(row)


def test_runtime_observation_keeps_drain_outputs_and_restores_globals(tmp_path, monkeypatch):
    import drain3
    miner_type = drain3.TemplateMiner
    messages = ["request 42 from user alice", "request 43 from user bob", "connected to database db1",
        "request 44 from user claire", "connected to database db2", "request 42 from user alice"]

    def computation():
        miner = drain3.TemplateMiner()
        insertion = [miner.add_log_message(m)["cluster_id"] for m in messages]
        templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
        matches = [miner.match(m).cluster_id for m in messages]
        return insertion, templates, matches, runtime.data._templates_disjoint(templates)

    expected = computation(); actual = []
    monkeypatch.setattr(runtime.data, "prepare_fold", lambda *args: actual.append(computation()))
    normalize = runtime.data._normalize; disjoint = runtime.data._templates_disjoint
    runtime.prepare("re2_ss", 0, tmp_path)
    assert actual == [expected]
    assert drain3.TemplateMiner is miner_type
    assert runtime.data._normalize is normalize
    assert runtime.data._templates_disjoint is disjoint


def test_recovery_receipt_cannot_complete_before_provenance(tmp_path, monkeypatch):
    import pytest

    def interrupted(dataset, fold, output):
        runtime.data.json_dump(output / "run_meta.json", dict(status="completed_re2_fold"))
        raise RuntimeError("interrupted before recovery publication")

    monkeypatch.setattr(runtime.data, "prepare_fold", interrupted)
    original_dump = runtime.data.json_dump
    with pytest.raises(RuntimeError, match="interrupted"):
        runtime.prepare("re2_ss", 0, tmp_path, "re2_fold_re2_ss_f0")
    assert json.loads((tmp_path / "run_meta.json").read_text())["status"] == "pending_recovery_receipt"
    assert runtime.data.json_dump is original_dump
