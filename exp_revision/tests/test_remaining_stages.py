import json
from types import SimpleNamespace
from exp_revision import remaining_stages as s


def test_phase_ready_requires_each_verified_completed_output(tmp_path, monkeypatch):
    monkeypatch.setattr(s.q,"MANIFESTS",tmp_path)
    (tmp_path/"phase.json").write_text(json.dumps(dict(jobs=[dict(id="a"),dict(id="b")])))
    state=dict(jobs={"a":dict(status="completed",output="a"),"b":dict(status="running")})
    monkeypatch.setattr(s.q,"verified",lambda *_: True)
    assert not s.ready("phase.json",state)
    state["jobs"]["b"].update(status="completed",output="b")
    assert s.ready("phase.json",state)
    monkeypatch.setattr(s.q,"verified",lambda *_: False)
    assert not s.ready("phase.json",state)


def test_unfinished_phases_do_not_register_followups(tmp_path, monkeypatch):
    monkeypatch.setattr(s.q,"FOLDER",tmp_path); monkeypatch.setattr(s.q,"MANIFESTS",tmp_path)
    (tmp_path/"queue_state.json").write_text(json.dumps(dict(status="running",jobs={},manifests={})))
    monkeypatch.setattr(s.adaptive_queue,"check",lambda: dict(action="none",status="running"))
    def forbidden(*_args,**_kwargs): raise AssertionError("phase advanced before prerequisite")
    monkeypatch.setattr(s.composition,"create",forbidden); monkeypatch.setattr(s.collection,"create",forbidden)
    monkeypatch.setattr(s.report_queue,"create",forbidden)
    result=s.check()
    assert result["created_manifests"]==[] and result["errors"]==[]


def test_report_jobs_are_serialized_before_new_snapshot(tmp_path, monkeypatch):
    from exp_revision import report_queue as r
    monkeypatch.setattr(r.q,"FOLDER",tmp_path)
    row=dict(status="running",spec=dict(expected_config=dict(component="statistical_report",smoke=False)))
    (tmp_path/"queue_state.json").write_text(json.dumps(dict(jobs={"already":row})))
    assert r.create() is None
