import json
import pytest

from exp_revision import compute_budget_policy as policy


def test_missing_author_grant_keeps_default_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(policy,"APPROVAL",tmp_path/"missing.json")
    monkeypatch.setattr(policy.q,"CPU_BUDGET",200.)
    monkeypatch.setattr(policy.q,"GPU_BUDGET",48.)
    current=policy.apply()
    assert current["limits"]==dict(cpu=200.,gpu=48.)
    assert current["author_approval_sha256"] is None
    assert current["longer_window_authorized"] is False


def test_explicit_grant_changes_only_accounting_cap(tmp_path,monkeypatch):
    approval=tmp_path/"grant.json"
    approval.write_text(json.dumps(dict(approved=True,limits=dict(cpu=260.,gpu=56.),
        scope="existing_protocol_only",source="explicit author reply in this conversation")))
    monkeypatch.setattr(policy,"APPROVAL",approval)
    monkeypatch.setattr(policy.q,"CPU_BUDGET",200.)
    monkeypatch.setattr(policy.q,"GPU_BUDGET",48.)
    current=policy.apply()
    assert current["limits"]==dict(cpu=260.,gpu=56.)
    assert current["author_approval_sha256"]
    assert policy.q.CPU_BUDGET==260. and policy.q.GPU_BUDGET==56.
    assert current["longer_window_authorized"] is False
    approval.write_text(json.dumps(dict(approved=True,limits=dict(cpu=1000.,gpu=500.),
        scope="existing_protocol_only",source="explicit author reply in this conversation")))
    with pytest.raises(ValueError,match="unsupported"):
        policy.apply()
