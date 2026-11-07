import json
import fcntl

from exp_revision import t8_admission as admission
from exp_revision import extended_stages as stages


def test_pending_pilots_reserve_deadlines_and_deduplicate_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(admission.q, "MANIFESTS", tmp_path)
    monkeypatch.setattr(admission, "scientific_blocks", lambda current: set())
    job = dict(id="pilot", resource="cpu", cores=4, timeout_seconds=1800,
        expected_config=dict(task="T8", dataset="re2_ob", component="evidence", smoke=True))
    payload = dict(jobs=[job], forecast=None)
    (tmp_path / "one.json").write_text(json.dumps(payload))
    (tmp_path / "alias.json").write_text(json.dumps(payload))
    assert admission.pending_forecasts(dict(jobs={})) == dict(cpu=1810*4/3600, gpu=0.)


def test_shared_phase_lock_prevents_any_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(stages.q, "FOLDER", tmp_path)
    monkeypatch.setattr(stages.admission, "create_preprocessing_full", lambda: (_ for _ in ()).throw(AssertionError("admission under busy phase lock")))
    with (tmp_path / ".stages.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = stages.check()
    assert result == dict(status="extended_stage_check_already_running")
