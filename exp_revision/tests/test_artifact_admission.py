import copy
import json

from exp_revision import artifact_admission as admission
from exp_revision import remaining_queue as q


def test_priority_alias_keeps_one_job_and_original_attempts(tmp_path, monkeypatch):
    root = tmp_path
    manifests = root / "manifests"
    folder = root / "queue"
    manifests.mkdir()
    folder.mkdir()
    monkeypatch.setattr(admission, "ROOT", root)
    monkeypatch.setattr(q, "ROOT", root)
    monkeypatch.setattr(q, "MANIFESTS", manifests)
    monkeypatch.setattr(q, "FOLDER", folder)
    source = root / "exp_revision/artifact_admission.py"
    source.parent.mkdir()
    source.write_bytes(__import__("pathlib").Path(admission.__file__).read_bytes())
    job = dict(id="report_full_ready", resource="cpu", cores=1, max_attempts=2,
        expected_config=dict(component="statistical_report", smoke=False), timeout_seconds=1800,
        command=["-m", "exp_revision.report_queue"], expected_status=["completed_statistical_report"])
    manifest = dict(backend="local", phase="full_statistical_report", source_commit="original",
        precharged_hours=q.PRECHARGED,
        source_sha256={"exp_revision/remaining_queue.py": q.sha256(__import__("pathlib").Path(q.__file__))}, jobs=[job])
    original = manifests / "31_report_full_ready.json"
    original.write_text(json.dumps(manifest))
    state = dict(jobs={job["id"]: dict(spec=job, status="pending", attempts=[dict(number=1, elapsed_seconds=10)])}, manifests={})
    before = copy.deepcopy(state)
    (folder / "queue_state.json").write_text(json.dumps(state))
    # Hashing __file__ refers to the production helper, which was copied above.
    monkeypatch.setattr(admission, "__file__", str(source))
    created = admission.create()
    assert len(created) == 1
    alias = json.loads(created[0].read_text())
    assert alias["jobs"] == [job] and "pilot" in alias["phase"]
    assert admission.create() == []
    monkeypatch.setattr(q, "sha256", lambda p: admission.sha256(source) if p == source else admission.sha256(p))
    # The loader requires the scheduler hash at ROOT; provide the original file.
    scheduler = root / "exp_revision/remaining_queue.py"
    scheduler.write_bytes(__import__("pathlib").Path(q.__file__).read_bytes())
    q.load_manifests(state, {})
    assert state["jobs"] == before["jobs"] and len(state["jobs"]) == 1
    assert json.loads((folder / "queue_state.json").read_text()) == before


def test_ordinary_training_is_never_given_artifact_priority(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    folder = tmp_path / "queue"
    manifests.mkdir()
    folder.mkdir()
    monkeypatch.setattr(q, "MANIFESTS", manifests)
    monkeypatch.setattr(q, "FOLDER", folder)
    (folder / "queue_state.json").write_text(json.dumps(dict(jobs={})))
    (manifests / "20_full_training.json").write_text(json.dumps(dict(jobs=[dict(id="train")])))
    assert admission.create() == []
