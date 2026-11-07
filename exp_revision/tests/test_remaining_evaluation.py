import csv
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from exp_revision import remaining_evaluation as e


def fixture_batch(root, monkeypatch, scores):
    monkeypatch.setattr(e, "ROOT", root)
    ids = [f"event-{i}" for i in range(5)]
    monkeypatch.setattr(e, "load_bundle", lambda _: SimpleNamespace(
        rows=[{"incident_id": key, "fault_type": "cpu"} for key in ids],
        labels=np.zeros(5, dtype=int), services=["a", "b", "c"]))
    cfg = {"dataset": "toy", "method": "baseline", "fold": 0, "seed": 42}
    job = {"id": "job", "expected_config": cfg, "expected_status": ["completed_scores"],
           "score_shape": [len(e.SCENARIOS), 5, 3], "test_ids": ids}
    folder = root / "attempt1"
    folder.mkdir()
    if scores is not None:
        np.savez_compressed(folder / "scores.npz", scores=scores, incident_ids=ids,
                            scenario_keys=[f"{s}_{p or 'none'}" for s, p in e.SCENARIOS])
        (folder / "run_meta.json").write_text(json.dumps({"config": cfg, "errors": ["invalid score"],
            "scores_sha256": e.sha256(folder / "scores.npz")}))
    state = root / "results_v2/remaining_queue/queue_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"jobs": {"job": {"status": "stuck",
        "attempts": [{"output": "attempt1"}], "error_tail": "worker failed"}}}))
    rates = root / "results_v2/T2/realized_rates.csv"
    rates.parent.mkdir(parents=True)
    with rates.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "incident_id", "seed", "scenario", "scenario_param", "realized_rate"])
        for key in ids:
            for scenario, param in e.SCENARIOS:
                writer.writerow(["toy", key, 42, scenario, param, 0])
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"source_sha256": {}, "phase": "full", "jobs": [job]}))
    return manifest, root / "evaluation"


def test_failed_scenario_keeps_events_and_checks_its_own_threshold(tmp_path, monkeypatch):
    scores = np.ones((len(e.SCENARIOS), 5, 3), dtype=np.float32)
    scores[1, 0] = np.nan
    manifest, output = fixture_batch(tmp_path, monkeypatch, scores)
    with pytest.raises(ValueError, match="exceeds 10%"):
        e.evaluate(manifest, output)
    frame = pd.read_parquet(output / "predictions.parquet")
    assert len(frame) == len(e.SCENARIOS) * 5
    failed = frame[frame.status == "failed"]
    assert len(failed) == 1 and np.isinf(failed.rank_of_root).all()
    assert failed.top5.tolist() == ["[]"]
    receipt = json.loads((output / "evaluation_meta.json").read_text())
    assert receipt["failed"] / receipt["rows"] < .1
    assert len(receipt["failure_threshold_exceeded"]) == 1


def test_terminal_worker_without_scores_retains_entire_denominator(tmp_path, monkeypatch):
    manifest, output = fixture_batch(tmp_path, monkeypatch, None)
    with pytest.raises(ValueError, match="exceeds 10%"):
        e.evaluate(manifest, output)
    frame = pd.read_parquet(output / "predictions.parquet")
    assert len(frame) == len(e.SCENARIOS) * 5
    assert (frame.status == "failed").all() and np.isinf(frame.rank_of_root).all()
    assert frame.groupby(["scenario", "scenario_param"]).size().tolist() == [5] * len(e.SCENARIOS)
