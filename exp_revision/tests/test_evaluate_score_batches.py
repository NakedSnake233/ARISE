import json
import numpy as np
import pandas as pd
import pytest

from exp_revision import evaluate_score_batches as batch
from exp_revision import remaining_evaluation as single
from exp_revision.tests.test_remaining_evaluation import fixture_batch


def test_multiple_method_axes_and_failed_scene_denominator(tmp_path, monkeypatch):
    manifest, output = fixture_batch(tmp_path, monkeypatch, None)
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "load_bundle", single.load_bundle)
    spec = json.loads(manifest.read_text()); job = spec["jobs"][0]
    cfg = job["expected_config"]; cfg.pop("method"); cfg.update(task="T2", methods=["one", "two"])
    job["score_shape"] = [2, len(batch.SCENARIOS), 5, 3]
    scores = np.ones(job["score_shape"]); scores[1, 1, 0] = np.nan
    folder = tmp_path / "attempt1"
    np.savez_compressed(folder / "scores.npz", scores=scores, method_names=cfg["methods"],
                        incident_ids=job["test_ids"], scenario_keys=[f"{s}_{p or 'none'}" for s, p in batch.SCENARIOS])
    (folder / "run_meta.json").write_text(json.dumps({"config": cfg, "scores_sha256": batch.sha256(folder / "scores.npz")}))
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="exceeds 10%"):
        batch.evaluate(manifest, output)
    frame = pd.read_parquet(output / "predictions.parquet")
    assert len(frame) == 120
    assert frame.groupby("method").size().to_dict() == {"one": 60, "two": 60}
    failed = frame[frame.status == "failed"]
    assert len(failed) == 1 and failed.method.tolist() == ["two"]
    assert np.isinf(failed.rank_of_root).all()
    assert (frame[frame.method == "one"].rank_of_root == 1).all()
