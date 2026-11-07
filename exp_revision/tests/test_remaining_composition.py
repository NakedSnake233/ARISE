import copy
import numpy as np
from exp_revision.score_receipts import fused
from exp_revision.train_evidence import fuse_all


def test_fusion_is_independent_for_every_scenario_and_incident():
    context = np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 0., 0.]]])
    evidence = np.array([[[0., 3., 0.], [0., 0., 7.]], [[2., 0., 0.], [0., 4., 0.]]])
    actual = fused(context, evidence)
    expected = np.array([[[.25, .75, 0.], [0., .25, .75]], [[.75, 0., .25], [.25, .75, 0.]]])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual[1], fuse_all((.25, context[1]), (.75, evidence[1])))


def test_main_rejects_dropped_event_duplicate_and_changed_rate(monkeypatch):
    import pandas as pd
    import pytest
    from exp_revision import merge_main as m
    monkeypatch.setattr(m, "METHODS", ["ARISE"]); monkeypatch.setattr(m, "SEEDS", [42]); monkeypatch.setattr(m, "SCENARIOS", [("S0", "")])
    frame = pd.DataFrame([dict(dataset="toy", method="ARISE", incident_id=name, seed=42, scenario="S0", scenario_param="",
        split="kfold5", invalid_by_design=False, fold=0, root_true="a", fault_type="cpu", realized_rate=0.,
        status="ok", rank_of_root=1., top5='["a"]') for name in ("i", "j")])
    args = ({"toy": ["i", "j"]}, {("toy", name): 0 for name in ("i", "j")},
        {("toy", name, 42, "S0", ""): 0. for name in ("i", "j")}, {("toy", name): ("a", "cpu", 2) for name in ("i", "j")})
    m.validate(frame, *args)
    with pytest.raises(ValueError, match="denominator"): m.validate(frame.iloc[:1], *args)
    with pytest.raises(ValueError, match="duplicate"): m.validate(pd.concat([frame, frame.iloc[:1]]), *args)
    changed = frame.copy(); changed.loc[0, "realized_rate"] = .5
    with pytest.raises(ValueError, match="deletion rate"): m.validate(changed, *args)
    failed = frame.copy(); failed.loc[0, "status"] = "failed"
    with pytest.raises(ValueError, match="rank inf"): m.validate(failed, *args)
    failed.loc[0, "rank_of_root"] = np.inf; failed.loc[0, "top5"] = "[]"
    m.validate(failed, *args)
