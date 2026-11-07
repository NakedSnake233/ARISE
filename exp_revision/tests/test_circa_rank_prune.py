import numpy as np
from exp_revision.rcaeval_rank_worker import select_columns


def test_redundancy_selection_is_mask_local_and_preserves_original_values():
    rng = np.random.default_rng(2027)
    x = rng.normal(size=(7, 8)); x[:, 1] = 3 * x[:, 0] + 2; x[:, 6] = 1
    services = ["a", "b", "c", "d"]
    mapping = [service for service in services for _ in range(2)]
    names = [f"v{i//2:03d}_c{i%2:03d}" for i in range(8)]
    before = x.copy()
    selected, audit = select_columns(x, names, mapping, services)
    assert len(selected) <= 5 and 1 not in selected and 6 not in selected
    assert selected[:3] == [0, 2, 4]  # Same service round-robin, no labels.
    centered = x[:, selected] - x[:, selected].mean(0)
    assert np.linalg.matrix_rank(centered) == len(selected)
    assert np.array_equal(x, before) and not audit["values_changed"] and not audit["uses_labels"]
    hidden_b = [i for i, service in enumerate(mapping) if service != "b"]
    chosen, masked_audit = select_columns(x[:, hidden_b], [names[i] for i in hidden_b], [mapping[i] for i in hidden_b], services)
    assert "b" not in masked_audit["selected_services"]


def test_no_observed_metric_view_stays_empty():
    selected, audit = select_columns(np.empty((7, 0)), [], [], ["a", "b"])
    assert selected == [] and audit["varying_complete_columns"] == 0


def test_empty_rank_with_observed_metrics_is_failed(tmp_path, monkeypatch):
    import pytest
    from exp_revision import rcaeval_rank_worker as module

    def empty(*args, **kwargs): return np.zeros(2), dict(status="completed_official", metric_ranks=[])
    monkeypatch.setattr(module.worker, "infer", empty)

    def invocation():
        x = np.arange(7.)[:, None]
        module.worker.infer("CIRCA", x, np.arange(7), ["a_cpu"], ["a"], ["a", "b"], 3, 42)

    monkeypatch.setattr(module.worker, "main", invocation)
    with pytest.raises(RuntimeError, match="no ranked observed service"): module.main()
    assert module.worker.infer is empty
