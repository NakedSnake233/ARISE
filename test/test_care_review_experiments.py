"""Checks for reviewer-response robustness experiments."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "care"))

from robust_evidence import (  # noqa: E402
    MODALITIES,
    aggregate_type_agnostic,
    apply_availability,
    fault_type_holdout,
    feature_groups,
    fuse_all,
    local_missing_availability,
)
from train_robust_base import modality_mask  # noqa: E402
from run_review_experiments import BasePredictor, retain_adjacency  # noqa: E402


def test_unavailable_modalities_cannot_change_transformed_features():
    names = ["metric:m:positive_z", "log:template_js", "trace:t:delta_mean", "rpc:incoming:gap_mean_log"]
    x = np.arange(2 * 3 * len(names), dtype=np.float32).reshape(2, 3, -1)
    availability = np.ones((2, 3, len(MODALITIES)), dtype=np.float32)
    availability[:, :, MODALITIES.index("trace")] = 0
    a = apply_availability(x, names, availability)
    modified = x.copy()
    modified[:, :, feature_groups(names)["trace"]] += 10_000
    b = apply_availability(modified, names, availability)
    assert np.array_equal(a, b)
    assert np.all(a[:, :, -3:] == availability)


def test_partial_missingness_removes_cross_service_derived_leakage():
    names = [
        "metric:cpu:positive_z",
        "metric:cpu:shock_share",
        "trace:latency:relative_shock",
        "rpc:incoming:gap_mean_log",
        "callers:max_shock",
        "callers:local_minus_neighbor",
        "callers:degree",
    ]
    x = np.ones((1, 3, len(names)), dtype=np.float32)
    availability = np.ones((1, 3, len(MODALITIES)), dtype=np.float32)
    availability[0, 1, MODALITIES.index("metric")] = 0
    availability[0, 2, MODALITIES.index("trace")] = 0
    transformed = apply_availability(x, names, availability, append_flags=False)
    for name in (
        "metric:cpu:shock_share",
        "trace:latency:relative_shock",
        "rpc:incoming:gap_mean_log",
        "callers:max_shock",
        "callers:local_minus_neighbor",
    ):
        assert np.all(transformed[:, :, names.index(name)] == 0)
    # Current-incident call degree is extracted from traces and is removed too.
    assert np.all(transformed[:, :, names.index("callers:degree")] == 0)


def test_local_missing_mask_is_seeded_and_never_removes_every_modality():
    first = local_missing_availability((20, 5), 0.75, 42)
    second = local_missing_availability((20, 5), 0.75, 42)
    assert np.array_equal(first, second)
    assert np.all(first.sum(-1) >= 1)
    assert (first == 0).any()


def test_type_agnostic_transform_removes_channel_identity():
    names = [
        "metric:cpu:positive_z",
        "metric:memory:positive_z",
        "metric:cpu:delta_mean",
        "metric:memory:delta_mean",
        "log:template_js",
    ]
    x = np.asarray([[[1.0, 3.0, 2.0, 4.0, 0.5]]], dtype=np.float32)
    transformed, output_names = aggregate_type_agnostic(x, names)
    assert transformed.shape[-1] == 11
    assert all(":cpu:" not in name and ":memory:" not in name for name in output_names)
    assert "log:template_js" in output_names


def test_all_service_fusion_can_recover_outside_base_top_k():
    base = np.asarray([[0.6, 0.3, 0.1]])
    evidence = np.asarray([[0.01, 0.01, 0.98]])
    final = fuse_all((0.25, base), (0.75, evidence))
    assert final.argmax(-1).item() == 2
    assert np.allclose(final.sum(-1), 1)


def test_training_modality_mask_is_reproducible_and_keeps_one_source():
    first = modality_mask(32, 0.9, np.random.default_rng(7), services=5)
    second = modality_mask(32, 0.9, np.random.default_rng(7), services=5)
    assert np.array_equal(first, second)
    assert first.shape == (32, 5, 3)
    assert first.any(-1).all()


def test_base_and_evidence_modality_orders_are_explicitly_aligned():
    assert BasePredictor.AVAILABILITY_INDEX == {
        "metrics": MODALITIES.index("metric"),
        "traces": MODALITIES.index("trace"),
        "logs": MODALITIES.index("log"),
    }


def test_topology_retention_is_symmetric_and_preserves_self_loops():
    adjacency = np.ones((6, 6), dtype=bool)
    first = retain_adjacency(adjacency, 0.5, 11)
    second = retain_adjacency(adjacency, 0.5, 11)
    assert np.array_equal(first, second)
    assert np.array_equal(first, first.T)
    assert np.diag(first).all()
    assert np.array_equal(retain_adjacency(adjacency, 0.0, 11), np.eye(6))
    assert retain_adjacency(adjacency, 1.0, 11).all()


def test_fault_type_holdout_purges_different_type_overlap_groups():
    rows = [
        {"fault_type": "held", "overlap_group": 7, "incident_id": "a"},
        {"fault_type": "other", "overlap_group": 7, "incident_id": "b"},
        {"fault_type": "other", "overlap_group": 8, "incident_id": "c"},
        {"fault_type": "held", "overlap_group": 9, "incident_id": "d"},
    ]
    fit, held, purged, groups = fault_type_holdout(
        rows, [0, 1, 2], "held", held_scope_indices=range(len(rows))
    )
    assert fit.tolist() == [2]
    assert held.tolist() == [0]
    assert purged.tolist() == [1]
    assert groups == [7, 9]
