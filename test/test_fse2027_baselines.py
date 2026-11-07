from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/care"))

from fse2027_baseline_common import (  # noqa: E402
    DatasetBundle,
    fallback_scores,
    normalize_rows,
    ranks_from_scores,
)
from fse2027_classical_baselines import METHODS  # noqa: E402


PROTOCOL = json.loads(
    (ROOT / "artifacts/care/fse2027-baselines/BASELINE_PROTOCOL_FROZEN.json").read_text()
)


def first_configuration(grid):
    return {key: values[0] for key, values in grid.items()}


def synthetic_bundle(zero: bool = False) -> DatasetBundle:
    rng = np.random.default_rng(2027)
    n, s, t = 12, 3, 8
    features = {
        "metrics": rng.normal(size=(n, s, t, 4)),
        "logs": rng.poisson(0.3, size=(n, s, t, 5)).astype(float),
        "traces": rng.normal(size=(n, s, t, 3)),
    }
    if zero:
        for value in features.values():
            value.fill(0)
    labels = np.arange(n) % s
    rows = []
    for index in range(n):
        rows.append(
            {
                "incident_id": f"synthetic-{index}",
                "root_service": f"s{labels[index]}",
                "fault_type": "synthetic",
                "split": "train" if index < 8 else "validation" if index < 10 else "test",
                "calls": [
                    {"source": "s0", "target": "s1", "count": 2},
                    {"source": "s1", "target": "s2", "count": 1},
                ],
            }
        )
    return DatasetBundle(
        name="synthetic",
        path=ROOT,
        rows=rows,
        metadata={"services": ["s0", "s1", "s2"], "before": 4, "step": 1},
        features=features,
        services=["s0", "s1", "s2"],
        labels=labels,
        train=np.arange(8),
        validation=np.arange(8, 10),
        fit=np.arange(8),
        test=np.arange(10, 12),
        split_bin=4,
    )


def test_all_classical_adapters_are_finite_and_label_blind_on_test():
    bundle = synthetic_bundle()
    changed = bundle.labels.copy()
    changed[bundle.test] = (changed[bundle.test] + 1) % bundle.service_count
    relabeled = dataclasses.replace(bundle, labels=changed)
    for name, scorer in METHODS.items():
        config = first_configuration(PROTOCOL["methods"][name]["tuning"])
        original = scorer(bundle, config)
        counterfactual = scorer(relabeled, config)
        assert original.shape == (len(bundle.rows), bundle.service_count), name
        assert np.isfinite(original).all(), name
        np.testing.assert_allclose(original, counterfactual, atol=0, rtol=0, err_msg=name)


def test_all_zero_inputs_have_stable_service_order_and_finite_fallback():
    bundle = synthetic_bundle(zero=True)
    fallback = fallback_scores(bundle)
    assert np.isfinite(fallback).all()
    np.testing.assert_allclose(fallback, 0)
    ranks = ranks_from_scores(normalize_rows(fallback), np.zeros(len(bundle.rows), dtype=int))
    np.testing.assert_array_equal(ranks, np.ones(len(bundle.rows), dtype=int))
    for name, scorer in METHODS.items():
        config = first_configuration(PROTOCOL["methods"][name]["tuning"])
        scores = scorer(bundle, config)
        assert scores.shape == fallback.shape, name
        assert np.isfinite(scores).all(), name


def test_neural_adapters_are_finite_and_ignore_test_labels():
    try:
        from fse2027_neural_baselines import score_once
    except (ImportError, ModuleNotFoundError):
        import pytest
        pytest.skip("base environment has an incomplete torch installation")
    bundle = synthetic_bundle()
    changed = bundle.labels.copy()
    changed[bundle.test] = (changed[bundle.test] + 1) % bundle.service_count
    relabeled = dataclasses.replace(bundle, labels=changed)
    configurations = {
        "TraceAnomaly": {
            "latent_dim": 1,
            "aggregation": "mean",
            "hidden_dim": 4,
            "learning_rate": 0.001,
            "epochs": 2,
        },
        "Eadro": {
            "learning_rate": 0.001,
            "dropout": 0.2,
            "hidden_dim": 8,
            "attention_heads": 2,
            "max_epochs": 3,
            "patience": 2,
            "batch_size": 4,
            "weight_decay": 0.0001,
        },
        "MULAN": {
            "temperature": 0.1,
            "graph_mix": 0.25,
            "hidden_dim": 8,
            "learning_rate": 0.001,
            "max_epochs": 3,
            "patience": 2,
            "batch_size": 4,
            "weight_decay": 0.0001,
        },
    }
    for method, config in configurations.items():
        original, _ = score_once(bundle, method, config, 77)
        counterfactual, _ = score_once(relabeled, method, config, 77)
        assert original.shape == (len(bundle.rows), bundle.service_count), method
        assert np.isfinite(original).all(), method
        np.testing.assert_allclose(original, counterfactual, atol=0, rtol=0, err_msg=method)


def test_llm_parser_and_serializer_have_frozen_fallback_and_no_test_metadata():
    from fse2027_llm_baselines import TelemetryBuilder, canonical, complete_ranking, extract_json

    bundle = synthetic_bundle()
    builder = TelemetryBuilder(bundle)
    payload = canonical(builder.observation(int(bundle.test[0])))
    assert "synthetic-10" not in payload
    assert '"fault_type"' not in payload
    assert '"root_service"' not in payload
    parsed = extract_json('```json\n{"ranking":["s2","unknown","s2"]}\n```')
    assert parsed == {"ranking": ["s2", "unknown", "s2"]}
    ranking, valid = complete_ranking(parsed["ranking"], bundle.services, ["s0", "s1", "s2"])
    assert valid
    assert ranking == ["s2", "s0", "s1"]
    fallback, valid = complete_ranking("not-a-list", bundle.services, ["s0", "s1", "s2"])
    assert not valid
    assert fallback == ["s0", "s1", "s2"]
