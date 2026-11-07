"""Targeted guards for the new inner-validation boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from exp_revision.train_context import _fit_adjacency


def test_context_graph_excludes_validation_incident_calls():
    fold = SimpleNamespace(
        services=["a", "b", "c"], train_ids=["fit", "val"],
        metadata={"calls": {
            "fit": [{"source": "a", "target": "b"}],
            "val": [{"source": "b", "target": "c"}],
        }},
    )
    graph = _fit_adjacency(fold, np.asarray([0]))
    assert graph.tolist() == [[True, True, False],
                              [True, True, False],
                              [False, False, True]]
