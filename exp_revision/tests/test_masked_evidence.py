from types import SimpleNamespace
import numpy as np

from exp_revision.masked_evidence import MaskedEvidence, available_cross_shock
from exp_revision.raw_spans_v2 import rpc_features
from evidence_reranker import extract


def toy_pipeline():
    services = ["a", "b", "c"]
    metrics = np.array([[[[0.], [1.]], [[0.], [4.]], [[0.], [9.]]]], dtype=np.float32)
    traces = metrics.copy(); traces[:, :, 1, 0] = [.5, 1, 1.5]
    logs = np.zeros((1, 3, 2, 1), dtype=np.float32)
    calls = [{"source": a, "target": b} for a, b in [("a", "b"), ("b", "c"), ("a", "c")]]
    metadata = dict(before=1, step=1, services=services, metric_names=["kpi"], trace_names=["latency"],
                    log_templates=["normal"], calls={"event": calls})
    start, duration, service, kind, parent, child = [], [], [], [], [], []
    for i in range(6):
        offset = len(start); t = (1 if i < 3 else 2) * 1e6 + i * 100000
        factor = 1 if i < 3 else 10
        start.extend([t] * 4); duration.extend(np.array([2000, 1000, 800, 400]) * factor)
        service.extend([0, 1, 1, 2]); kind.extend([2, 1, 2, 1])
        parent.extend([offset, offset + 1, offset + 2]); child.extend([offset + 1, offset + 2, offset + 3])
    table = {"start_us": np.array(start), "duration_us": np.array(duration), "service": np.array(service),
             "kind": np.array(kind), "parent": np.array(parent), "child": np.array(child), "native": np.array(True)}
    row = {"start": 1, "end": 3}
    x, names = extract(dict(metrics=metrics, traces=traces, logs=logs), metadata, [{"calls": calls}])
    rpc, rpc_names = rpc_features(table, row, services, 1, [1, 1, 1])
    pipe = MaskedEvidence.__new__(MaskedEvidence)
    pipe.data = SimpleNamespace(services=services, metadata=metadata, availability=np.ones((1, 3, 3)),
                                metrics=metrics, traces=traces, logs=logs)
    pipe.ids = ["event"]; pipe.rows = {"event": row}; pipe.names = names + rpc_names
    pipe.cached = np.concatenate([x, rpc[None]], -1); pipe.raw_metrics = None
    pipe.table = lambda _: table
    return pipe


def test_vd_three_service_hand_values_and_masked_endpoint():
    pipe = toy_pipeline()
    masks = np.ones((1, 3, 3), dtype=np.uint8); masks[0, 2, 2] = 0
    x, names, _ = pipe.recompute([0], masks, "edge_gated")
    def column(name): return x[0, :, names.index(name)]
    np.testing.assert_array_equal(column("observable:n_valid_in_edges"), [0, 1, 0])
    np.testing.assert_array_equal(column("observable:n_valid_out_edges"), [1, 0, 0])
    np.testing.assert_array_equal(column("observable:n_valid_neighbors"), [1, 1, 0])
    np.testing.assert_array_equal(column("callers:max_shock"), [0, 3, 0])
    np.testing.assert_array_equal(column("callees:max_shock"), [10, 0, 0])
    np.testing.assert_array_equal(column("callers:local_minus_neighbor"), [3, 7, 0])
    np.testing.assert_array_equal(column("callees:local_minus_neighbor"), [-7, 10, 0])
    np.testing.assert_array_equal(column("trace:latency:relative_shock"), [-.5, .5, 0])
    np.testing.assert_array_equal(column("metric:kpi:relative_shock"), [-6, 0, 10])
    np.testing.assert_allclose(column("metric:kpi:shock_share"), np.array([2, 8, 18]) / 28, atol=1e-7)
    assert column("rpc:incident:degree").tolist() == [1, 1, 0]


def test_cross_service_empty_or_singleton_returns_zero():
    shock = np.array([[[2.], [8.], [18.]], [[1.], [2.], [3.]]])
    relative, share = available_cross_shock(shock, np.array([[1, 0, 0], [0, 0, 0]]))
    assert not relative.any() and not share.any()


def test_recompute_s0_identity_and_intentional_cached_leak():
    pipe = toy_pipeline()
    masks = np.ones((1, 3, 3), dtype=np.uint8)
    x, names, _ = pipe.recompute([0], masks, "family_gated")
    np.testing.assert_array_equal(x, pipe.cached)
    assert names == pipe.names
    masks[0, 1, 2] = 0
    leaky, _, _ = pipe.recompute([0], masks, "cached_leaky")
    fresh, _, _ = pipe.recompute([0], masks, "recompute_ungated")
    np.testing.assert_array_equal(leaky, pipe.cached)
    assert leaky[0, 0, names.index("rpc:outgoing:degree")] == 1
    assert fresh[0, 0, names.index("rpc:outgoing:degree")] == 0
