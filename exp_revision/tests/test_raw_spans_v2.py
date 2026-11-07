import collections
import gzip
import json

import numpy as np
import pandas as pd

from exp_revision.raw_spans_v2 import exported_table, native_table, rpc_features
from prepare_aiops_rpc import records
from trace_residual import summarize_records, trace_records


def summary(local_records, edge_records):
    local, edge = collections.defaultdict(list), collections.defaultdict(list)
    for stamp, service, duration, exclusive in local_records:
        phase = int(stamp >= 2)
        local[(service, phase, 0)].append(duration)
        local[(service, phase, 1)].append(exclusive)
    for stamp, source, target, gap in edge_records:
        edge[(source, target, int(stamp >= 2))].append(gap)
    return summarize_records(local, edge, ["a", "b", "c"])


def test_native_spans_recompute_exclusive_time_and_masked_edges(tmp_path):
    traces = []
    for i in range(6):
        start = (1 if i < 3 else 2) * 1_000_000 + i * 100_000
        duration = 1000 if i < 3 else 10000
        spans = []
        for sid, process, kind, length, parent, offset in [
            ("client", "a", "client", duration * 2, None, 0),
            ("server", "b", "server", duration, "client", 0),
            ("child", "c", "internal", duration * .4, "server", 100)]:
            spans.append({"spanID": sid, "traceID": str(i), "processID": process,
                "startTime": start + offset, "duration": length,
                "tags": [{"key": "span.kind", "value": kind}],
                "references": [] if parent is None else [{"refType": "CHILD_OF", "traceID": str(i), "spanID": parent}]})
        traces.append({"spans": spans, "processes": {s: {"serviceName": s} for s in ["a", "b", "c"]}})
    path = tmp_path / "traces.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for trace in traces: f.write(json.dumps(trace) + "\n")
    table = native_table(path, ["a", "b", "c"])
    row = {"start": 1, "end": 3}
    expected_local, expected_edge = [], []
    for trace in traces:
        local, edge = trace_records(trace, {"a", "b", "c"})
        expected_local.extend(local); expected_edge.extend(edge)
    expected, names = summary(expected_local, expected_edge)
    actual, _ = rpc_features(table, row, ["a", "b", "c"], 1, [1, 1, 1])
    np.testing.assert_array_equal(actual, expected)
    no_child, _ = rpc_features(table, row, ["a", "b", "c"], 1, [1, 1, 0])
    assert no_child[1, names.index("rpc:exclusive:median:post_log")] == np.float32(np.log1p(10))
    no_parent, _ = rpc_features(table, row, ["a", "b", "c"], 1, [0, 1, 1])
    assert no_parent[:, names.index("rpc:incident:degree")].sum() == 0
    assert no_parent[1, names.index("rpc:server:observed")] == 1


def test_exported_pair_dependency_is_removed_before_local_inference():
    values = []
    for i in range(6):
        stamp = (1000 if i < 3 else 2000) + i * 100
        duration = 1000 if i < 3 else 10000
        for sid, service, length, parent, op in [
            ("p", "a", 2 * duration, "none", "method"),
            ("s", "b", duration, "p", "method"),
            ("c", "c", .4 * duration, "s", "other")]:
            values.append(dict(timestamp=stamp, cmdb_id=service, trace_id=str(i), span_id=sid,
                               parent_span=parent, duration=length, operation_name=op))
    frame = pd.DataFrame(values)
    # AIOps service normalization is specific to its pod naming; use known
    # services directly here to test the real pairing and interval algorithm.
    from unittest.mock import patch
    with patch("exp_revision.raw_spans_v2.service_name", lambda s: s), patch("prepare_aiops_rpc.service_name", lambda s: s):
        table = exported_table(frame, ["a", "b", "c"])
        ls, es, _ = records(frame)
    expected, names = summary(ls, es)
    actual, _ = rpc_features(table, {"start": 1, "end": 3}, ["a", "b", "c"], 1, [1, 1, 1])
    np.testing.assert_array_equal(actual, expected)
    masked, _ = rpc_features(table, {"start": 1, "end": 3}, ["a", "b", "c"], 1, [0, 1, 1])
    assert masked[1, names.index("rpc:server:observed")] == 0
    assert masked[:, names.index("rpc:incident:degree")].sum() == 0
