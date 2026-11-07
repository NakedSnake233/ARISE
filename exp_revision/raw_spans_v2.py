"""Label-free span primitives that preserve pairing and exclusive-time dependencies.

Service masks remove spans before reconstructing RPC pairs and child interval
unions. Cloud windows match the original RPC extractor; each incident retains
whole trace IDs within that union so boundary dependencies are not discarded.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, config_snapshot, json_dump, sha256

sys.path.insert(0, str(ROOT / "scripts/care"))
from prepare_aiops import service_name
from prepare_aiops_rpc import operation
from trace_residual import interval_union, summarize_records


def native_table(path: Path, services: list[str]) -> dict:
    index = {name: i for i, name in enumerate(services)}
    stamp, duration, service, kind, parent, child = [], [], [], [], [], []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            trace = json.loads(line)
            spans = {s["spanID"]: s for s in trace["spans"]}
            lookup = {key: len(stamp) + i for i, key in enumerate(spans)}
            for sid, span in spans.items():
                stamp.append(span["startTime"])
                duration.append(span["duration"])
                service.append(index.get(trace["processes"][span["processID"]]["serviceName"], -1))
                tag = next((t["value"] for t in span.get("tags", []) if t["key"] == "span.kind"), None)
                kind.append({"server": 1, "client": 2}.get(tag, 0))
                for ref in span.get("references", []):
                    if (ref.get("refType") == "CHILD_OF" and ref.get("traceID") == span.get("traceID")
                            and ref["spanID"] in lookup):
                        parent.append(lookup[ref["spanID"]]); child.append(lookup[sid])
    return {"start_us": np.asarray(stamp, dtype=np.float64), "duration_us": np.asarray(duration, dtype=np.float64),
            "service": np.asarray(service, dtype=np.int16), "kind": np.asarray(kind, dtype=np.int8),
            "parent": np.asarray(parent, dtype=np.int32), "child": np.asarray(child, dtype=np.int32),
            "native": np.asarray(True)}


def exported_table(frame: pd.DataFrame, services: list[str]) -> dict:
    # The original extractor also retains the last duplicate key, in first-key
    # order. IDs remain strings until this identity mapping is established.
    keyed = {(str(s.trace_id), str(s.span_id)): s for s in frame.itertuples(index=False)}
    rows = list(keyed.values())
    lookup = {(str(s.trace_id), str(s.span_id)): i for i, s in enumerate(rows)}
    parent, child, operations = [], [], []
    codes = {}
    for i, span in enumerate(rows):
        p = lookup.get((str(span.trace_id), str(span.parent_span)))
        if p is not None: parent.append(p); child.append(i)
        name = operation(span.operation_name)
        operations.append(-1 if name in ("", "nan") else codes.setdefault(name, len(codes)))
    index = {name: i for i, name in enumerate(services)}
    return {"start_us": np.asarray([s.timestamp * 1000. for s in rows], dtype=np.float64),
            "duration_us": np.asarray([s.duration for s in rows], dtype=np.float64),
            "service": np.asarray([index.get(service_name(s.cmdb_id), -1) for s in rows], dtype=np.int16),
            "operation": np.asarray(operations, dtype=np.int32),
            "kind": np.zeros(len(rows), dtype=np.int8),
            "parent": np.asarray(parent, dtype=np.int32), "child": np.asarray(child, dtype=np.int32),
            "native": np.asarray(False)}


def rpc_features(table: dict, row: dict, services: list[str], before: float,
                 trace_keep: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Reconstruct from surviving spans, including ancillary unknown services."""
    svc, start_us, duration_us = table["service"], table["start_us"], table["duration_us"]
    stamp, duration = start_us / 1e6, duration_us / 1000.
    trace_keep = np.asarray(trace_keep, dtype=bool)
    if trace_keep.shape != (len(services),): raise ValueError("invalid service trace mask")
    active = np.ones(len(svc), dtype=bool)
    known = svc >= 0
    active[known] = trace_keep[svc[known]]
    parent, child = table["parent"], table["child"]
    select = active[parent] & active[child]
    p, c = parent[select], child[select]
    children = collections.defaultdict(list)
    for a, b in zip(p.tolist(), c.tolist()): children[a].append(b)
    cross = (svc[p] >= 0) & (svc[c] >= 0) & (svc[p] != svc[c])
    if bool(table["native"]):
        cross &= (table["kind"][p] == 2) & (table["kind"][c] == 1)
    degree = np.bincount(p[cross], minlength=len(svc))
    pairs = cross & (degree[p] == 1)
    if not bool(table["native"]):
        op = table["operation"]
        pairs &= (op[p] >= 0) & (op[p] == op[c])
    clients, servers = p[pairs], c[pairs]
    if not bool(table["native"]):
        conflict = np.intersect1d(clients, servers)
        good = (~np.isin(clients, conflict) & ~np.isin(servers, conflict)
                & (duration[clients] >= 0) & (duration[servers] >= 0))
        clients, servers = clients[good], servers[good]
        local_ids = np.unique(servers)
    else:
        local_ids = np.flatnonzero(active & known & (table["kind"] == 1))
    local, edge = collections.defaultdict(list), collections.defaultdict(list)
    for sid in local_ids:
        start = stamp[sid]
        if not row["start"] <= start < row["end"]: continue
        intervals = [(start_us[k], start_us[k] + duration_us[k]) for k in children[sid]
                     if bool(table["native"]) or duration[k] >= 0]
        occupied = interval_union(intervals, start_us[sid], start_us[sid] + duration_us[sid])
        phase = int(start >= row["start"] + before)
        name = services[svc[sid]]
        local[(name, phase, 0)].append(duration[sid])
        local[(name, phase, 1)].append(max(0., duration_us[sid] - occupied) / 1000.)
    for a, b in zip(clients, servers):
        start = stamp[a]
        if row["start"] <= start < row["end"]:
            edge[(services[svc[a]], services[svc[b]], int(start >= row["start"] + before))].append(
                max(0., duration_us[a] - duration_us[b]) / 1000.)
    return summarize_records(local, edge, services)


def cloud_rows(path: Path, rows: list[dict]) -> pd.DataFrame:
    selected = []
    columns = ["timestamp", "cmdb_id", "trace_id", "span_id", "parent_span", "duration", "operation_name"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100000, engine="python",
                             dtype={"trace_id": str, "span_id": str, "parent_span": str}):
        keep = np.zeros(len(chunk), dtype=bool)
        for row in rows:
            keep |= (chunk.timestamp >= row["start"] * 1000.) & (chunk.timestamp < row["end"] * 1000.)
        if keep.any(): selected.append(chunk.loc[keep, columns])
    if not selected: raise ValueError("cloud span windows contain no observations")
    return pd.concat(selected, ignore_index=True)


def prepare(dataset: str, incident_ids: list[str], output: Path) -> Path:
    started = time.perf_counter()
    start_time = dt.datetime.now(dt.timezone.utc).isoformat()
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable span output exists")
    output.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / f"splits/{dataset}_unlabeled_events.json"
    rows = json.loads(manifest.read_text()); by_id = {r["incident_id"]: r for r in rows}
    meta_path = ROOT / ("data/care/trainticket-prepared/metadata.json" if dataset == "tt"
                        else "data/care/aiops2022-paper128-prepared/metadata.json")
    meta = json.loads(meta_path.read_text()); services = meta["services"]
    inputs = [manifest, meta_path]; frame = None
    if dataset == "aiops":
        clouds = {key.split(":")[0] for key in incident_ids}
        if len(clouds) != 1: raise ValueError("one exported cloud per cache job")
        cloud = next(iter(clouds))
        raw = ROOT / "data/care/aiops2022-original/training_data_with_faults/training_data_with_faults/tar" / cloud / "trace/all/trace_jaeger-span.csv"
        inputs.append(raw)
        frame = cloud_rows(raw, [r for r in rows if r["incident_id"].split(":")[0] == cloud])
    rpc_path = ROOT / ("artifacts/care/trace-residual-trainticket/features.npz" if dataset == "tt"
                       else "artifacts/care/full-v3/aiops2022/rpc/features.npz")
    with np.load(rpc_path) as archive: legacy = archive["x"]
    inputs.append(rpc_path)
    original = {r["incident_id"]: i for i, r in enumerate(rows)}
    files, rpc, differences, nodes = {}, [], {}, {}
    for i, key in enumerate(incident_ids):
        row = by_id[key]
        if dataset == "tt":
            raw = ROOT / Path(row["source"]).parent / "traces.jsonl.gz"
            inputs.append(raw); table = native_table(raw, services)
        else:
            # Pairing/conflicts are trace-local. Preserve complete trace-ID
            # closure from the cloud union, including boundary parents/children.
            inside = (frame.timestamp >= row["start"] * 1000.) & (frame.timestamp < row["end"] * 1000.)
            trace_ids = frame.loc[inside, "trace_id"].unique()
            table = exported_table(frame.loc[frame.trace_id.isin(trace_ids)], services)
        values, names = rpc_features(table, row, services, meta["before"], np.ones(len(services)))
        difference = float(np.max(np.abs(values - legacy[original[key]])))
        # Preserve native microsecond arithmetic for exclusive interval unions.
        if difference > 1e-5: raise ValueError(f"S0 RPC parity failed for {key}: {difference}")
        path = output / f"event-{i:03d}.npz"
        np.savez_compressed(path, **table)
        files[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        rpc.append(values); differences[key] = difference; nodes[key] = len(table["start_us"])
        print(json.dumps({"incident_id": key, "spans": nodes[key], "s0_difference": difference}), flush=True)
    # Queue acceptance witness: original RPC server-observed descriptor only.
    # It is not an RCA prediction and is excluded from experiment evaluation.
    np.savez_compressed(output / "scores.npz", scores=np.stack(rpc)[None, :, :, 0],
                        rpc_features=np.stack(rpc), incident_ids=incident_ids, scenario_keys=["S0_none"])
    config = {"task": "raw_span_preprocessing", "dataset": dataset, "incident_ids": incident_ids}
    receipt = config_snapshot("python -m exp_revision.raw_spans_v2", config)
    source = [ROOT / name for name in ("exp_revision/raw_spans_v2.py", "scripts/care/trace_residual.py",
              "scripts/care/prepare_aiops_rpc.py", "scripts/care/prepare_aiops.py")]
    receipt.update(status="completed_span_cache", config=config, start_time=start_time,
                   end_time=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                   incident_files=files, feature_names=names, span_counts=nodes, s0_rpc_max_abs_differences=differences,
                   input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in dict.fromkeys(inputs)},
                   source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source},
                   scores_sha256=sha256(output / "scores.npz"), witness="RPC server-observed; not RCA predictions")
    json_dump(output / "run_meta.json", receipt)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    p.add_argument("--incident-ids", required=True, help="JSON list of unlabeled event IDs")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); print(prepare(a.dataset, json.loads(a.incident_ids), a.output))


if __name__ == "__main__": main()
