"""Recompute fold evidence after shared service/modality masks.

Frozen fold normalization and Drain features stay fixed. Whole-service removal
commutes with their per-service binning; RPC dependencies are reconstructed
from raw span primitives. No held-out label file is opened by this module.
"""
from __future__ import annotations
import collections
from functools import lru_cache
import json
import sys

import numpy as np

from .common import ROOT, sha256
from .fold_data import load_fold
from .raw_spans_v2 import rpc_features

sys.path.insert(0, str(ROOT / "scripts/care"))
from evidence_reranker import extract
from robust_evidence import aggregate_type_agnostic, apply_availability, feature_groups

PIPELINES = ("cached_leaky", "recompute_ungated", "family_gated", "edge_gated")
COUNTS = ["observable:n_valid_in_edges", "observable:n_valid_out_edges", "observable:n_valid_neighbors"]


def available_cross_shock(shock, availability):
    """Median/share over observed services, zero with fewer than two."""
    relative, share = np.zeros_like(shock), np.zeros_like(shock)
    for i, valid in enumerate(availability.astype(bool)):
        if valid.sum() >= 2:
            values = shock[i, valid]
            relative[i, valid] = values - np.median(values, axis=0, keepdims=True)
            share[i, valid] = values / (values.sum(axis=0, keepdims=True) + 1e-6)
    return relative, share


def edge_gate(x, names, availability, rows, services):
    """Keep local evidence, observable cross contrasts and eligible neighbors."""
    x = np.asarray(x, dtype=np.float32).copy()
    groups = feature_groups(names)
    # trace contains graph/RPC columns in the old grouping. They are already
    # reconstructed on observable endpoints, and local trace gating agrees.
    for m, name in enumerate(("metric", "log", "trace")):
        x[:, :, groups[name]] *= availability[:, :, m, None]
    shocks = {}
    for prefix, modality in (("metric", 0), ("raw_metric", 0), ("trace", 2)):
        channels = [n.split(":")[1] for n in names if n.startswith(prefix + ":") and n.endswith(":positive_z")]
        if not channels: continue
        values = np.stack([np.maximum(x[:, :, names.index(f"{prefix}:{c}:positive_z")],
                                     x[:, :, names.index(f"{prefix}:{c}:negative_z")]) for c in channels], axis=-1)
        relative, share = available_cross_shock(values, availability[:, :, modality])
        for c, channel in enumerate(channels):
            x[:, :, names.index(f"{prefix}:{channel}:relative_shock")] = relative[:, :, c]
            x[:, :, names.index(f"{prefix}:{channel}:shock_share")] = share[:, :, c]
        shocks[prefix] = values.max(-1)
    metric = shocks.get("raw_metric", shocks.get("metric", np.zeros(x.shape[:2])))
    total = metric + shocks.get("trace", np.zeros(x.shape[:2]))
    index = {s: i for i, s in enumerate(services)}
    neighbor_count = np.zeros(x.shape[:2], dtype=np.float32)
    for i, row in enumerate(rows):
        incoming, outgoing = collections.defaultdict(set), collections.defaultdict(set)
        for call in row["calls"]:
            u, v = index.get(call["source"]), index.get(call["target"])
            if u is None or v is None or not availability[i, u, 2] or not availability[i, v, 2]: continue
            incoming[v].add(u); outgoing[u].add(v)
        eligible = availability[i, :, 0].astype(bool) & availability[i, :, 2].astype(bool)
        for s in range(len(services)):
            union = (incoming[s] | outgoing[s]) & set(np.flatnonzero(eligible).tolist())
            neighbor_count[i, s] = len(union)
            for prefix, neighbors in (("callers", incoming[s]), ("callees", outgoing[s])):
                qualified = [n for n in neighbors if eligible[n]]
                maximum = float(total[i, qualified].max()) if qualified else 0.
                x[i, s, names.index(prefix + ":degree")] = len(neighbors)
                x[i, s, names.index(prefix + ":max_shock")] = maximum
                x[i, s, names.index(prefix + ":local_minus_neighbor")] = total[i, s] - maximum if availability[i, s, 2] else 0.
    counts = np.stack([x[:, :, names.index("rpc:incoming:degree")],
                       x[:, :, names.index("rpc:outgoing:degree")], neighbor_count], axis=-1)
    return np.concatenate([x, counts], axis=-1), names + COUNTS


class MaskedEvidence:
    def __init__(self, dataset, fold):
        self.data = load_fold(dataset, fold)
        self.ids = self.data.train_ids + self.data.test_ids
        rows = json.loads((ROOT / f"splits/{dataset}_unlabeled_events.json").read_text())
        self.rows = {r["incident_id"]: r for r in rows}
        folder = ROOT / f"cache/fold_evidence_v2/{dataset}/fold{fold}"
        meta = json.loads((folder / "metadata.json").read_text())
        with np.load(folder / "features.npz") as archive:
            if archive["incident_ids"].tolist() != self.ids: raise ValueError("evidence fold identity changed")
            self.cached = archive["x"].copy()
        self.names = meta["feature_names"]
        self.inputs = {str(p.relative_to(ROOT)): sha256(p) for p in (folder / "features.npz", folder / "metadata.json")}
        self.raw_metrics = None
        if dataset == "tt":
            path = ROOT / "cache/raw_metric_tt.npz"
            with np.load(path) as archive:
                original = {key: i for i, key in enumerate(archive["incident_ids"].tolist())}
                self.raw_metrics = archive["metric"][[original[key] for key in self.ids]]
            self.inputs[str(path.relative_to(ROOT))] = sha256(path)
        state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
        self.spans = {}
        for key, row in state["jobs"].items():
            if not key.startswith(f"full_rpc_cache_{dataset}_") or row["status"] != "completed": continue
            path = ROOT / row["output"] / "run_meta.json"
            receipt = json.loads(path.read_text())
            if receipt["status"] != "completed_span_cache": raise ValueError("unverified span cache")
            self.inputs[str(path.relative_to(ROOT))] = sha256(path)
            for event, entry in receipt["incident_files"].items():
                if event in self.spans: raise ValueError("duplicate raw span cache event")
                self.spans[event] = entry
        if set(self.ids) != set(self.spans): raise ValueError("full label-free span cache is incomplete")

    @lru_cache(maxsize=160)
    def table(self, key):
        entry = self.spans[key]
        if sha256(ROOT / entry["path"]) != entry["sha256"]: raise ValueError("raw span primitive changed")
        self.inputs[entry["path"]] = entry["sha256"]
        with np.load(ROOT / entry["path"]) as archive: return {key: archive[key].copy() for key in archive.files}

    @lru_cache(maxsize=16000)
    def rpc(self, key, mask_bytes):
        # Log/metric masking and the four pipelines often share trace masks.
        # Reuse only identical raw incident + endpoint eligibility, in memory.
        return rpc_features(self.table(key), self.rows[key], self.data.services,
                            self.data.metadata["before"], np.frombuffer(mask_bytes, dtype=np.uint8))

    def recompute(self, indices, masks, pipeline):
        if pipeline not in PIPELINES: raise ValueError("unknown evidence pipeline")
        indices = np.asarray(indices, dtype=int)
        masks = np.asarray(masks, dtype=np.uint8)
        a0 = self.data.availability[indices]
        if masks.shape != a0.shape or not np.isin(masks, (0, 1)).all(): raise ValueError("invalid evidence mask")
        availability = a0 * masks
        if pipeline == "cached_leaky": return self.cached[indices].copy(), self.names, availability
        features = {name: getattr(self.data, name)[indices] * masks[:, :, m, None, None]
                    for name, m in (("metrics", 0), ("logs", 1), ("traces", 2))}
        raw = None if self.raw_metrics is None else self.raw_metrics[indices] * masks[:, :, 0, None, None]
        trace_keep = availability[:, :, 2] if pipeline == "edge_gated" else masks[:, :, 2]
        index = {s: i for i, s in enumerate(self.data.services)}
        rows = []
        for i, position in enumerate(indices):
            key = self.ids[position]
            calls = [c for c in self.data.metadata["calls"][key]
                     if c["source"] in index and c["target"] in index
                     and trace_keep[i, index[c["source"]]] and trace_keep[i, index[c["target"]]]]
            rows.append({"incident_id": key, "calls": calls})
        x, names = extract(features, self.data.metadata, rows, raw)
        rpc = []
        for i, position in enumerate(indices):
            key = self.ids[position]
            values, rpc_names = self.rpc(key, np.asarray(trace_keep[i], dtype=np.uint8).tobytes())
            rpc.append(values)
        x = np.concatenate([x, np.stack(rpc)], axis=-1)
        names += rpc_names
        if names != self.names: raise ValueError("recomputed evidence feature order changed")
        if pipeline == "edge_gated": x, names = edge_gate(x, names, availability, rows, self.data.services)
        return np.nan_to_num(x).astype(np.float32), names, availability


def transform(x, names, availability, pipeline, *, generic=False, append_flags=True):
    if generic: x, names = aggregate_type_agnostic(x, names)
    if pipeline == "family_gated":
        return apply_availability(x, names, availability, append_flags=append_flags)
    if append_flags: x = np.concatenate([x, availability], axis=-1)
    return np.nan_to_num(x).astype(np.float32)
