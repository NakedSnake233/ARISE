"""Fold-only AIOps2022 preprocessing from original metrics, spans, and log text.

The event manifest contains locations and windows only. Raw event caches are
identity-checked and contain no fitted statistics; Drain, IDF, scalers, feature
selection, and the training call graph are fitted separately for each fold.
"""

from __future__ import annotations

import collections
import datetime as dt
import gzip
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .common import ROOT, config_snapshot, dataset_dir, json_dump
from .preprocess_tt import _normalize, _templates_disjoint
from .splits import UNLABELED_FIELDS, ids_to_indices


RAW_BASE = ROOT / "data/care/aiops2022-original/training_data_with_faults/training_data_with_faults/tar"
METRIC_BASE = Path("/workspace/iwqos2026/data/2022_CCF_AIOps_challenge/raw_data/train_valid")


def _service_name(value: object, services: list[str]) -> str | None:
    name = re.sub(r"-\d+$", "", str(value))
    if name in services:
        return name
    if name.endswith("2") and name[:-1] in services:
        return name[:-1]
    return None


def _cloud(row: dict) -> str:
    return row["incident_id"].split(":", 1)[0]


def _identity(row: dict, services: list[str], step: int, bins: int, metric_names: list[str]) -> dict:
    return {"incident_id": row["incident_id"], "start": row["start"], "end": row["end"],
            "services": services, "step": step, "bins": bins, "metric_names": metric_names}


def _cached(cache: Path, row: dict, services: list[str], step: int, bins: int,
            metric_names: list[str]) -> bool:
    if not (cache / "raw.npz").exists() or not (cache / "logs.jsonl.gz").exists():
        return False
    with np.load(cache / "raw.npz") as archive:
        return "identity_json" in archive.files and json.loads(str(archive["identity_json"].item())) == _identity(
            row, services, step, bins, metric_names)


def _read_cloud_raw(rows: list[dict], indices: list[int], services: list[str],
                    metric_names: list[str], step: int, bins: int, cache_root: Path,
                    targets: list[int] | None = None) -> None:
    # `indices` fixes the original full-cloud parent-span lookup scope.
    # `targets` bounds the materialized event windows and avoids rebuilding
    # caches that have already passed the identity check.
    targets = indices if targets is None else targets
    cloud = _cloud(rows[indices[0]])
    base = RAW_BASE / cloud
    trace_path = base / "trace/all/trace_jaeger-span.csv"
    log_path = base / "log/all/log_filebeat-testbed-log-service.csv"
    for path in (trace_path, log_path):
        if not path.exists() or not path.with_name(path.name + ".verified.json").exists():
            raise FileNotFoundError(f"verified raw AIOps telemetry missing: {path}")
    print(f"raw AIOps cloud {cloud}: {len(targets)} uncached of {len(indices)} incidents", flush=True)
    metric = {i: np.full((len(services), bins, len(metric_names)), np.nan, np.float32) for i in targets}
    trace = {i: np.zeros((len(services), bins, 3), np.float32) for i in targets}
    calls = {i: collections.Counter() for i in targets}
    metric_dir = METRIC_BASE / cloud[:10] / ("cloudbed-" + cloud[-1]) / "raw_metric/container-all"
    for service in services:
        frames = [pd.read_csv(path) for path in sorted(metric_dir.glob("*.csv"))
                  if _service_name(path.stem, services) == service]
        if not frames:
            raise FileNotFoundError(f"raw metrics missing for {cloud}/{service}")
        frame = pd.concat(frames, ignore_index=True).groupby("timestamp", as_index=False).mean(numeric_only=True)
        sid = services.index(service)
        for i in targets:
            row = rows[i]
            window = frame[(frame.timestamp >= row["start"]) & (frame.timestamp < row["end"])].copy()
            if window.empty:
                continue
            window["bin"] = ((window.timestamp - row["start"]) // step).astype(int)
            for bin_index, values in window.groupby("bin"):
                if 0 <= bin_index < bins:
                    metric[i][sid, bin_index] = values.reindex(columns=metric_names).mean().values
    selected = []
    trace_columns = ["timestamp", "cmdb_id", "trace_id", "span_id",
                     "parent_span", "duration", "status_code"]
    # The pandas C parser segfaults on the verified 2022-03-21-cloudbed2 trace
    # CSV after its first chunk; the Python parser reads the same columns
    # without crashing. No rows or fields are intentionally changed.
    for chunk in pd.read_csv(trace_path, usecols=trace_columns, chunksize=200000,
                             engine="python"):
        chunk["timestamp"] = pd.to_numeric(chunk.timestamp) / 1000.0
        mask = np.zeros(len(chunk), dtype=bool)
        for i in targets:
            row = rows[i]
            mask |= (chunk.timestamp >= row["start"]) & (chunk.timestamp < row["end"])
        if mask.any():
            selected.append(chunk.loc[mask].copy())
    if not selected:
        raise ValueError(f"no raw spans within AIOps windows for {cloud}")
    spans = pd.concat(selected, ignore_index=True)
    spans["service"] = spans.cmdb_id.map(lambda value: _service_name(value, services))
    needed_parents = set(zip(spans.trace_id, spans.parent_span))
    parent = {}
    # The original parser resolves each parent from the union of *all* event
    # windows in this cloud. Stream that same scope but retain only parent keys
    # referenced by target child spans. First occurrence wins, as in
    # drop_duplicates([trace_id, span_id]) on file-ordered rows.
    parent_columns = ["timestamp", "cmdb_id", "trace_id", "span_id"]
    for chunk in pd.read_csv(trace_path, usecols=parent_columns, chunksize=200000,
                             engine="python"):
        chunk["timestamp"] = pd.to_numeric(chunk.timestamp) / 1000.0
        mask = np.zeros(len(chunk), dtype=bool)
        for i in indices:
            row = rows[i]
            mask |= (chunk.timestamp >= row["start"]) & (chunk.timestamp < row["end"])
        if not mask.any():
            continue
        for trace_id, span_id, cmdb_id in chunk.loc[mask, ["trace_id", "span_id", "cmdb_id"]].itertuples(index=False, name=None):
            key = (trace_id, span_id)
            if key in needed_parents and key not in parent:
                parent[key] = _service_name(cmdb_id, services)
        if len(parent) == len(needed_parents):
            break
    for i in targets:
        row = rows[i]
        window = spans[(spans.timestamp >= row["start"]) & (spans.timestamp < row["end"])].copy()
        window["bin"] = ((window.timestamp - row["start"]) // step).astype(int)
        for (service, bin_index), group in window.dropna(subset=["service"]).groupby(["service", "bin"]):
            if 0 <= bin_index < bins:
                trace[i][services.index(service), bin_index] = [
                    group.duration.mean() / 1000.0, (group.status_code != 0).mean(), len(group) / step]
        for span in window.itertuples():
            source = parent.get((span.trace_id, span.parent_span))
            target = span.service
            if source in services and target in services and source != target:
                calls[i][source, target] += 1
    del selected, spans, parent
    writers = {}
    try:
        for i in targets:
            event_dir = cache_root / rows[i]["incident_id"].replace(":", "_")
            event_dir.mkdir(parents=True, exist_ok=True)
            writers[i] = gzip.open(event_dir / "logs.jsonl.gz.tmp", "wt")
        for chunk in pd.read_csv(log_path, chunksize=50000):
            chunk["timestamp"] = pd.to_numeric(chunk.timestamp)
            entity_col = next((key for key in ("cmdb_id", "service_name", "service") if key in chunk), None)
            message_col = next((key for key in ("value", "log", "message", "log_message") if key in chunk), None)
            if entity_col is None or message_col is None:
                raise ValueError(f"unsupported log columns: {list(chunk)}")
            for i in targets:
                row = rows[i]
                window = chunk[(chunk.timestamp >= row["start"]) & (chunk.timestamp < row["end"])]
                for record in window.itertuples(index=False):
                    stamp = getattr(record, "timestamp")
                    service = _service_name(getattr(record, entity_col), services)
                    if service is None:
                        continue
                    message = str(getattr(record, message_col))
                    bin_index = int((stamp - row["start"]) // step)
                    if 0 <= bin_index < bins:
                        writers[i].write(json.dumps([services.index(service), bin_index, message]) + "\n")
    finally:
        for writer in writers.values():
            writer.close()
    for i in targets:
        row = rows[i]
        event_dir = cache_root / row["incident_id"].replace(":", "_")
        (event_dir / "logs.jsonl.gz.tmp").replace(event_dir / "logs.jsonl.gz")
        edges = [{"source": u, "target": v, "count": count} for (u, v), count in sorted(calls[i].items())]
        np.savez_compressed(event_dir / "raw.npz", metric=metric[i], trace=trace[i],
                            metric_presence=np.isfinite(metric[i]).any(axis=(1, 2)),
                            calls_json=np.array(json.dumps(edges)),
                            identity_json=np.array(json.dumps(_identity(row, services, step, bins, metric_names))))


def _iter_logs(row: dict, cache_root: Path):
    event_dir = cache_root / row["incident_id"].replace(":", "_")
    with gzip.open(event_dir / "logs.jsonl.gz", "rt") as stream:
        for line in stream:
            yield json.loads(line)


def prepare_fold(fold: int = 0, smoke: bool = False) -> Path:
    from drain3 import TemplateMiner

    started = time.perf_counter()
    start_time = dt.datetime.now(dt.timezone.utc).isoformat()
    unlabeled = Path("splits/aiops_unlabeled_events.json")
    fold_file = Path("splits/aiops_kfold5.json")
    if not unlabeled.exists() or not fold_file.exists():
        raise FileNotFoundError("freeze AIOps fold and label-free event manifests with T1 first")
    rows = json.loads(unlabeled.read_text())
    if any(set(row) != set(UNLABELED_FIELDS) for row in rows):
        raise ValueError("AIOps event manifest has unexpected fields")
    split = json.loads(fold_file.read_text())[f"fold_{fold}"]
    bundle = SimpleNamespace(rows=rows)
    train = ids_to_indices(bundle, split["train"])
    test = ids_to_indices(bundle, split["test"])
    if smoke:
        test = test[:5]
    selected = np.concatenate([train, test])
    metadata = json.loads((dataset_dir("aiops") / "metadata.json").read_text())
    services = list(metadata["services"])
    metric_names = list(metadata["metric_names"])
    step, before = int(metadata["step"]), int(metadata["before"])
    bins = int((metadata["before"] + metadata["after"]) / step)
    cache_root = Path("cache/raw_telemetry_aiops")
    cache_root.mkdir(parents=True, exist_ok=True)
    by_cloud = collections.defaultdict(list)
    for i in selected:
        by_cloud[_cloud(rows[i])].append(int(i))
    for cloud, indices in sorted(by_cloud.items()):
        missing = [i for i in indices if not _cached(
            cache_root / rows[i]["incident_id"].replace(":", "_"), rows[i],
            services, step, bins, metric_names)]
        for start in range(0, len(missing), 6):
            _read_cloud_raw(rows, indices, services, metric_names, step, bins,
                            cache_root, targets=missing[start:start + 6])
    raw_cache_seconds = time.perf_counter() - started
    metric = np.empty((len(selected), len(services), bins, len(metric_names)), np.float32)
    trace = np.empty((len(selected), len(services), bins, 3), np.float32)
    metric_presence = np.empty((len(selected), len(services)), bool)
    calls = {}
    for pos, index in enumerate(selected):
        with np.load(cache_root / rows[index]["incident_id"].replace(":", "_") / "raw.npz") as archive:
            metric[pos], trace[pos], metric_presence[pos] = archive["metric"], archive["trace"], archive["metric_presence"]
            calls[rows[index]["incident_id"]] = json.loads(str(archive["calls_json"].item()))
    miner = TemplateMiner()
    cluster_for_message = {}
    total_messages = 0
    for pos, index in enumerate(train):
        for _, _, message in _iter_logs(rows[index], cache_root):
            total_messages += 1
            if message not in cluster_for_message:
                cluster_for_message[message] = miner.add_log_message(message)["cluster_id"]
        if (pos + 1) % 20 == 0:
            print(f"AIOps Drain training {pos + 1}/{len(train)} in {time.perf_counter() - started:.1f}s", flush=True)
    templates = {cluster.cluster_id: cluster.get_template() for cluster in miner.drain.clusters}
    train_unique_count = len(cluster_for_message)
    drain_assignment_mode = "insertion_id_disjoint_templates"
    if not _templates_disjoint(templates):
        cluster_for_message.clear()
        drain_assignment_mode = "final_template_rematch"
    ids = {cluster_id: j + 1 for j, cluster_id in enumerate(sorted(templates))}
    logs = np.zeros((len(selected), len(services), bins, len(ids) + 1), np.float32)
    log_presence = np.zeros((len(selected), len(services)), bool)
    for pos, index in enumerate(selected):
        for service, bin_index, message in _iter_logs(rows[index], cache_root):
            if message not in cluster_for_message:
                match = miner.match(message)
                cluster_for_message[message] = match.cluster_id if match else None
            cluster_id = cluster_for_message[message]
            logs[pos, service, bin_index, ids[cluster_id] if cluster_id else 0] += 1
            log_presence[pos, service] = True
    train_local = np.arange(len(train))
    docs = logs[train_local].sum(2).reshape(-1, logs.shape[-1])
    idf = np.log((len(docs) + 1) / (1 + (docs > 0).sum(0))) + 1
    logs = (np.log1p(logs) * idf).astype(np.float32)
    metric_available = np.isfinite(metric[train_local]).any(axis=(0, 1, 2))
    if not metric_available.any() or not templates:
        raise ValueError("training fold lacks metric or log features")
    metric = metric[:, :, :, metric_available]
    trace_presence = (trace[:, :, :, 2] > 0).any(axis=2)
    metric, metric_scaler = _normalize(metric, train_local, before // step)
    trace, trace_scaler = _normalize(trace, train_local, before // step)
    availability = np.stack([metric_presence, log_presence, trace_presence], axis=-1).astype(np.uint8)
    training_graph = sorted({(edge["source"], edge["target"]) for index in train
                             for edge in calls[rows[index]["incident_id"]]})
    output = Path("cache/fold_preprocessing/aiops") / f"fold{fold}{'_smoke' if smoke else ''}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "features.npz", metrics=metric, traces=trace, logs=logs, availability=availability)
    json_dump(output / "metadata.json", {
        "train_ids": [rows[i]["incident_id"] for i in train], "test_ids": [rows[i]["incident_id"] for i in test],
        "services": services, "metric_names": [name for name, okay in zip(metric_names, metric_available) if okay],
        "trace_names": ["latency_ms", "error_rate", "call_frequency"],
        "log_templates": ["<UNSEEN>"] + [templates[key] for key in sorted(templates)], "log_idf": idf.tolist(),
        "metric_scaler": metric_scaler, "trace_scaler": trace_scaler,
        "training_graph_edges": training_graph, "calls": calls,
        "before": before, "after": int(metadata["after"]), "step": step,
        "limitations": metadata.get("limitations", []),
        "raw_cache_seconds": raw_cache_seconds, "training_log_lines": total_messages,
        "unique_training_messages_fit_to_drain": train_unique_count,
        "drain_assignment_mode": drain_assignment_mode,
        "drain_config": {key: value for key, value in vars(miner.config).items()
                         if isinstance(value, (str, int, float, bool, type(None)))},
        "elapsed_seconds": time.perf_counter() - started,
        "preprocessing_scope": "all fitted statistics and Drain templates use training fold only",
    })
    run_meta = config_snapshot(
        f"python -m exp_revision.run --task T1 --dataset aiops --prepare-fold {fold}{' --smoke' if smoke else ''}",
        {"task": "T1", "dataset": "aiops", "fold": fold, "smoke": smoke,
         "preprocessing": "train-fold-only Drain, IDF, scalers, and call graph"},
    )
    run_meta.update(start_time=start_time, end_time=dt.datetime.now(dt.timezone.utc).isoformat())
    json_dump(output / "run_meta.json", run_meta)
    return output
