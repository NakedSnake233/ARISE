"""Train-fold-only preprocessing of raw TrainTicket telemetry.

No root or fault-type field is read here. The split is supplied by incident ID;
the training set alone fits Drain, IDF, metric/trace scalers, and G_tr.
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

from .common import ROOT, config_snapshot, dataset_dir, json_dump
from .splits import ids_to_indices, UNLABELED_FIELDS


METRICS = [
    "container_cpu_usage_seconds_total", "container_memory_working_set_bytes",
    "container_memory_usage_bytes", "container_network_receive_bytes_total",
    "container_network_transmit_bytes_total", "container_fs_reads_bytes_total",
    "container_fs_writes_bytes_total",
]


def _timestamp(value: str) -> float:
    value = re.sub(r"(\.\d{6})\d+(?=Z$|[+-]\d{2}:\d{2}$)", r"\1", value)
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("log timestamp lacks timezone")
    return parsed.timestamp()


def _service_for_pod(name: str, services: list[str]) -> int | None:
    for i, service in enumerate(services):
        if name.startswith(service + "-"):
            return i
    return None


def _metric_and_trace(row: dict, services: list[str], step: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    folder = ROOT / Path(row["source"]).parent
    metric = np.full((len(services), bins, len(METRICS)), np.nan, dtype=np.float32)
    trace = np.zeros((len(services), bins, 3), dtype=np.float32)
    metric_presence = np.zeros(len(services), dtype=bool)
    with gzip.open(folder / "metrics.json.gz", "rt") as f:
        series = json.load(f)["data"]["result"]
    bucket = collections.defaultdict(list)
    for item in series:
        labels = item["metric"]
        service = _service_for_pod(labels.get("pod", ""), services)
        name = labels.get("__name__")
        if service is None or name not in METRICS:
            continue
        if "network" not in name and labels.get("container") not in (None, "", services[service]):
            continue
        values = np.asarray(item["values"], dtype=float)
        if len(values) < 2:
            continue
        if name.endswith("_total"):
            values = np.column_stack([
                values[1:, 0],
                np.maximum(0, np.diff(values[:, 1])) / np.maximum(1e-6, np.diff(values[:, 0])),
            ])
        for stamp, value in values:
            bin_index = int((stamp - row["start"]) // step)
            if 0 <= bin_index < bins and np.isfinite(value):
                bucket[service, bin_index, METRICS.index(name)].append(float(value))
                metric_presence[service] = True
    for key, values in bucket.items():
        metric[key] = float(np.mean(values))
    latencies = collections.defaultdict(list)
    errors = collections.Counter()
    calls = collections.Counter()
    with gzip.open(folder / "traces.jsonl.gz", "rt") as f:
        for line in f:
            record = json.loads(line)
            processes = record["processes"]
            lookup = {span["spanID"]: processes[span["processID"]]["serviceName"] for span in record["spans"]}
            for span in record["spans"]:
                service_name = lookup[span["spanID"]]
                bin_index = int((span["startTime"] / 1e6 - row["start"]) // step)
                if service_name not in services or not 0 <= bin_index < bins:
                    continue
                service = services.index(service_name)
                key = (service, bin_index)
                latencies[key].append(float(span["duration"]) / 1000)
                errors[key] += any(tag.get("key") == "error" and tag.get("value") is True for tag in span.get("tags", []))
                for reference in span.get("references", []):
                    parent = lookup.get(reference["spanID"])
                    if reference.get("refType") == "CHILD_OF" and parent in services and parent != service_name:
                        calls[parent, service_name] += 1
    for key, values in latencies.items():
        trace[key] = [float(np.mean(values)), errors[key] / len(values), len(values) / step]
    call_list = [{"source": u, "target": v, "count": count} for (u, v), count in sorted(calls.items())]
    return metric, trace, metric_presence, call_list


def _iter_logs(row: dict, services: list[str], step: int, bins: int):
    folder = ROOT / Path(row["source"]).parent
    for path in sorted(folder.glob("*.log.gz")):
        service = _service_for_pod(path.name, services)
        if service is None:
            continue
        with gzip.open(path, "rt") as f:
            for line in f:
                fields = line.rstrip().split(" ", 1)
                if len(fields) != 2:
                    continue
                try:
                    stamp = _timestamp(fields[0])
                except ValueError:
                    continue
                bin_index = int((stamp - row["start"]) // step)
                if 0 <= bin_index < bins:
                    message = re.sub(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]\d+\s*", "", fields[1])
                    yield service, bin_index, message


def _normalize(raw: np.ndarray, train: np.ndarray, before: int) -> tuple[np.ndarray, dict]:
    with np.errstate(invalid="ignore"):
        baseline = raw[train, :, :before, :]
        mean = np.nanmean(baseline, axis=(0, 2), keepdims=True)
        std = np.nanstd(baseline, axis=(0, 2), keepdims=True)
    observed = np.isfinite(mean)
    mean = np.nan_to_num(mean)
    std = np.maximum(np.nan_to_num(std, nan=1.0), 1e-6)
    values = np.clip((raw - mean) / std, -20, 20)
    values = np.where(observed, values, 0.0)
    return np.nan_to_num(values).astype(np.float32), {"mean": mean.tolist(), "std": std.tolist(), "available": observed.tolist()}


def _templates_disjoint(templates: dict[int, str]) -> bool:
    """Check when insertion-time IDs must equal final Drain matching IDs."""
    token_lists = [template.split() for template in templates.values()]
    for left_index, left in enumerate(token_lists):
        for right in token_lists[left_index + 1:]:
            if len(left) == len(right) and all(a == b or a == "<*>" or b == "<*>" for a, b in zip(left, right)):
                return False
    return True


def prepare_fold(fold: int = 0, smoke: bool = False) -> Path:
    from drain3 import TemplateMiner

    started = time.perf_counter()
    start_time = dt.datetime.now(dt.timezone.utc).isoformat()
    data_dir = dataset_dir("tt")
    unlabeled_path = Path("splits/tt_unlabeled_events.json")
    if not unlabeled_path.exists():
        raise FileNotFoundError(f"Freeze a label-free event manifest with T1 first: {unlabeled_path}")
    rows = json.loads(unlabeled_path.read_text())
    if any(set(row) != set(UNLABELED_FIELDS) for row in rows):
        raise ValueError("label-free event manifest has unexpected fields")
    metadata = json.loads((data_dir / "metadata.json").read_text())
    bundle = SimpleNamespace(rows=rows)
    split_path = Path("splits/tt_kfold5.json")
    if not split_path.exists():
        raise FileNotFoundError(f"Freeze folds with T1 first: {split_path}")
    split = json.loads(split_path.read_text())[f"fold_{fold}"]
    train = ids_to_indices(bundle, split["train"])
    test = ids_to_indices(bundle, split["test"])
    if smoke:
        test = test[:5]
    selected = np.concatenate([train, test])
    services = list(metadata["services"])
    step, before = int(metadata["step"]), int(metadata["before"])
    bins = int((metadata["before"] + metadata["after"]) / step)
    train_local = np.arange(len(train))
    metric = np.empty((len(selected), len(services), bins, len(METRICS)), dtype=np.float32)
    trace = np.empty((len(selected), len(services), bins, 3), dtype=np.float32)
    metric_presence = np.empty((len(selected), len(services)), dtype=bool)
    calls = {}
    raw_cache = Path("cache/raw_telemetry_tt")
    raw_cache.mkdir(parents=True, exist_ok=True)
    for pos, index in enumerate(selected):
        cached = raw_cache / f"case-{int(index):03d}.npz"
        cache_identity = {
            "incident_id": rows[index]["incident_id"], "source": rows[index]["source"],
            "start": rows[index]["start"], "end": rows[index]["end"],
            "step": step, "bins": bins, "services": services,
        }
        reusable = False
        if cached.exists():
            with np.load(cached) as archive:
                if "identity_json" in archive.files and json.loads(str(archive["identity_json"].item())) == cache_identity:
                    metric[pos] = archive["metric"]
                    trace[pos] = archive["trace"]
                    metric_presence[pos] = archive["metric_presence"]
                    calls[rows[index]["incident_id"]] = json.loads(str(archive["calls_json"].item()))
                    reusable = True
        if not reusable:
            metric[pos], trace[pos], metric_presence[pos], calls[rows[index]["incident_id"]] = _metric_and_trace(rows[index], services, step, bins)
            np.savez_compressed(cached, metric=metric[pos], trace=trace[pos], metric_presence=metric_presence[pos],
                                calls_json=np.array(json.dumps(calls[rows[index]["incident_id"]])),
                                identity_json=np.array(json.dumps(cache_identity)))
        if (pos + 1) % 20 == 0:
            print(f"raw metric/trace {pos + 1}/{len(selected)} in {time.perf_counter() - started:.1f}s", flush=True)
    miner = TemplateMiner()
    cluster_for_message = {}
    total_messages = 0
    for train_pos, index in enumerate(train):
        for _, _, message in _iter_logs(rows[index], services, step, bins):
            total_messages += 1
            if message not in cluster_for_message:
                cluster_for_message[message] = miner.add_log_message(message)["cluster_id"]
        if (train_pos + 1) % 20 == 0:
            print(f"Drain training {train_pos + 1}/{len(train)}: {len(cluster_for_message)} unique in {time.perf_counter() - started:.1f}s", flush=True)
    train_unique_count = len(cluster_for_message)
    print(f"Drain fitted {train_unique_count} unique of {total_messages} training messages in {time.perf_counter() - started:.1f}s", flush=True)
    templates = {cluster.cluster_id: cluster.get_template() for cluster in miner.drain.clusters}
    drain_assignment_mode = "insertion_id_disjoint_templates"
    if not _templates_disjoint(templates):
        # The original parser matches against the final templates. When two
        # templates can match the same line, rematch each distinct line once.
        cluster_for_message.clear()
        drain_assignment_mode = "final_template_rematch"
    ids = {cluster_id: j + 1 for j, cluster_id in enumerate(sorted(templates))}
    logs = np.zeros((len(selected), len(services), bins, len(ids) + 1), dtype=np.float32)
    log_presence = np.zeros((len(selected), len(services)), dtype=bool)
    for pos, index in enumerate(selected):
        for service, bin_index, message in _iter_logs(rows[index], services, step, bins):
            if message not in cluster_for_message:
                match = miner.match(message)
                cluster_for_message[message] = match.cluster_id if match else None
            cluster_id = cluster_for_message[message]
            logs[pos, service, bin_index, ids[cluster_id] if cluster_id else 0] += 1
            log_presence[pos, service] = True
        if (pos + 1) % 20 == 0:
            print(f"log encoding {pos + 1}/{len(selected)} in {time.perf_counter() - started:.1f}s", flush=True)
    docs = logs[train_local].sum(2).reshape(-1, logs.shape[-1])
    idf = np.log((len(docs) + 1) / (1 + (docs > 0).sum(0))) + 1
    logs = (np.log1p(logs) * idf).astype(np.float32)
    metric_available = np.isfinite(metric[train_local]).any(axis=(0, 1, 2))
    if not metric_available.any() or not templates:
        raise ValueError("training fold lacks metric or log features")
    metric = metric[:, :, :, metric_available]
    trace_presence = np.any(trace[:, :, :, 2] > 0, axis=2)
    metric, metric_scaler = _normalize(metric, train_local, before // step)
    trace, trace_scaler = _normalize(trace, train_local, before // step)
    availability = np.stack([metric_presence, log_presence, trace_presence], axis=-1).astype(np.uint8)
    training_graph = sorted({(c["source"], c["target"]) for index in train for c in calls[rows[index]["incident_id"]]})
    output = Path("cache/fold_preprocessing/tt") / f"fold{fold}{'_smoke' if smoke else ''}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "features.npz", metrics=metric, logs=logs, traces=trace, availability=availability)
    json_dump(output / "metadata.json", {
        "train_ids": [rows[i]["incident_id"] for i in train], "test_ids": [rows[i]["incident_id"] for i in test],
        "services": services, "metric_names": [m for m, ok in zip(METRICS, metric_available) if ok],
        "trace_names": ["latency_ms", "error_rate", "call_frequency"],
        "log_templates": ["<UNSEEN>"] + [templates[k] for k in sorted(templates)],
        "log_idf": idf.tolist(), "metric_scaler": metric_scaler, "trace_scaler": trace_scaler,
        "training_graph_edges": training_graph, "calls": calls,
        "before": before, "after": int(metadata["after"]), "step": step,
        "limitations": metadata.get("limitations", []),
        "preprocessing_scope": "all fitted statistics and Drain templates use training fold only",
        "raw_parse_cache": str(raw_cache), "training_log_lines": total_messages,
        "unique_training_messages_fit_to_drain": train_unique_count,
        "drain_assignment_mode": drain_assignment_mode,
        "drain_config": {key: value for key, value in vars(miner.config).items()
                         if isinstance(value, (str, int, float, bool, type(None)))},
        "elapsed_seconds": time.perf_counter() - started,
    })
    run_meta = config_snapshot(
        f"python -m exp_revision.run --task T1 --dataset tt --prepare-fold {fold}{' --smoke' if smoke else ''}",
        {"task": "T1", "dataset": "tt", "fold": fold, "smoke": smoke,
         "preprocessing": "train-fold-only Drain, IDF, scalers, and call graph"},
    )
    run_meta.update(start_time=start_time, end_time=dt.datetime.now(dt.timezone.utc).isoformat())
    json_dump(output / "run_meta.json", run_meta)
    return output
