"""Frozen T5 cohorts and train-only preprocessing using verified raw caches."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .common import ROOT, config_snapshot, dataset_dir, json_dump, sha256
from .fold_data import FoldData
from . import preprocess_aiops as ai
from . import preprocess_tt as tt
from .splits import UNLABELED_FIELDS


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value: raise FileExistsError(f"frozen file differs: {path}")
    else: json_dump(path, value)


def split_file(dataset, task):
    return ROOT / f"splits/{dataset}_{'heldtype' if task == 'T5a' else 'heldservice'}.json"


def training_file(dataset, task, fold):
    return ROOT / f"splits/{dataset}_{task}_training_labels/fold_{fold}.json"


def inner_partition(rows, fraction, *, string_groups=False):
    groups = np.asarray([str(row["overlap_group"]) if string_groups else row["overlap_group"] for row in rows])
    fit, validation = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=2027)
                           .split(np.arange(len(rows)), groups=groups))
    return {name: [rows[int(i)]["incident_id"] for i in values]
            for name, values in (("fit", fit), ("validation", validation))}


def make_cohorts(rows, task, dataset):
    """Label access is confined to this explicit cohort-definition step."""
    faults = sorted({r["fault_type"] for r in rows})
    roots = sorted({r["root_service"] for r in rows})
    if task == "T5b" and len(roots) != 10: raise ValueError("T5b requires the documented ten root services")
    holdouts = faults if task == "T5a" else [[roots[j], roots[j + 5]] for j in range(5)]
    cohorts = {}
    for fold, held in enumerate(holdouts):
        test = [r for r in rows if r["fault_type"] == held] if task == "T5a" else [r for r in rows if r["root_service"] in held]
        test_ids = {r["incident_id"] for r in test}; test_groups = {str(r["overlap_group"]) for r in test}
        train = [r for r in rows if r["incident_id"] not in test_ids and str(r["overlap_group"]) not in test_groups]
        purged = [r["incident_id"] for r in rows if r["incident_id"] not in test_ids and str(r["overlap_group"]) in test_groups]
        if not train or not test: raise ValueError("empty generalization cohort")
        context_inner = inner_partition(train, .1, string_groups=True) if dataset == "tt" else {"fit": [r["incident_id"] for r in train], "validation": []}
        baseline_inner = context_inner if dataset == "tt" else inner_partition(train, .2)
        cohorts[f"fold_{fold}"] = dict(train=[r["incident_id"] for r in train], test=[r["incident_id"] for r in test],
            purged=purged, held_fault_type=held if task == "T5a" else None,
            held_root_services=held if task == "T5b" else [],
            split=f"heldtype_{held}" if task == "T5a" else f"heldsvc_{fold}",
            context_inner=context_inner, baseline_inner=baseline_inner)
    tests = [key for part in cohorts.values() for key in part["test"]]
    if len(tests) != len(set(tests)) or set(tests) != {r["incident_id"] for r in rows}: raise ValueError("held-out folds must partition all events")
    return cohorts


def freeze(dataset):
    rows = json.loads((dataset_dir(dataset) / "incidents.json").read_text())
    unlabeled = json.loads((ROOT / f"splits/{dataset}_unlabeled_events.json").read_text())
    if [{k: row[k] for k in UNLABELED_FIELDS} for row in rows] != unlabeled: raise ValueError("cohort definition differs from frozen event locations")
    lookup = {r["incident_id"]: r for r in rows}
    for task in ("T5a", "T5b"):
        cohorts = make_cohorts(rows, task, dataset); immutable_json(split_file(dataset, task), cohorts)
        for key, part in cohorts.items():
            labels = [{field: lookup[incident][field] for field in ("incident_id", "root_service", "fault_type", "overlap_group")}
                      for incident in part["train"]]
            immutable_json(training_file(dataset, task, int(key.split("_")[1])), labels)
    return {task: str(split_file(dataset, task).relative_to(ROOT)) for task in ("T5a", "T5b")}


def part_for(dataset, task, fold):
    return json.loads(split_file(dataset, task).read_text())[f"fold_{fold}"]


def prepare(dataset, task, fold, output, smoke=False):
    from drain3 import TemplateMiner
    start = dt.datetime.now(dt.timezone.utc).isoformat(); clock = time.perf_counter()
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable generalization preprocessing exists")
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT / f"splits/{dataset}_unlabeled_events.json").read_text())
    if any(set(row) != set(UNLABELED_FIELDS) for row in rows): raise ValueError("preprocessing event manifest exposes labels")
    part = part_for(dataset, task, fold); original = {r["incident_id"]: (i, r) for i, r in enumerate(rows)}
    train_ids = part["train"]; test_ids = part["test"][:5] if smoke else part["test"]
    ids = train_ids + test_ids; metadata = json.loads((dataset_dir(dataset) / "metadata.json").read_text())
    services = metadata["services"]; step = int(metadata["step"]); before = int(metadata["before"])
    bins = (before + int(metadata["after"])) // step
    metric_names = tt.METRICS if dataset == "tt" else metadata["metric_names"]
    raw_root = ROOT / f"cache/raw_telemetry_{dataset}"
    raw_metrics, raw_traces, presences, calls, inputs = [], [], [], {}, {}
    for key in ids:
        index, row = original[key]
        path = raw_root / f"case-{index:03d}.npz" if dataset == "tt" else raw_root / key.replace(":", "_") / "raw.npz"
        expected = dict(incident_id=key, source=row["source"], start=row["start"], end=row["end"], step=step, bins=bins, services=services) if dataset == "tt" else ai._identity(row, services, step, bins, metric_names)
        with np.load(path) as archive:
            if json.loads(str(archive["identity_json"].item())) != expected: raise ValueError("raw event cache identity differs")
            raw_metrics.append(archive["metric"].copy()); raw_traces.append(archive["trace"].copy())
            presences.append(archive["metric_presence"].copy()); calls[key] = json.loads(str(archive["calls_json"].item()))
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        if dataset == "tt":
            paths = sorted((ROOT / Path(row["source"]).parent).glob("*.log.gz"))
        else: paths = [raw_root / key.replace(":", "_") / "logs.jsonl.gz"]
        inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    metric, trace = np.stack(raw_metrics), np.stack(raw_traces)
    def messages(key):
        row = original[key][1]
        return tt._iter_logs(row, services, step, bins) if dataset == "tt" else ai._iter_logs(row, raw_root)
    miner = TemplateMiner(); assignments = {}; total_lines = 0
    for i, key in enumerate(train_ids):
        for _, _, message in messages(key):
            total_lines += 1
            if message not in assignments: assignments[message] = miner.add_log_message(message)["cluster_id"]
        if (i + 1) % 20 == 0: print(f"train-only Drain {i+1}/{len(train_ids)} {time.perf_counter()-clock:.2f}s", flush=True)
    unique_count = len(assignments)
    templates = {cluster.cluster_id: cluster.get_template() for cluster in miner.drain.clusters}
    mode = "insertion_id_disjoint_templates"
    if not tt._templates_disjoint(templates): assignments.clear(); mode = "final_template_rematch"
    cluster_ids = {key: i + 1 for i, key in enumerate(sorted(templates))}
    logs = np.zeros((len(ids), len(services), bins, len(templates) + 1), np.float32)
    log_presence = np.zeros((len(ids), len(services)), bool)
    for i, key in enumerate(ids):
        for service, bin_index, message in messages(key):
            if message not in assignments:
                match = miner.match(message); assignments[message] = match.cluster_id if match else None
            cluster = assignments[message]
            logs[i, service, bin_index, cluster_ids[cluster] if cluster else 0] += 1; log_presence[i, service] = True
    train = np.arange(len(train_ids)); docs = logs[train].sum(2).reshape(-1, logs.shape[-1])
    idf = np.log((len(docs) + 1) / (1 + (docs > 0).sum(0))) + 1
    logs = (np.log1p(logs) * idf).astype(np.float32)
    keep = np.isfinite(metric[train]).any(axis=(0, 1, 2))
    if not keep.any() or not templates: raise ValueError("training cohort lacks metric/log features")
    trace_presence = (trace[:, :, :, 2] > 0).any(2)
    metric, metric_scaler = tt._normalize(metric[:, :, :, keep], train, before // step)
    trace, trace_scaler = tt._normalize(trace, train, before // step)
    availability = np.stack([np.stack(presences), log_presence, trace_presence], axis=-1).astype(np.uint8)
    np.savez_compressed(output / "features.npz", metrics=metric, logs=logs, traces=trace, availability=availability)
    graph = sorted({(c["source"], c["target"]) for key in train_ids for c in calls[key]})
    json_dump(output / "metadata.json", dict(train_ids=train_ids, test_ids=test_ids, services=services,
        metric_names=[name for name, valid in zip(metric_names, keep) if valid], trace_names=["latency_ms", "error_rate", "call_frequency"],
        log_templates=["<UNSEEN>"] + [templates[k] for k in sorted(templates)], log_idf=idf.tolist(), metric_scaler=metric_scaler,
        trace_scaler=trace_scaler, training_graph_edges=graph, calls=calls, before=before, after=metadata["after"], step=step,
        preprocessing_scope="Drain/IDF/channel selection/scalers/graph fitted on T5 training cohort only",
        training_log_lines=total_lines, unique_training_messages_fit_to_drain=unique_count, drain_assignment_mode=mode,
        drain_config={key: value for key, value in vars(miner.config).items() if isinstance(value, (str, int, float, bool, type(None)))},
        limitations=metadata.get("limitations", [])))
    # Queue acceptance witness describes raw availability; it is not an RCA score.
    np.savez_compressed(output / "scores.npz", scores=availability[len(train_ids):, :, 0][None].astype(float), incident_ids=test_ids)
    for path in (split_file(dataset, task), ROOT / f"splits/{dataset}_unlabeled_events.json", dataset_dir(dataset) / "metadata.json"):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    config = dict(task=task, component="train_only_preprocessing", dataset=dataset, fold=fold, smoke=smoke, split=part["split"])
    meta = config_snapshot("python -m exp_revision.generalization_data --prepare", config)
    meta.update(status="completed_generalization_preprocessing", start_time=start, end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-clock, errors=[], input_sha256=inputs,
        source_sha256={name: sha256(ROOT / name) for name in ("exp_revision/generalization_data.py", "exp_revision/preprocess_tt.py", "exp_revision/preprocess_aiops.py")},
        features_sha256=sha256(output / "features.npz"), metadata_sha256=sha256(output / "metadata.json"), scores_sha256=sha256(output / "scores.npz"),
        score_role="raw availability witness; not root-cause inference")
    json_dump(output / "run_meta.json", meta); return output


def prepared_folder(dataset, task, fold, smoke=False):
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    key = f"{'smoke' if smoke else 'full'}_generalization_prepare_{dataset}_{task}_fold{fold}"
    row = state["jobs"][key]
    if row["status"] != "completed": raise ValueError("generalization preprocessing incomplete")
    return ROOT / row["output"]


def load_cohort(dataset, task, fold, smoke=False):
    folder = prepared_folder(dataset, task, fold, smoke); receipt = json.loads((folder / "run_meta.json").read_text())
    for name, field in (("features.npz", "features_sha256"), ("metadata.json", "metadata_sha256")):
        if sha256(folder / name) != receipt[field]: raise ValueError("generalization preprocessing changed")
    metadata = json.loads((folder / "metadata.json").read_text()); part = part_for(dataset, task, fold)
    train_ids = metadata["train_ids"]; test_ids = metadata["test_ids"]
    expected_test = part["test"][:5] if smoke else part["test"]
    if train_ids != part["train"] or test_ids != expected_test: raise ValueError("cohort cache differs from frozen split")
    labels = json.loads(training_file(dataset, task, fold).read_text())
    if [r["incident_id"] for r in labels] != train_ids: raise ValueError("nontraining labels in manifest")
    services = metadata["services"]; index = {s: i for i, s in enumerate(services)}
    with np.load(folder / "features.npz") as archive:
        if set(archive.files) != {"metrics", "logs", "traces", "availability"}: raise ValueError("unexpected feature fields")
        arrays = {key: archive[key].copy() for key in archive.files}
    n = len(train_ids) + len(test_ids)
    for name in ("metrics", "logs", "traces"):
        if arrays[name].ndim != 4 or arrays[name].shape[:2] != (n, len(services)) or not np.isfinite(arrays[name]).all(): raise ValueError("invalid cohort tensor")
    if arrays["availability"].shape != (n, len(services), 3) or not np.isin(arrays["availability"], (0, 1)).all(): raise ValueError("invalid cohort availability")
    adjacency = np.eye(len(services), dtype=bool)
    for u, v in metadata["training_graph_edges"]: adjacency[index[u], index[v]] = adjacency[index[v], index[u]] = True
    return FoldData(dataset, fold, services, train_ids, test_ids, np.asarray([index[r["root_service"]] for r in labels]),
        [r["fault_type"] for r in labels], [r["overlap_group"] for r in labels], arrays["metrics"], arrays["logs"], arrays["traces"],
        arrays["availability"], adjacency, metadata), folder


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    p.add_argument("--freeze", action="store_true"); p.add_argument("--prepare", action="store_true")
    p.add_argument("--task", choices=("T5a", "T5b")); p.add_argument("--fold", type=int); p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path); a = p.parse_args()
    print(freeze(a.dataset) if a.freeze else prepare(a.dataset, a.task, a.fold, a.output, a.smoke))


if __name__ == "__main__": main()
