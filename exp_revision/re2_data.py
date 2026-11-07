"""Official RE2 import, neutral event IDs and train-only fold preprocessing."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
from functools import lru_cache
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit

from .common import ROOT, config_snapshot, json_dump, sha256
from .fold_data import FoldData
from .generalization_data import immutable_json
from .preprocess_tt import _normalize, _templates_disjoint
from .raw_spans_v2 import rpc_features
from .splits import make_kfold, check_splits, write_training_label_manifests

DATASETS = ("re2_ob", "re2_ss", "re2_tt")
CATALOG = ROOT / "cache/rcaeval_re2/catalog_metric_services.json"
BEFORE = AFTER = 720
STEP = 15
BINS = 96
LOG_CHANNEL_LIMIT = 128
ANCILLARY_NAMES = {"istio-init", "InboundPassthroughClusterIpv4", "PassthroughCluster", "frontend-check", "frontend-external"}


def canonical(service):
    return {"frontendservice": "frontend"}.get(str(service), str(service))


def download_rows():
    index = pd.read_parquet(ROOT / "cache/rcaeval_re2/cases.parquet")
    index = index[index.suite == "RE2"].sort_values(["dataset", "case"])
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    result = []
    for dataset in DATASETS:
        subset = index[index.dataset == "RE2-" + dataset[-2:].upper()]
        if len(subset) != 90: raise ValueError("official RE2 must have 90 cases per system")
        for i, record in enumerate(subset.to_dict("records")):
            job = state["jobs"].get("re2_download_" + record["case"])
            if not job or job["status"] != "completed": raise ValueError("full official RE2 downloads are not complete")
            result.append(dict(dataset=dataset, incident_id=f"{dataset}:event_{i:03d}",
                record=record, raw_path=str(Path(job["output"]) / "raw")))
    return result


def inventory():
    """Labels are accessed only here to freeze splits/evaluation/training labels."""
    source_rows = download_rows()
    catalog = {}
    for dataset in DATASETS:
        selected = [r for r in source_rows if r["dataset"] == dataset]
        services, channels = set(), set()
        for item in selected:
            folder = ROOT / item["raw_path"]
            for column in pq.read_schema(folder / "metrics.parquet").names:
                if "_" in column:
                    s, channel = column.split("_", 1); services.add(canonical(s)); channels.add(channel)
        services = sorted(services - {"", "nan", "None"} - ANCILLARY_NAMES)
        # Union connected components of overlapping observed source windows.
        sorted_items = sorted(selected, key=lambda r: r["record"]["time_start"])
        groups = {}; group = -1; end = -float("inf")
        for item in sorted_items:
            record = item["record"]
            if record["time_start"] > end: group += 1
            end = max(end, record["time_end"])
            groups[item["incident_id"]] = f"{dataset}:overlap_{group:03d}"
        rows, locations = [], {}
        for item in selected:
            record = item["record"]; incident = item["incident_id"]
            root = canonical(record["root_cause_service"])
            if root not in services: raise ValueError("annotated root absent from observed service catalogue")
            rows.append(dict(incident_id=incident, start=float(record["inject_time"])-BEFORE,
                end=float(record["inject_time"])+AFTER, source=f"cache/re2_raw/{dataset}/{incident.split(':')[1]}",
                overlap_group=groups[incident], root_service=root, fault_type=record["fault"]))
            locations[incident] = item["raw_path"]
        unlabeled = [{k: r[k] for k in ("incident_id", "start", "end", "source", "overlap_group")} for r in rows]
        immutable_json(ROOT / f"splits/{dataset}_unlabeled_events.json", unlabeled)
        immutable_json(ROOT / f"splits/{dataset}_evaluation_labels.json", [{k: r[k] for k in ("incident_id", "root_service", "fault_type")} for r in rows])
        immutable_json(ROOT / f"cache/rcaeval_re2/{dataset}_locations.json", locations)
        bundle = SimpleNamespace(rows=rows, groups=[r["overlap_group"] for r in rows])
        split_path = ROOT / f"splits/{dataset}_kfold5.json"
        if split_path.exists(): split = json.loads(split_path.read_text())
        else: split = make_kfold(bundle, split_path)
        check_splits(bundle, split)
        labels_folder = ROOT / f"splits/{dataset}_training_labels"
        if not labels_folder.exists(): write_training_label_manifests(bundle, split, labels_folder)
        inner = {}
        lookup = {r["incident_id"]: r for r in rows}
        for name, part in split.items():
            fit, validation = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=2027).split(part["train"], groups=[lookup[k]["overlap_group"] for k in part["train"]]))
            inner[name] = {n: [part["train"][int(i)] for i in values] for n, values in (("fit", fit), ("validation", validation))}
        immutable_json(ROOT / f"splits/{dataset}_inner_validation.json", inner)
        catalog[dataset] = dict(services=services, metric_names=sorted(channels), before=BEFORE,
            after=AFTER, step=STEP, bins=BINS, events=len(rows), overlap_groups=group+1,
            fault_types=dict(collections.Counter(r["fault_type"] for r in rows)),
            log_channel_limit=LOG_CHANNEL_LIMIT, aliases={"frontendservice": "frontend"},
            candidate_catalogue="all monitored service prefixes except explicitly named init/proxy/probe telemetry; excluded spans remain ancillary dependencies",
            ancillary_names=sorted(ANCILLARY_NAMES),
            preprocessing_profile="frozen TrainTicket model settings, group-isolated 10% validation",
            limitations=["RE2-SS has no trace modality", "exported traces lack native span.kind; RPC uses unambiguous same-operation parent pairs"])
    immutable_json(CATALOG, catalog)
    return catalog


def event_rows(dataset):
    return json.loads((ROOT / f"splits/{dataset}_unlabeled_events.json").read_text())


def raw_locations(dataset):
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    result = {}
    for key, job in state["jobs"].items():
        if key.startswith(f"re2_raw_{dataset}_") and job["status"] == "completed":
            result[job["spec"]["expected_config"]["incident_id"]] = ROOT / job["output"]
    if set(result) != {r["incident_id"] for r in event_rows(dataset)}:
        raise ValueError("full neutral raw telemetry cache is not complete")
    return result


def span_table(frame, services):
    """Preserve string trace/span identity and complete parent closure."""
    frame = frame.drop_duplicates(["traceID", "spanID"], keep="last").reset_index(drop=True)
    ids = list(zip(frame.traceID.astype(str), frame.spanID.astype(str)))
    lookup = {key: i for i, key in enumerate(ids)}
    parent, child = [], []
    for i, key in enumerate(zip(frame.traceID.astype(str), frame.parentSpanID.astype(str))):
        p = lookup.get(key)
        if p is not None: parent.append(p); child.append(i)
    index = {s: i for i, s in enumerate(services)}
    # grpc. differs between caller and callee in the official OB export.
    ops = frame.operationName.fillna("").astype(str).str.replace(r"^grpc\.", "", regex=True)
    codes, _ = pd.factorize(ops, sort=True)
    codes[ops.eq("").to_numpy()] = -1
    return dict(start_us=frame.startTime.to_numpy(dtype=float), duration_us=frame.duration.to_numpy(dtype=float),
        service=np.asarray([index.get(canonical(s), -1) for s in frame.serviceName], np.int16),
        kind=np.zeros(len(frame), np.int8), operation=codes.astype(np.int32),
        parent=np.asarray(parent, np.int32), child=np.asarray(child, np.int32), native=np.asarray(False))


def prepare_raw(dataset, incident, output):
    start = dt.datetime.now(dt.timezone.utc).isoformat(); clock = time.perf_counter()
    catalog = json.loads(CATALOG.read_text())[dataset]; services=catalog["services"]; channels=catalog["metric_names"]
    row = next(r for r in event_rows(dataset) if r["incident_id"] == incident)
    source = ROOT / json.loads((ROOT / f"cache/rcaeval_re2/{dataset}_locations.json").read_text())[incident]
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    s_index = {s: i for i, s in enumerate(services)}; c_index = {c: i for i, c in enumerate(channels)}
    metric = np.full((len(services), BINS, len(channels)), np.nan, np.float32)
    presence = np.zeros((len(services),3), np.uint8)
    frame = pd.read_parquet(source / "metrics.parquet")
    native_columns=[c for c in frame.columns if "_" in c]
    native_mask=(frame.time >= row["start"]) & (frame.time < row["end"])
    np.savez_compressed(output / "native_metrics.npz",values=frame.loc[native_mask,native_columns].to_numpy(dtype=float),
        times=frame.loc[native_mask,"time"].to_numpy(dtype=float),columns=native_columns,
        column_services=[canonical(c.split("_",1)[0]) for c in native_columns])
    bins = ((frame.time.to_numpy(dtype=float)-row["start"])//STEP).astype(int)
    valid = (bins >= 0) & (bins < BINS)
    for column in frame.columns:
        if "_" not in column: continue
        s, c = column.split("_",1); s = canonical(s)
        if s not in s_index or c not in c_index: continue
        series = pd.DataFrame(dict(bin=bins[valid], value=pd.to_numeric(frame[column], errors="coerce").to_numpy()[valid])).groupby("bin").value.mean()
        metric[s_index[s], series.index, c_index[c]] = series.to_numpy()
        presence[s_index[s],0] |= int(series.notna().any())
    trace = np.zeros((len(services), BINS, 3), np.float32); calls = []
    table = dict(start_us=np.asarray([],float), duration_us=np.asarray([],float), service=np.asarray([],np.int16),
        kind=np.asarray([],np.int8), operation=np.asarray([],np.int32), parent=np.asarray([],np.int32), child=np.asarray([],np.int32),native=np.asarray(False))
    if (source / "traces.parquet").exists():
        spans = pd.read_parquet(source / "traces.parquet"); table = span_table(spans,services)
        bin_numbers = ((table["start_us"]/1e6-row["start"])//STEP).astype(int)
        usable = (table["service"] >= 0) & (bin_numbers >= 0) & (bin_numbers < BINS) & np.isfinite(table["duration_us"]) & (table["duration_us"] >= 0)
        rates = pd.DataFrame(dict(service=table["service"][usable], bin=bin_numbers[usable], latency=table["duration_us"][usable]/1000))
        # drop_duplicates in span_table preserves order for statusCode alignment.
        status = spans.drop_duplicates(["traceID", "spanID"], keep="last").statusCode.fillna(0)
        rates["error"] = status.astype(str).str.lower().isin(["error","2","500","503"]).to_numpy()[usable]
        grouped = rates.groupby(["service", "bin"]).agg(latency=("latency","mean"), error=("error","mean"), n=("latency","count"))
        for (s,b), r in grouped.iterrows(): trace[s,b] = [r.latency,r.error,r.n/STEP]; presence[s,2]=1
        p,c = table["parent"], table["child"]
        valid_pairs = (table["service"][p]>=0)&(table["service"][c]>=0)&(table["service"][p]!=table["service"][c])&(bin_numbers[c]>=0)&(bin_numbers[c]<BINS)
        counts = collections.Counter(zip(table["service"][p[valid_pairs]].tolist(),table["service"][c[valid_pairs]].tolist()))
        calls = [dict(source=services[u],target=services[v],count=n) for (u,v),n in sorted(counts.items())]
    logs = collections.Counter()
    if (source / "logs.parquet").exists():
        for r in pd.read_parquet(source / "logs.parquet").itertuples(index=False):
            s = s_index.get(canonical(r.container_name)); b = int((float(r.timestamp)-row["start"])//STEP)
            if s is not None and 0 <= b < BINS and pd.notna(r.message):
                logs[s,b,str(r.message)] += 1; presence[s,1]=1
    with gzip.open(output / "logs.jsonl.gz", "wt") as stream:
        for (s,b,message),n in sorted(logs.items()): stream.write(json.dumps([s,b,message,n]) + "\n")
    np.savez_compressed(output / "raw.npz", metrics=metric, traces=trace, availability=presence)
    np.savez_compressed(output / "spans.npz", **table)
    json_dump(output / "raw_meta.json", dict(incident_id=incident,services=services,metric_names=channels,calls=calls))
    np.savez_compressed(output / "scores.npz", scores=presence[None].astype(float),incident_ids=[incident])
    meta=config_snapshot("python -m exp_revision.re2_data --mode raw",dict(task="T8",component="neutral_raw",dataset=dataset,incident_id=incident))
    meta.update(status="completed_re2_raw",start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-clock,
        input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in source.iterdir() if p.is_file()},scores_sha256=sha256(output / "scores.npz"),errors=[])
    json_dump(output / "run_meta.json",meta)


def log_records(folder):
    with gzip.open(folder / "logs.jsonl.gz", "rt") as stream:
        for line in stream: yield json.loads(line)


def prepare_fold(dataset, fold, output, smoke=False):
    from drain3 import TemplateMiner
    start=dt.datetime.now(dt.timezone.utc).isoformat(); clock=time.perf_counter()
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads(CATALOG.read_text())[dataset]; part=json.loads((ROOT / f"splits/{dataset}_kfold5.json").read_text())[f"fold_{fold}"]
    train_ids=part["train"]; test_ids=part["test"][:3] if smoke else part["test"]; ids=train_ids+test_ids
    locations=raw_locations(dataset); miner=TemplateMiner(); assignments={}; count=collections.Counter(); lines=0
    for key in train_ids:
        for s,b,message,n in log_records(locations[key]):
            if message not in assignments: assignments[message]=miner.add_log_message(message)["cluster_id"]
            count[assignments[message]]+=n; lines+=n
    templates={c.cluster_id:c.get_template() for c in miner.drain.clusters}
    disjoint=_templates_disjoint(templates)
    if not disjoint:
        assignments.clear(); count=collections.Counter()
        for key in train_ids:
            for s,b,message,n in log_records(locations[key]):
                if message not in assignments:
                    match=miner.match(message); assignments[message]=match.cluster_id if match else None
                count[assignments[message]]+=n
    selected=sorted((c for c in templates),key=lambda c:(-count[c],c))[:LOG_CHANNEL_LIMIT]
    selected=sorted(selected); cluster_index={c:i+1 for i,c in enumerate(selected)}
    services=catalog["services"]; logs=np.zeros((len(ids),len(services),BINS,len(selected)+1),np.float32)
    metrics=[]; traces=[]; availability=[]; calls={}; inputs={}
    for i,key in enumerate(ids):
        folder=locations[key]
        with np.load(folder / "raw.npz") as a:
            metrics.append(a["metrics"]); traces.append(a["traces"]); availability.append(a["availability"])
        calls[key]=json.loads((folder / "raw_meta.json").read_text())["calls"]
        for s,b,message,n in log_records(folder):
            if message not in assignments:
                match=miner.match(message); assignments[message]=match.cluster_id if match else None
            logs[i,s,b,cluster_index.get(assignments[message],0)]+=n
        for name in ("raw.npz","raw_meta.json","logs.jsonl.gz"): inputs[str((folder/name).relative_to(ROOT))]=sha256(folder/name)
    train=np.arange(len(train_ids)); metrics=np.stack(metrics); traces=np.stack(traces); availability=np.stack(availability)
    keep=np.isfinite(metrics[train]).any(axis=(0,1,2)); metrics, metric_scaler=_normalize(metrics[:,:,:,keep],train,BEFORE//STEP)
    traces, trace_scaler=_normalize(traces,train,BEFORE//STEP)
    docs=logs[train].sum(2).reshape(-1,logs.shape[-1]); idf=np.log((len(docs)+1)/(1+(docs>0).sum(0)))+1
    logs=(np.log1p(logs)*idf).astype(np.float32)
    np.savez_compressed(output / "features.npz", metrics=metrics,logs=logs,traces=traces,availability=availability)
    meta=dict(dataset=dataset,fold=fold,services=services,train_ids=train_ids,test_ids=test_ids,
        metric_names=[n for n,k in zip(catalog["metric_names"],keep) if k],trace_names=["latency_ms","error_rate","call_frequency"],
        log_templates=["<UNSEEN_OR_RARE>"]+[templates[c] for c in selected],log_idf=idf.tolist(),log_channel_limit=LOG_CHANNEL_LIMIT,
        total_training_templates=len(templates),training_log_lines=lines,drain_assignment_mode="insertion_id" if disjoint else "final_template_rematch",
        metric_scaler=metric_scaler,trace_scaler=trace_scaler,before=BEFORE,after=AFTER,step=STEP,calls=calls,
        training_graph_edges=sorted({(c["source"],c["target"]) for k in train_ids for c in calls[k]}),
        preprocessing_scope="training incidents only: Drain, top-frequency template vocabulary, IDF, channel selection, scalers, graph",
        input_sha256=inputs)
    json_dump(output / "metadata.json",meta)
    np.savez_compressed(output / "scores.npz", scores=availability[len(train_ids):,:,0][None].astype(float),incident_ids=test_ids)
    receipt=config_snapshot("python -m exp_revision.re2_data --mode fold",dict(task="T8",component="fold_preprocessing",dataset=dataset,fold=fold,smoke=smoke))
    receipt.update(status="completed_re2_fold",start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-clock,
        scores_sha256=sha256(output / "scores.npz"),features_sha256=sha256(output / "features.npz"),metadata_sha256=sha256(output / "metadata.json"),errors=[])
    json_dump(output / "run_meta.json",receipt)


def fold_folder(dataset,fold):
    state=json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    job=state["jobs"].get(f"re2_fold_{dataset}_f{fold}")
    if not job or job["status"]!="completed": raise ValueError("RE2 fold preprocessing incomplete")
    return ROOT / job["output"]


def load_fold(dataset,fold):
    folder=fold_folder(dataset,fold); meta=json.loads((folder/"metadata.json").read_text())
    labels=json.loads((ROOT / f"splits/{dataset}_training_labels/fold_{fold}.json").read_text())
    if [r["incident_id"] for r in labels]!=meta["train_ids"]: raise ValueError("training label order differs")
    mapping={s:i for i,s in enumerate(meta["services"])}
    with np.load(folder / "features.npz") as a:
        arrays={k:a[k].copy() for k in ("metrics","logs","traces","availability")}
    adjacency=np.eye(len(mapping),dtype=bool)
    for u,v in meta["training_graph_edges"]: adjacency[mapping[u],mapping[v]]=adjacency[mapping[v],mapping[u]]=True
    return FoldData(dataset,fold,meta["services"],meta["train_ids"],meta["test_ids"],
        np.asarray([mapping[r["root_service"]] for r in labels]),[r["fault_type"] for r in labels],
        [r["overlap_group"] for r in labels],arrays["metrics"],arrays["logs"],arrays["traces"],arrays["availability"],adjacency,meta)


def inner_indices(data):
    part=json.loads((ROOT / f"splits/{data.dataset}_inner_validation.json").read_text())[f"fold_{data.fold}"]
    mapping={k:i for i,k in enumerate(data.train_ids)}
    return tuple(np.asarray([mapping[k] for k in part[name]]) for name in ("fit","validation"))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("inventory","raw","fold"),required=True)
    p.add_argument("--dataset",choices=DATASETS); p.add_argument("--incident"); p.add_argument("--fold",type=int)
    p.add_argument("--smoke",action="store_true"); p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.mode=="inventory": print(json.dumps(inventory()))
    elif a.mode=="raw": prepare_raw(a.dataset,a.incident,a.output)
    else: prepare_fold(a.dataset,a.fold,a.output,a.smoke)


if __name__=="__main__": main()
