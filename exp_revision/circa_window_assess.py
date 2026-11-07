"""Label-free native cadence, overlap and conservative fair-rerun costs."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .common import ROOT, json_dump, sha256
from .preprocess_aiops import METRIC_BASE
from .drain_kernel_env import guarded

POLICIES = [(180, 240), (900, 240), (600, 600), (900, 900), (1800, 1800)]


def windows(rows, before, after):
    result = {}
    for row in rows:
        onset = row["start"] + 180
        result[row["incident_id"]] = (row["incident_id"].split(":", 1)[0], onset-before, onset+after)
    return result


def overlap_groups(view):
    grouped = collections.defaultdict(list)
    for key, (cloud, start, end) in view.items(): grouped[cloud].append((start, end, key))
    groups = {}; count = 0
    for cloud in sorted(grouped):
        finish = float("-inf"); current = None
        for start, end, key in sorted(grouped[cloud]):
            if start >= finish:
                current = count; count += 1
            finish = max(finish, end); groups[key] = current
    return groups


def assess(output):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    event_path = ROOT / "splits/aiops_unlabeled_events.json"
    split_path = ROOT / "splits/aiops_kfold5.json"
    rows = json.loads(event_path.read_text()); parts = json.loads(split_path.read_text())
    clouds = sorted({r["incident_id"].split(":", 1)[0] for r in rows})
    timestamps = {}; source_hashes = {}; cadence = []
    for cloud in clouds:
        folder = METRIC_BASE / cloud[:10] / ("cloudbed-" + cloud[-1]) / "raw_metric/container-all"
        paths = sorted(folder.glob("*.csv"))
        if not paths: raise FileNotFoundError(folder)
        arrays = []
        for path in paths:
            times = np.sort(pd.read_csv(path, usecols=["timestamp"])["timestamp"].dropna().unique())
            if len(times) < 2: raise ValueError("no native cadence")
            arrays.append(times); source_hashes[str(path)] = sha256(path)
        timestamps[cloud] = arrays
        cadence.append(dict(cloud=cloud, files=len(paths), median_step_seconds=float(np.median(np.concatenate([np.diff(a) for a in arrays]))),
            file_rows_min=min(map(len, arrays)), file_rows_max=max(map(len, arrays))))
    summaries = []
    for before, after in POLICIES:
        view = windows(rows, before, after); groups = overlap_groups(view)
        sizes = collections.Counter(groups.values()); fold_overlap = {}
        counts = []
        for cloud, start, end in view.values():
            counts.extend(int(np.sum((a>=start)&(a<end))) for a in timestamps[cloud])
        for fold, part in parts.items():
            cross = 0
            for key in part["test"]:
                cloud, start, end = view[key]
                cross += any(c==cloud and a<end and start<b for c,a,b in (view[k] for k in part["train"]))
            fold_overlap[fold] = cross
        summaries.append(dict(before_seconds=before, after_seconds=after, duration_seconds=before+after,
            native_rows_min=min(counts), native_rows_median=float(np.median(counts)), native_rows_max=max(counts),
            overlap_groups=len(sizes), largest_group=max(sizes.values()),
            frozen_split_test_events_overlapping_train=fold_overlap,
            five_group_folds_feasible=len(sizes)>=5, raw_window_cell_factor=(before+after)/420))
    proposed_groups = overlap_groups(windows(rows,900,240)); proposed_splits = {}; purge = {}
    for fold,part in parts.items():
        excluded = {proposed_groups[key] for key in part["test"]}
        train = [key for key in part["train"] if proposed_groups[key] not in excluded]
        removed = [key for key in part["train"] if proposed_groups[key] in excluded]
        if {proposed_groups[key] for key in train}&excluded: raise ValueError("purged cohorts overlap")
        proposed_splits[fold] = dict(train=train,test=part["test"])
        purge[fold] = dict(original_train=len(part["train"]),proposed_train=len(train),test=len(part["test"]),
            purged_training_events=len(removed),purged_ids=removed)
    json_dump(output/"proposed_purged_splits.json",proposed_splits)
    json_dump(output/"proposed_purge_audit.json",dict(status="prospective_not_active",groups=proposed_groups,folds=purge))
    current = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    durations = collections.defaultdict(float); job_counts = collections.Counter()
    for name, row in current["jobs"].items():
        cfg = row["spec"]["expected_config"]
        if (cfg.get("dataset") != "aiops" or cfg.get("smoke") or row["status"] != "completed"
                or not name.startswith(("full_baseline_aiops_", "full_evidence_masks_aiops_", "full_official_aiops_"))): continue
        component = cfg.get("method", "evidence_eval" if cfg.get("task")=="T2" else cfg.get("component", "unspecified"))
        duration = sum(a.get("elapsed_seconds", 0) for a in row["attempts"])
        resource = row["spec"]["resource"]; factor = row["spec"]["cores"] if resource=="cpu" else 1
        durations[component+"|"+resource] += duration*factor/3600; job_counts[component+"|"+resource] += 1
    historical_meta = {}
    for folder, component in (("results_v2/T1_context_full/aiops", "context-training"),
            ("results_v2/T1_evidence_full/aiops", "evidence-training")):
        valid = []
        for path in sorted((ROOT / folder).glob("*/attempt*/run_meta.json")):
            m = json.loads(path.read_text())
            if m["config"].get("smoke"): continue
            threads = m["config"].get("threads", m["config"].get("base",{}).get("threads",1))
            valid.append(float(m["elapsed_seconds"])*threads/3600)
            historical_meta[str(path.relative_to(ROOT))] = sha256(path)
        if len(valid)!=25: raise ValueError("full historical25-job timing matrix missing")
        durations[component+"|cpu"] = sum(valid); job_counts[component+"|cpu"] = len(valid)
    legacy_preprocessing = []
    for fold in range(5):
        path = ROOT / f"cache/fold_preprocessing/aiops/fold{fold}/metadata.json"
        if path.exists(): legacy_preprocessing.append(float(json.loads(path.read_text())["elapsed_seconds"])/3600)
    cache_size = sum(p.stat().st_size for p in (ROOT / "cache/raw_telemetry_aiops").rglob("*") if p.is_file())
    folds_size = sum(p.stat().st_size for p in (ROOT / "cache/fold_preprocessing/aiops").rglob("*") if p.is_file())
    budget = dict(limits_cpu_core_hours=200., limits_gpu_hours=48., charged_hours=current.get("charged_hours"),
        conservative_running_bound=current.get("admission_hours"),
        aiops_completed_main_matrix_hours=dict(durations), completed_job_counts=dict(job_counts),
        five_original_fold_preprocessing_cpu_hours=sum(legacy_preprocessing),
        existing_raw_cache_bytes=cache_size, existing_five_fold_bytes=folds_size,
        primary_fair_rerun_cpu_hours_reference=sum(v for k,v in durations.items() if k.endswith("|cpu") and not k.startswith(("BARO|","RCD|")))+sum(legacy_preprocessing),
        historical_timing_meta_sha256=historical_meta,
        estimate_policy="planning scale only: multiply observed matrix work and cache by window-duration ratio and1.25; PC cost is not linear and requires a fresh smoke")
    artifact = dict(status="assessed_no_baselines_run", time=dt.datetime.now(dt.timezone.utc).isoformat(),
        label_policy="only label-free timestamps/event IDs/split membership/source cadence; no roots/faults/accuracy read",
        cadence=cadence, candidate_windows=summaries, cost=budget,
        input_sha256={str(event_path.relative_to(ROOT)):sha256(event_path), str(split_path.relative_to(ROOT)):sha256(split_path)},
        native_source_sha256=source_hashes)
    json_dump(output / "assessment.json", artifact)
    print(json.dumps({k:artifact[k] for k in ("status", "candidate_windows", "cost")}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inside", action="store_true"); args = parser.parse_args()
    if args.inside: assess(args.output)
    else: guarded(args.output.resolve(), [sys.executable, "-m", "exp_revision.circa_window_assess", "--inside", "--output", str(args.output.resolve())], 120)


if __name__ == "__main__": main()
