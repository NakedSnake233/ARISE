"""Combine verified main/missingness evaluations with exact full-cohort coverage."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import ROOT, SEEDS, config_snapshot, json_dump, load_bundle, sha256
from .missingness import SCENARIOS
from .score_receipts import check_hashes, read_job

INPUTS = [("T1_T2_baselines_full", "evaluation_meta.json"),
          ("T2_evidence_masks_full", "evaluation_meta.json"),
          ("T2_context_full/aiops", "aggregate_meta.json")]
METHODS = ["ARISE", "context-only", "ET-spec", "ET-spec+gate+aug", "RootFreqPrior", "Random",
    "Eadro", "MULAN", "Eadro+MD", "MULAN+MD", "Nezha", "MicroRCA", "MicroScope", "MicroRank", "SBLD", "LogFaultFlagger", "TraceAnomaly"]
KEYS = ["dataset", "method", "incident_id", "seed", "scenario", "scenario_param"]


def validate(frame, cohorts, fold_for, rates, truth):
    if frame.duplicated(KEYS).any(): raise ValueError("duplicate main predictions")
    if set(frame.method) != set(METHODS) or set(frame.dataset) != set(cohorts): raise ValueError("main method/dataset coverage differs")
    if set(frame.split) != {"kfold5"} or frame.invalid_by_design.any(): raise ValueError("invalid main split or leaky variant")
    expected = {(scenario, param) for scenario, param in SCENARIOS}
    for dataset, events in cohorts.items():
        subset = frame[frame.dataset == dataset]
        for method in METHODS:
            current = subset[subset.method == method]
            if set(current.seed) != set(SEEDS): raise ValueError("main seed coverage differs")
            for seed in SEEDS:
                by_seed = current[current.seed == seed]
                if set(zip(by_seed.scenario, by_seed.scenario_param)) != expected: raise ValueError("main scenario coverage differs")
                for _, group in by_seed.groupby(["scenario", "scenario_param"], dropna=False):
                    if set(group.incident_id) != set(events) or len(group) != len(events): raise ValueError("main event denominator differs")
    for row in frame.itertuples(index=False):
        key = (row.dataset, row.incident_id)
        if row.fold != fold_for[key]: raise ValueError("prediction belongs to wrong outer fold")
        if (row.root_true, row.fault_type) != truth[key][:2]: raise ValueError("main evaluation truth differs")
        rate = rates[(row.dataset, row.incident_id, row.seed, row.scenario, row.scenario_param)]
        if not np.isclose(row.realized_rate, rate, atol=1e-12): raise ValueError("realized deletion rate differs")
        if row.status not in ("ok", "failed", "fallback"): raise ValueError("unknown prediction status")
        if row.status == "failed" and (not np.isinf(row.rank_of_root) or json.loads(row.top5)): raise ValueError("failed prediction must retain rank inf")
        if row.status == "ok" and (not np.isfinite(row.rank_of_root) or not 1 <= row.rank_of_root <= truth[key][2] or row.rank_of_root != int(row.rank_of_root)):
            raise ValueError("invalid main rank")


def merge(output, publish=False):
    start = time.perf_counter(); from .remaining_queue import now, FOLDER, MANIFESTS
    began = now(); inputs, frames, cache = {}, [], {}
    state = json.loads((FOLDER / "queue_state.json").read_text())
    manifests = {sha256(path): path for path in MANIFESTS.glob("*.json")}
    for relative, receipt in INPUTS:
        folder = ROOT / "results_v2" / relative; meta_path = folder / receipt
        meta = json.loads(meta_path.read_text()); path = folder / "predictions.parquet"
        if sha256(path) != meta["predictions_sha256"]: raise ValueError("main evaluation predictions changed")
        if meta.get("failure_threshold_exceeded"): raise ValueError("input failure rate exceeds 10%; author decision required")
        if "manifest_sha256" in meta:
            manifest_path = manifests[meta["manifest_sha256"]]; manifest = json.loads(manifest_path.read_text())
            check_hashes(manifest["source_sha256"], cache)
            for job in manifest["jobs"]:
                row = state["jobs"][job["id"]]
                if row["status"] == "completed": read_job(job["id"], state, cache)
                elif row["status"] in ("failed", "stuck"):
                    if job["id"] not in meta.get("terminal_failures", {}): raise ValueError("terminal failure missing from evaluation receipt")
                    location = row.get("output") or (row["attempts"][-1]["output"] if row["attempts"] else None)
                    child = ROOT / location if location else None
                    child_meta = child / "run_meta.json" if child else None
                    if child_meta and child_meta.exists():
                        cm = json.loads(child_meta.read_text())
                        if any(cm["config"].get(k) != v for k, v in job["expected_config"].items()): raise ValueError("failed job configuration differs")
                        for group in ("source_sha256", "input_sha256", "mask_sha256"): check_hashes(cm.get(group, {}), cache)
                        if cm.get("scores_sha256") and sha256(child / "scores.npz") != cm["scores_sha256"]: raise ValueError("failed score file changed")
                else: raise ValueError("main evaluation input job unfinished")
            inputs[str(manifest_path.relative_to(ROOT))] = sha256(manifest_path)
        else:
            # The earlier AIOps masked-context queue has separate immutable receipts.
            for fold in range(5):
                for seed in SEEDS:
                    child = folder / f"aiops_fold{fold}_seed{seed}"; cm = json.loads((child / "run_meta.json").read_text())
                    if cm["status"] != "partial_T2_context_only" or cm["scores_sha256"] != sha256(child / "scores.npz"):
                        raise ValueError("masked context receipt changed")
                    for group in ("source_sha256", "input_sha256", "mask_sha256"): check_hashes(cm.get(group, {}), cache)
                    inputs.update({str((child / f).relative_to(ROOT)): sha256(child / f) for f in ("scores.npz", "run_meta.json")})
        inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in (path, meta_path)})
        frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    rate_path = ROOT / "results_v2/T2/realized_rates.csv"
    rate_frame = pd.read_csv(rate_path, keep_default_na=False)
    rate_keys = ["dataset", "incident_id", "seed", "scenario", "scenario_param"]
    if rate_frame.duplicated(rate_keys).any(): raise ValueError("duplicate deletion rate records")
    rates = {tuple(getattr(r, k) for k in rate_keys): r.realized_rate for r in rate_frame.itertuples(index=False)}
    cohorts, fold_for, truth = {}, {}, {}
    # Ground truth is first opened here, strictly during evaluation.
    for dataset in ("tt", "aiops"):
        bundle = load_bundle(dataset); cohorts[dataset] = [r["incident_id"] for r in bundle.rows]
        split_path = ROOT / f"splits/{dataset}_kfold5.json"
        for name, part in json.loads(split_path.read_text()).items():
            for key in part["test"]: fold_for[(dataset, key)] = int(name.split("_")[1])
        for i, row in enumerate(bundle.rows): truth[(dataset, row["incident_id"])] = (bundle.services[bundle.labels[i]], row["fault_type"], len(bundle.services))
        inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in (split_path, bundle.data_dir / "incidents.json", bundle.data_dir / "features.npz")})
    validate(frame, cohorts, fold_for, rates, truth)
    failure_rates = frame.assign(failed=frame.status != "ok").groupby(["dataset", "method", "scenario", "scenario_param"], dropna=False).failed.mean()
    if (failure_rates > .1).any(): raise ValueError("main failure rate exceeds 10%; author decision required")
    frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True); products = {}
    for task in ("T1", "T2"):
        current = frame[frame.scenario == "S0"].copy() if task == "T1" else frame.copy()
        current["task"] = task; current = current.reset_index(drop=True)
        folder = output / task; folder.mkdir(exist_ok=True); path = folder / "predictions.parquet"
        if path.exists():
            if not pd.read_parquet(path).equals(current): raise FileExistsError("immutable merged main output differs")
        else: current.to_parquet(path, index=False)
        products[task] = dict(rows=len(current), failed=int((current.status == "failed").sum()), predictions_sha256=sha256(path))
        receipt = folder / "evaluation_meta.json"
        if not receipt.exists(): json_dump(receipt, dict(time=now(), task=task, coverage="17 methods, all incidents, 5 shared seeds, exact scenarios/folds", **products[task]))
        if publish:
            canonical = ROOT / "results_v2" / task; canonical.mkdir(exist_ok=True)
            for file in ("predictions.parquet", "evaluation_meta.json"):
                target = canonical / file
                if target.exists():
                    if sha256(target) != sha256(folder / file): raise FileExistsError("different canonical main result exists")
                else: os.link(folder / file, target)
    meta = config_snapshot("python -m exp_revision.merge_main", dict(tasks=["T1", "T2"], methods=METHODS))
    meta.update(status="completed", start_time=began, end_time=now(), elapsed_seconds=time.perf_counter() - start,
        products=products, input_sha256={**inputs, str(rate_path.relative_to(ROOT)): sha256(rate_path)},
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)), "exp_revision/score_receipts.py": sha256(ROOT / "exp_revision/score_receipts.py")})
    receipt = output / "run_meta.json"
    if not receipt.exists(): json_dump(receipt, meta)
    return products


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--publish", action="store_true")
    a = p.parse_args(); print(json.dumps(merge(a.output, a.publish)))


if __name__ == "__main__": main()
