"""Fixed-window training-only technical feasibility, with no accuracy output."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from .common import ROOT, config_snapshot, dataset_dir, json_dump, sha256
from .preprocess_aiops import METRIC_BASE, _service_name
from .circa_window_assess import windows, overlap_groups

POLICY = ROOT / "exp_revision/circa_long_window_policy.json"


def run(output):
    decision=ROOT/"docs/circa_long_window_execution_decision.json"
    if not decision.exists() or json.loads(decision.read_text()).get("run_long_window_smoke") is not True:
        raise ValueError("author deferred longer-window execution; this entrypoint is prospective only")
    started = dt.datetime.now(dt.timezone.utc).isoformat(); clock = time.perf_counter()
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = json.loads(POLICY.read_text())
    part = json.loads((ROOT / "splits/aiops_kfold5.json").read_text())["fold_0"]
    raw_rows = json.loads((ROOT / "splits/aiops_unlabeled_events.json").read_text())
    groups = overlap_groups(windows(raw_rows,policy["before_seconds"],policy["after_seconds"]))
    held_groups = {groups[key] for key in part["test"]}
    ids = [key for key in part["train"] if groups[key] not in held_groups][:5]
    if len(ids)!=5 or set(ids)&set(part["test"]): raise ValueError("invalid training-only calibration cohort")
    rows = {r["incident_id"]: r for r in raw_rows}
    meta = json.loads((dataset_dir("aiops") / "metadata.json").read_text())
    services = meta["services"]; channels = meta["metric_names"]; frames = {}; hashes = {}
    tasks = []
    for i, key in enumerate(ids):
        row = rows[key]; cloud = key.split(":", 1)[0]; onset = row["start"]+180
        if cloud not in frames:
            base = METRIC_BASE / cloud[:10] / ("cloudbed-"+cloud[-1]) / "raw_metric/container-all"
            groups = {s: [] for s in services}
            for path in sorted(base.glob("*.csv")):
                service = _service_name(path.stem, services)
                if service is not None:
                    groups[service].append(pd.read_csv(path)); hashes[str(path)] = sha256(path)
            if any(not values for values in groups.values()): raise ValueError("native service metric source missing")
            frames[cloud] = {s: pd.concat(values, ignore_index=True).groupby("timestamp", as_index=False).mean(numeric_only=True)
                for s, values in groups.items()}
        selected = {s: frame[(frame.timestamp>=onset-policy["before_seconds"]) &
            (frame.timestamp<onset+policy["after_seconds"])] for s, frame in frames[cloud].items()}
        times = np.sort(np.unique(np.concatenate([v.timestamp.to_numpy() for v in selected.values()])))
        if len(times)!=19: raise ValueError("native proposed window does not contain19observations")
        values = np.concatenate([selected[s].set_index("timestamp").reindex(index=times,columns=channels).to_numpy(float)
            for s in services], axis=1)
        columns = [f"v{si:03d}_c{ci:03d}" for si in range(len(services)) for ci in range(len(channels))]
        column_services = [s for s in services for _ in channels]
        path = output / f"input_{i:03d}.npz"
        np.savez_compressed(path, values=values, times=times, columns=columns, column_services=column_services)
        tasks.append(dict(incident_id=key,scenario="S0_none",path=str(path),metric_keep=[True]*len(services),inject_time=onset))
    spec = dict(services=services,seed=42,tasks=tasks,official_dataset="generic")
    json_dump(output / "worker_input.json",spec)
    scores = []; failed = []; details = {}; durations = {}
    runtime = ROOT / "cache/rcaeval_runtime/default/bin/python"
    for method,module in (("CIRCA","rcaeval_worker"),("CIRCA+rank-prune","rcaeval_rank_worker")):
        path = output / (module+"_audit.json"); begin = time.perf_counter()
        subprocess.run([str(runtime),"-m","exp_revision."+module,"--method","CIRCA","--input",str(output/"worker_input.json"),"--output",str(path)],
            cwd=ROOT,env={**os.environ,"PYTHONHASHSEED":"42"},check=True)
        durations[method] = time.perf_counter()-begin; records = json.loads(path.read_text())
        if [r["incident_id"] for r in records]!=ids or any(r["scenario"]!="S0_none" for r in records):
            raise ValueError("technical probe incident/scenario order differs")
        interpreted = []
        for record in records:
            if (record["status"]=="completed_official" and not record.get("metric_ranks")
                    and not record.get("failed")):
                record={**record,"failed":True,"status":"algorithm_failed_empty_observed_ranking"}
            interpreted.append(record)
        scores.append([r["scores"] for r in interpreted]); failed.append([r["failed"] for r in interpreted]); details[method] = interpreted
        json_dump(output/(module+"_interpreted.json"),interpreted)
    rates = {m:float(np.mean(f)) for m,f in zip(policy["algorithms"],failed)}
    np.savez_compressed(output/"scores.npz",scores=np.asarray(scores),failed=np.asarray(failed),incident_ids=ids,method_names=policy["algorithms"])
    cfg = dict(task="T8",component="window_technical_diagnostic",dataset="aiops",fold=0,seed=42,smoke=True,protocol=policy["protocol"])
    receipt = config_snapshot("python -m exp_revision.circa_window_probe",cfg)
    receipt.update(status="completed_window_diagnostic",errors=[],start_time=started,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-clock,scores_sha256=sha256(output/"scores.npz"),native_source_sha256=hashes,
        policy_sha256=sha256(POLICY),failure_rates=rates,algorithm_seconds=durations,
        viable_methods=[m for m,v in rates.items() if v<=.1],
        scope="training-only technical diagnostic; completion is not algorithm success or held-out evaluation; observed empty ranks count failed; full admission disabled",
        no_accuracy_computed=True,static_metric_schema_only=True,purged_training_ids=True)
    json_dump(output/"run_meta.json",receipt)
    print(json.dumps(dict(diagnostic="completed",failure_rates=rates,algorithm_seconds=durations)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();run(args.output)


if __name__=="__main__":main()
