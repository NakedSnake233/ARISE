"""Original BARO/CIRCA/RCD on the same cases, service set and shared masks."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import ROOT,config_snapshot,json_dump,sha256
from .fold_data import load_fold as legacy_fold
from . import re2_data as re2
from .missingness import SCENARIOS

SCENES=[f"{s}_{p or 'none'}" for s,p in SCENARIOS]


def run(dataset,fold,seed,method,output,smoke=False):
    start=dt.datetime.now(dt.timezone.utc).isoformat(); clock=time.perf_counter()
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    data=re2.load_fold(dataset,fold) if dataset in re2.DATASETS else legacy_fold(dataset,fold)
    ids=data.test_ids[:1] if smoke else data.test_ids
    tasks=[]; hashes={}; native_paths={}
    if dataset in re2.DATASETS:
        locations=re2.raw_locations(dataset); row_map={r["incident_id"]:r for r in re2.event_rows(dataset)}
        for key in ids:native_paths[key]=locations[key]/"native_metrics.npz"
    else:
        # Legacy datasets use the same frozen train-scaled binned metric tensor
        # supplied to other baselines; provenance explicitly records this input.
        # CIRCA reconstructs nodes by splitting on exactly one underscore.
        # Stable opaque channel tokens preserve node identity without using
        # labels; actual service/metric names stay in the explicit audit map.
        columns=[f"v{si:03d}_c{ci:03d}" for si in range(len(data.services)) for ci in range(len(data.metadata["metric_names"]))]
        column_services=[s for s in data.services for _ in data.metadata["metric_names"]]
        for key,position in zip(ids,data.test_indices):
            path=output/f"input_{len(native_paths):03d}.npz"
            values=data.metrics[position].transpose(1,0,2).reshape(data.metrics.shape[2],-1)
            np.savez_compressed(path,values=values,times=np.arange(len(values))*data.metadata["step"],columns=columns,column_services=column_services)
            native_paths[key]=path
        json_dump(output/"metric_transport_map.json",{c:dict(service=s,metric=m) for c,s,m in zip(columns,column_services,[m for _ in data.services for m in data.metadata["metric_names"]])})
    for scene in SCENES:
        path=ROOT/f"masks/{dataset}/{scene}/seed{seed}.npz"; hashes[str(path.relative_to(ROOT))]=sha256(path)
        with np.load(path) as a:
            for key in ids:
                tasks.append(dict(incident_id=key,scenario=scene,path=str(native_paths[key]),metric_keep=a[key][:,0].tolist(),
                    inject_time=row_map[key]["start"]+data.metadata["before"] if dataset in re2.DATASETS else data.metadata["before"]))
    spec=dict(services=data.services,seed=seed,tasks=tasks,official_dataset={"re2_ob":"online-boutique","re2_ss":"sock-shop","re2_tt":"train-ticket"}.get(dataset,"generic"))
    json_dump(output/"worker_input.json",spec)
    runtime=ROOT/"cache/rcaeval_runtime"/("rcd" if method=="RCD" else "default")/"bin/python"
    command=[str(runtime),"-m","exp_revision.rcaeval_worker","--method",method,"--input",str(output/"worker_input.json"),"--output",str(output/"official_audit.json")]
    # The parent queue watchdog covers this entire child process group, log,
    # output and inherited TMPDIR; no unguarded writers or per-event processes.
    subprocess.run(command,cwd=ROOT,env={**os.environ,"PYTHONHASHSEED":"42"},check=True)
    records=json.loads((output/"official_audit.json").read_text()); n=len(ids)
    scores=np.asarray([r["scores"] for r in records]).reshape(len(SCENES),n,len(data.services))
    failed=np.asarray([r["failed"] for r in records]).reshape(len(SCENES),n)
    if len(records)!=len(tasks) or not np.isfinite(scores).all():raise ValueError("invalid official output coverage")
    rates=failed.mean(1)
    np.savez_compressed(output/"scores.npz",scores=scores,failed=failed,incident_ids=ids,scenario_keys=SCENES)
    meta=config_snapshot("python -m exp_revision.official_baselines",dict(task="T8",component="official_baseline",method=method,dataset=dataset,fold=fold,seed=seed,smoke=smoke))
    meta.update(status="blocked_algorithm_failure_rate" if (rates>.1).any() else "completed_official_scores",
        errors=[] if not (rates>.1).any() else [dict(error="official algorithm failures exceed 10% in a scenario")],
        start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-clock,
        input_policy="native official metric series" if dataset in re2.DATASETS else "frozen train-fold scaled metric tensor; 15-second or dataset-native bins",
        failure_rates=dict(zip(SCENES,rates.tolist())),code_commit="259ea4167160256a74ad30004aa3d96a998c846e",
        mask_sha256=hashes,input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in native_paths.values()},scores_sha256=sha256(output/"scores.npz"))
    json_dump(output/"run_meta.json",meta)
    if (rates>.1).any(): raise RuntimeError("official algorithm failure rate gate; raw failures retained")


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--fold",type=int,required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--method",choices=("BARO","CIRCA","RCD"),required=True)
    p.add_argument("--smoke",action="store_true");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.dataset,a.fold,a.seed,a.method,a.output,a.smoke)


if __name__=="__main__":main()
