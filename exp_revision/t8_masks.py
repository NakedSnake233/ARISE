"""Freeze RE2 masks; held-out truth is used solely for declared S3 targets."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .common import ROOT,json_dump
from . import re2_data as d
from .missingness import SCENARIOS,make_mask


def create():
    target=ROOT/"results_v2/T8_inputs/realized_rates.csv"
    if target.exists():return None
    rows=[]
    for dataset in d.DATASETS:
        catalogue=json.loads(d.CATALOG.read_text())[dataset];services=catalogue["services"]
        locations=d.raw_locations(dataset)
        labels={r["incident_id"]:r for r in json.loads((ROOT/f"splits/{dataset}_evaluation_labels.json").read_text())}
        availability={}
        for key,folder in locations.items():
            with np.load(folder/"raw.npz") as a:availability[key]=a["availability"].copy()
        for scenario,param in SCENARIOS:
            folder=ROOT/f"masks/{dataset}/{scenario}_{param or 'none'}";folder.mkdir(parents=True,exist_ok=True)
            for seed in range(42,47):
                path=folder/f"seed{seed}.npz"; masks={};rates={}
                for row in d.event_rows(dataset):
                    key=row["incident_id"];a0=availability[key]
                    root=services.index(labels[key]["root_service"]) if scenario in ("S3","S3c") else None
                    k,rate=make_mask(dict(dataset=dataset,incident_id=key),a0,scenario,param,seed,root)
                    masks[key]=k;rates[key]=rate
                    rows.append(dict(dataset=dataset,incident_id=key,seed=seed,scenario=scenario,scenario_param=param,
                        requested_p=float(param) if scenario=="S2" else "",available_cells=int(a0.sum()),deleted_cells=int(((a0==1)&(k==0)).sum()),realized_rate=rate))
                if path.exists():
                    with np.load(path) as a:
                        if set(a.files)!=set(masks) or any(not np.array_equal(a[k],v) for k,v in masks.items()):raise FileExistsError("existing RE2 masks differ")
                else:np.savez_compressed(path,**masks)
                rp=folder/f"seed{seed}.rates.json"
                if rp.exists() and json.loads(rp.read_text())!=rates:raise FileExistsError("existing rates differ")
                if not rp.exists():json_dump(rp,rates)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return target


if __name__=="__main__":print(create())
