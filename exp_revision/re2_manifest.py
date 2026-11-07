"""Immutable admission of neutral RE2 import and preprocessing jobs."""
from __future__ import annotations
import argparse
import json
import subprocess
from .common import ROOT,json_dump,sha256
from . import remaining_queue as q
from . import re2_data as d


def closure():
    names=["exp_revision/re2_manifest.py","exp_revision/re2_data.py","exp_revision/common.py",
        "exp_revision/fold_data.py","exp_revision/generalization_data.py","exp_revision/preprocess_tt.py","exp_revision/preprocess_aiops.py",
        "exp_revision/splits.py","exp_revision/raw_spans_v2.py","scripts/care/trace_residual.py",
        "scripts/care/prepare_aiops.py","scripts/care/prepare_aiops_rpc.py","exp_revision/remaining_queue.py",
        "exp_revision/disk_guard.py","exp_revision/disk_guard_local.py",str(d.CATALOG.relative_to(ROOT))]
    for ds in d.DATASETS:
        names += [f"splits/{ds}_unlabeled_events.json",f"splits/{ds}_kfold5.json",f"cache/rcaeval_re2/{ds}_locations.json"]
    return {n:sha256(ROOT/n) for n in names}


def create(stage="raw",smoke=True):
    path=q.MANIFESTS/f"{42 if stage=='raw' else 44}_{'smoke' if smoke else 'full'}_re2_{stage}.json"
    if stage=="raw" and not smoke: path=q.MANIFESTS/"43_full_re2_raw.json"
    if stage=="fold" and not smoke: path=q.MANIFESTS/"45_full_re2_fold.json"
    if path.exists():return None
    state=json.loads((q.FOLDER/"queue_state.json").read_text())
    if not smoke:
        pilot=q.MANIFESTS/("42_smoke_re2_raw.json" if stage=="raw" else "44_smoke_re2_fold.json")
        if not pilot.exists(): raise ValueError("reviewed preprocessing smoke is required before full admission")
        for spec in json.loads(pilot.read_text())["jobs"]:
            row=state["jobs"].get(spec["id"])
            if not row or row["status"]!="completed" or not q.verified(ROOT/row["output"],spec):
                raise ValueError("preprocessing smoke witness incomplete")
    if stage=="fold":
        for ds in d.DATASETS:
            d.raw_locations(ds)
            for row in d.event_rows(ds):
                key=f"re2_raw_{ds}_{row['incident_id'].split(':')[1]}"; job=state["jobs"].get(key)
                if not job or job["status"]!="completed" or not q.verified(ROOT/job["output"],job["spec"]):
                    raise ValueError("full neutral raw cache must precede fold preprocessing")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    jobs=[]; catalog=json.loads(d.CATALOG.read_text())
    for ds in d.DATASETS:
        rows=d.event_rows(ds)
        if stage=="raw":
            for row in rows[:1] if smoke else rows:
                incident=row["incident_id"]; short=incident.split(":")[1]
                cfg=dict(task="T8",component="neutral_raw",dataset=ds,incident_id=incident)
                # Full import reuses reviewed raw smoke jobs byte-for-byte.
                job=dict(id=f"re2_raw_{ds}_{short}",resource="cpu",cores=1,max_attempts=2,timeout_seconds=300,
                    source_commit=commit,expected_config=cfg,command=["-m","exp_revision.re2_data","--mode","raw","--dataset",ds,"--incident",incident],
                    test_ids=[incident],score_shape=[1,len(catalog[ds]["services"]),3],expected_status=["completed_re2_raw"])
                if not smoke:
                    old=q.MANIFESTS/"42_smoke_re2_raw.json"
                    existing={j["id"]:j for j in json.loads(old.read_text())["jobs"]} if old.exists() else {}
                    job=existing.get(job["id"],job)
                jobs.append(job)
        else:
            splits=json.loads((ROOT/f"splits/{ds}_kfold5.json").read_text())
            for fold in range(1 if smoke else 5):
                # Fold smoke executes full preprocessing. Only downstream model
                # smoke limits test inference; no separate partial fit cache.
                ids=splits[f"fold_{fold}"]["test"]
                job=dict(id=f"re2_fold_{ds}_f{fold}",resource="cpu",cores=1,max_attempts=2,timeout_seconds=900,
                    source_commit=commit,expected_config=dict(task="T8",component="fold_preprocessing",dataset=ds,fold=fold,smoke=False),
                    command=["-m","exp_revision.re2_data","--mode","fold","--dataset",ds,"--fold",str(fold)],
                    test_ids=ids,score_shape=[1,len(ids),len(catalog[ds]["services"])],expected_status=["completed_re2_fold"])
                if not smoke:
                    old=q.MANIFESTS/"44_smoke_re2_fold.json"
                    existing={j["id"]:j for j in json.loads(old.read_text())["jobs"]} if old.exists() else {}
                    job=existing.get(job["id"],job)
                jobs.append(job)
    json_dump(path,dict(backend="local",phase=f"{'pilot' if smoke else 'full'}_re2_{stage}",source_commit=commit,
        precharged_hours=q.PRECHARGED,source_sha256=closure(),jobs=jobs))
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("raw","fold"),required=True);p.add_argument("--full",action="store_true")
    a=p.parse_args(); print(create(a.stage,not a.full))


if __name__=="__main__":main()
