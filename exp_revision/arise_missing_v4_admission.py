"""Admit only full-coverage development pilots within the existing budget."""
import argparse
import fcntl
import json
import subprocess
from .common import ROOT,json_dump,sha256
from . import remaining_queue as q,compute_budget_policy as compute,t8_admission as admission


def create(ds):
    if ds not in ('tt','aiops'):raise ValueError('unsupported pilot')
    path=q.MANIFESTS/f'72_pilot_arise_r3_{ds}.json'
    if path.exists():return path
    if 'Verdict: CLEAR' not in (ROOT/'review/ARISE_R3_IMPLEMENTATION_REVIEW.md').read_text().splitlines():raise ValueError('review pending')
    policy=compute.apply();state=admission.state()
    original=json.loads((q.MANIFESTS/f'70_pilot_r1_{ds}_f0_s42.json').read_text());sources=dict(original['source_sha256'])
    row=state['jobs'][f'r1_{ds}_f0_s42']
    if row['status']!='completed':raise ValueError('context prerequisite missing')
    for n in ['exp_revision/arise_missing_v4.py','exp_revision/arise_missing_v4_train.py','exp_revision/arise_missing_v4_admission.py','exp_revision/arise_missing_v3.py','implement-stage/arise_missing_v4/SPEC.md','implement-stage/arise_missing_v4/ASSUMPTIONS.md']:sources[n]=sha256(ROOT/n)
    for n,h in sources.items():
        if sha256(ROOT/n)!=h:raise ValueError('frozen input changed '+n)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();deadline=1200 if ds=='tt' else 900
    cfg=dict(task='R3',component='arise_missing_v4',method='ARISE-R3',dataset=ds,fold=0,seed=42,smoke=False)
    job=dict(id=f'r3_{ds}_f0_s42',resource='cpu',cores=4,max_attempts=2,timeout_seconds=deadline,source_commit=commit,expected_config=cfg,
        command=['-m','exp_revision.arise_missing_v4_train','--dataset',ds,'--fold','0','--seed','42'],test_ids=row['spec']['test_ids'],score_shape=row['spec']['score_shape'],expected_status=['completed_arise_missing_v4'])
    hours=2*(deadline+10)*4/3600
    forecast=dict(kind='arise_r3',forecast_hours=dict(cpu=hours,gpu=0),details=[dict(dataset=ds,method='ARISE-R3',per_job_hours=hours)],basis='both CPU4 deadline+10s attempts reserved; threefold supervised cross-fitting and matched evidence controls')
    admission.admit_forecast(forecast,f'arise_r3_{ds}_pilot')
    json_dump(path,dict(backend='local',phase='pilot_arise_r3',source_commit=commit,source_sha256=sources,jobs=[job],forecast=forecast,precharged_hours=q.PRECHARGED,compute_budget_policy=policy))
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['tt','aiops'],required=True);a=p.parse_args()
    with (q.FOLDER/'.stages.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);print(create(a.dataset))
