"""Two cheap frozen-expert pilots; existing global budget and disk policy retained."""
import fcntl
import json
import subprocess
from .common import ROOT,json_dump,sha256
from . import remaining_queue as q,compute_budget_policy as compute,t8_admission as admission


def create():
    path=q.MANIFESTS/'71_pilot_arise_r2.json'
    if path.exists():return path
    if 'Verdict: CLEAR' not in (ROOT/'review/ARISE_R2_IMPLEMENTATION_REVIEW.md').read_text().splitlines():raise ValueError('review pending')
    policy=compute.apply();state=admission.state();sources={};jobs=[]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for ds in ['tt','aiops']:
        old=json.loads((q.MANIFESTS/f'70_pilot_r1_{ds}_f0_s42.json').read_text());sources.update(old['source_sha256'])
        row=state['jobs'][f'r1_{ds}_f0_s42'];folder=ROOT/row['output']
        if row['status']!='completed' or not q.verified(folder,row['spec']):raise ValueError('R1 prerequisite')
        for name in ['calibration.npz','inner_split.json','scores.npz','run_meta.json']:sources[str((folder/name).relative_to(ROOT))]=sha256(folder/name)
        for n in ['exp_revision/arise_missing_v3.py','exp_revision/arise_missing_v3_run.py','exp_revision/arise_missing_v3_admission.py','implement-stage/arise_missing_v3/SPEC.md','implement-stage/arise_missing_v3/ASSUMPTIONS.md']:sources[n]=sha256(ROOT/n)
        cfg=dict(task='R2',component='arise_missing_v3',method='ARISE-R2',dataset=ds,fold=0,seed=42,smoke=False)
        jobs.append(dict(id=f'r2_{ds}_f0_s42',resource='cpu',cores=1,max_attempts=2,timeout_seconds=180,source_commit=commit,
            expected_config=cfg,command=['-m','exp_revision.arise_missing_v3_run','--dataset',ds,'--fold','0','--seed','42'],
            test_ids=row['spec']['test_ids'],score_shape=row['spec']['score_shape'],expected_status=['completed_arise_missing_v3']))
    for name,digest in sources.items():
        if sha256(ROOT/name)!=digest:raise ValueError('source changed '+name)
    forecast=dict(kind='arise_r2',forecast_hours=dict(cpu=4*190/3600,gpu=0),details=[dict(dataset=d,method='ARISE-R2',per_job_hours=2*190/3600) for d in ['tt','aiops']],basis='two CPU1 jobs; two190s attempt bounds each; no model refit')
    admission.admit_forecast(forecast,'arise_r2_pilots')
    json_dump(path,dict(backend='local',phase='pilot_arise_r2',source_commit=commit,source_sha256=sources,jobs=jobs,forecast=forecast,precharged_hours=q.PRECHARGED,compute_budget_policy=policy))
    return path

if __name__=='__main__':
    with (q.FOLDER/'.stages.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);print(create())
