"""Register reviewed R1 pilots inside the existing budget/disk-controlled queue."""
from __future__ import annotations
import argparse
import fcntl
import json
import subprocess
from .common import ROOT,json_dump,sha256
from . import remaining_queue as q,compute_budget_policy as compute,t8_admission as admission
from .arise_missing_v2_train import SCENES


def create(dataset,fold,seed,timeout=900,phase='pilot'):
    if dataset not in ('tt','aiops') or fold not in range(5) or seed not in range(42,47):raise ValueError('unsupported initial cohort')
    if timeout not in (600,900):raise ValueError('initial runtime bound must be prespecified')
    name=f'r1_{dataset}_f{fold}_s{seed}'
    path=q.MANIFESTS/f'70_{phase}_{name}.json'
    if path.exists():return path
    review=ROOT/'review/ARISE_R1_IMPLEMENTATION_REVIEW.md'
    if not review.exists() or 'Verdict: CLEAR' not in review.read_text().splitlines():raise ValueError('R1 predeployment review pending')
    policy=compute.apply()
    part=json.loads((ROOT/f'splits/{dataset}_kfold5.json').read_text())[f'fold_{fold}']
    metadata=json.loads((ROOT/f'cache/fold_preprocessing/{dataset}/fold{fold}/metadata.json').read_text())
    from .infer_evidence_masks import context_scores
    _,folder=context_scores(dataset,fold,seed,part['test'])
    # Reuse the full existing source/raw-data closure; add only the new method's inputs.
    original=json.loads((q.MANIFESTS/'08_full_t3_evidence.json').read_text())
    sources=dict(original['source_sha256'])
    names=['exp_revision/arise_missing_v2.py','exp_revision/arise_missing_v2_train.py','exp_revision/arise_missing_v2_admission.py',
        'exp_revision/train_context_variant.py','exp_revision/train_context.py','exp_revision/compute_budget_policy.py',
        'implement-stage/arise_missing_v2/SPEC.md','implement-stage/arise_missing_v2/ASSUMPTIONS.md']
    for p in [ROOT/n for n in names]+[folder/'scores.npz',folder/'run_meta.json',compute.APPROVAL]:sources[str(p.relative_to(ROOT))]=sha256(p)
    for relative,digest in sources.items():
        if sha256(ROOT/relative)!=digest:raise ValueError('input closure mismatch: '+relative)
    hours=(timeout+10)*4/3600
    forecast=dict(kind='arise_r1',forecast_hours=dict(cpu=hours,gpu=0),
        details=[dict(dataset=dataset,component='arise_missing_v2',method='ARISE-R1',per_job_hours=hours)],
        basis='initial full timeout at four declared CPU cores; actual attempt time charged by shared queue')
    admission.admit_forecast(forecast,name)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    job=dict(id=name,resource='cpu',cores=4,max_attempts=2,timeout_seconds=timeout,source_commit=commit,
        expected_config=dict(task='R1',component='arise_missing_v2',method='ARISE-R1',dataset=dataset,fold=fold,seed=seed,smoke=False),
        command=['-m','exp_revision.arise_missing_v2_train','--dataset',dataset,'--fold',str(fold),'--seed',str(seed)],
        test_ids=part['test'],score_shape=[len(SCENES),len(part['test']),len(metadata['services'])],expected_status=['completed_arise_missing_v2'])
    json_dump(path,dict(backend='local',phase=phase+'_arise_r1',source_commit=commit,precharged_hours=q.PRECHARGED,
        source_sha256=sources,jobs=[job],forecast=forecast,compute_budget_policy=policy,
        author_scope='explicit latest request to improve ARISE; existing total numeric caps retained',
        evaluation_scope='development on inspected benchmark, not independent confirmation'))
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['tt','aiops'],required=True);p.add_argument('--fold',type=int,default=0)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
    with (q.FOLDER/'.stages.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(create(a.dataset,a.fold,a.seed,a.timeout))


if __name__=='__main__':main()
