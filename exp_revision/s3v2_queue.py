"""Local inference queue using the existing process/lease/watchdog machinery."""
import argparse
from collections import Counter
import fcntl
import json
import subprocess
import time
from .common import ROOT,json_dump,sha256
from . import remaining_queue as q
from . import compute_budget_policy as policy
from .extended_queue import charged as legacy_charged
from .t8_admission import pending_forecasts
from .disk_guard_fast import DiskGuard,guarded_command
from .scenario_v2 import DATASETS

FOLDER=ROOT/'results_v2/S3v2_models'


def cost(state,running=False,gpu=False):
    return sum((a.get('elapsed_seconds',(row['spec']['timeout_seconds']+10) if running else 0))*row['spec']['cores']/3600
        for row in state['jobs'].values() for a in row['attempts'] if not gpu or a.get('resource',row['spec']['resource'])=='gpu')


def remaining_budget():
    limits=policy.apply()['limits']
    authorization=ROOT/'docs/s3v2_compute_authorization.json'
    if authorization.exists():
        approval=json.loads(authorization.read_text())
        if approval.get('approved') is True:
            if approval.get('limits')!={'cpu':300.,'gpu':56.} or approval.get('source')!='explicit author reply in this conversation':raise ValueError('invalid S3v2 budget authorization')
            limits=approval['limits']
    state=json.loads((ROOT/'results_v2/remaining_queue/queue_state.json').read_text())
    return dict(limit=limits['cpu'],legacy_charged=legacy_charged(state,'cpu',True),legacy_pending=pending_forecasts(state)['cpu'],gpu_limit=limits['gpu'],gpu_charged=legacy_charged(state,'gpu',True))


def spec(ds,fold,seed):
    part=json.loads((ROOT/f'splits/{ds}_kfold5.json').read_text())[f'fold_{fold}']
    services=json.loads((ROOT/'results_v2/S3v2_protocol/masks_manifest.json').read_text())['datasets'][ds]['services']
    methods=13 if ds.startswith('re2_') else 35 if fold==0 and seed==42 else 30
    return dict(id=f's3v2_models_{ds}_f{fold}_s{seed}',resource='cpu' if ds=='aiops' else 'gpu',cores=1,max_attempts=2,timeout_seconds=600,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        expected_config=dict(dataset=ds,fold=fold,seed=seed,component='frozen_models'),
        command=['-m','exp_revision.s3v2_infer','--dataset',ds,'--fold',str(fold),'--seed',str(seed)],
        test_ids=part['test'],score_shape=[methods,4,len(part['test']),len(services)],expected_status=['completed_s3v2_frozen_models'])


OFFICIAL_GROUPS=[(ds,'BARO') for ds in DATASETS]+[('tt','RCD'),('aiops','RCD'),('re2_ss','RCD'),('tt','CIRCA+rank-prune'),('re2_ob','CIRCA+rank-prune')]


def official_spec(ds,fold,seed,method):
    job=spec(ds,fold,seed)
    job.update(timeout_seconds=3600 if (ds,method) in [('re2_ob','CIRCA+rank-prune'),('re2_ss','RCD')] else 600,id=f's3v2_official_{ds}_{method.replace("+","plus")}_f{fold}_s{seed}',resource='cpu',
        expected_config=dict(dataset=ds,fold=fold,seed=seed,component='official',method=method),
        command=['-m','exp_revision.s3v2_official_v2','--dataset',ds,'--fold',str(fold),'--seed',str(seed),'--method',method],
        score_shape=job['score_shape'][1:],expected_status=['completed_s3v2_official','limited_s3v2_official'])
    return job


def refresh_scientific_blocks(state):
    blocked=set()
    for row in state['jobs'].values():
        cfg=row['spec']['expected_config']
        if cfg.get('component')=='official' and row['status']=='completed':
            meta=json.loads((ROOT/row['output']/'run_meta.json').read_text())
            if meta.get('scientific_stop'):blocked.add((cfg['dataset'],cfg['method']))
    for row in state['jobs'].values():
        cfg=row['spec']['expected_config']
        if row['status']=='pending' and cfg.get('component')=='official' and (cfg['dataset'],cfg['method']) in blocked:
            row['status']='scientific_blocked'
            row['reason']='failure rate exceeded10% in a completed official job; no more jobs admitted in this dataset/method'
    return blocked


def run(full=False,official=False):

    FOLDER.mkdir(parents=True,exist_ok=True);q.FOLDER=FOLDER;q.guarded_command=guarded_command
    with (FOLDER/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=FOLDER/'queue_state.json'
        state=json.loads(path.read_text()) if path.exists() else dict(jobs={},status='new')
        official_groups=OFFICIAL_GROUPS
        if full and official:
            official_groups=[(ds,m) for ds,m in OFFICIAL_GROUPS if state['jobs'].get(official_spec(ds,0,42,m)['id'],{}).get('status')=='completed']
            if not official_groups:raise ValueError('no completed official pilots yet')
        jobs=([official_spec(ds,fold,seed,method) for ds,method in official_groups for fold in (range(5) if full else [0]) for seed in (range(42,47) if full else [42])] if official else
              [spec(ds,fold,seed) for ds in DATASETS for fold in (range(5) if full else [0]) for seed in (range(42,47) if full else [42])])
        for job in jobs:
            if job['id'] not in state['jobs']:state['jobs'][job['id']]=dict(spec=job,status='pending',attempts=[])
        budget=remaining_budget()
        if full:
            pilots=[state['jobs'][official_spec(ds,0,42,m)['id']] for ds,m in official_groups] if official else [state['jobs'][f's3v2_models_{ds}_f0_s42'] for ds in DATASETS]
            if not all(r['status']=='completed' for r in pilots):raise ValueError('all group pilot/parity runs required')
            estimates={(r['spec']['expected_config']['dataset'],r['spec']['expected_config'].get('method')):max(10,r['attempts'][-1]['elapsed_seconds'])*1.5/3600 for r in pilots}
            blocked=set()
            for r in pilots:
                meta=json.loads((ROOT/r['output']/'run_meta.json').read_text())
                if meta.get('scientific_stop'):blocked.add((r['spec']['expected_config']['dataset'],r['spec']['expected_config'].get('method')))
            for row in state['jobs'].values():
                cfg=row['spec']['expected_config']
                if row['status']=='pending' and (cfg['dataset'],cfg.get('method')) in blocked:row['status']='scientific_blocked'
            pending=[r for r in state['jobs'].values() if r['status']=='pending' and (r['spec']['expected_config']['dataset'],r['spec']['expected_config'].get('method')) in estimates]
            forecast=sum(estimates[(r['spec']['expected_config']['dataset'],r['spec']['expected_config'].get('method'))] for r in pending)
            gpu_forecast=sum(estimates[(r['spec']['expected_config']['dataset'],r['spec']['expected_config'].get('method'))] for r in pending if r['spec']['resource']=='gpu')
            if budget['legacy_charged']+budget['legacy_pending']+cost(state,True)+forecast>budget['limit']:raise ValueError('forecast exceeds remaining campaign CPU budget')
            if budget['gpu_charged']+cost(state,True,gpu=True)+gpu_forecast>budget['gpu_limit']:raise ValueError('forecast exceeds remaining campaign GPU budget')
            state['forecast']=dict(additional_cpu_hours=forecast,additional_gpu_hours=gpu_forecast,per_job_hours={str(k):v for k,v in estimates.items()},basis='full-fold pilots;1.5x observed wall/CPU1')
        guard=DiskGuard(ROOT);processes={}
        for name,row in state['jobs'].items():
            if row['status']=='running' and q.alive(row['attempts'][-1]):
                a=row['attempts'][-1];guard.reserve(a['key'],ROOT/a['output'],ROOT/a['log'],child_pid=a['pid'])
        while True:
            for name,row in state['jobs'].items():
                if row['status']=='running' and not q.alive(row['attempts'][-1]):
                    q.finish(row['spec'],row,row['attempts'][-1],processes.pop(name,None),guard)
            refresh_scientific_blocks(state)
            budget=remaining_budget();active=sum(r['status']=='running' for r in state['jobs'].values())
            storage=guard.poll();limited=False
            # On unexpected code/parity failures stop admission, preserving all scores/logs.
            failed=any(r['status']=='stuck'  for r in state['jobs'].values())
            for name,row in state['jobs'].items():
                if row['status']!='pending' or time.time()<row.get('retry_after',0) or failed:continue
                if active>=10 or not storage['safe_to_launch']:break
                if row['spec']['resource']=='gpu' and sum(r['status']=='running' and r['spec']['resource']=='gpu' for r in state['jobs'].values())>=3:continue
                bound=(row['spec']['timeout_seconds']+10)/3600
                if budget['legacy_charged']+budget['legacy_pending']+cost(state,True)+bound>budget['limit']:limited=True;break
                if row['spec']['resource']=='gpu' and budget['gpu_charged']+cost(state,True,gpu=True)+bound>budget['gpu_limit']:limited=True;continue
                worker_module=row['spec']['command'][1]
                row['inference_source_sha256']=sha256(ROOT/(worker_module.replace('.','/')+'.py'))
                processes[name]=q.start(row['spec'],row,state,guard);active+=1
                row['attempts'][-1].update(resource=row['spec']['resource'],inference_source_sha256=row['inference_source_sha256'])
                storage=guard.poll()
            state.update(updated=q.now(),budget=budget,cpu_hours=cost(state),gpu_hours=cost(state,gpu=True),counts=dict(Counter(r['status'] for r in state['jobs'].values())))
            if active:state['status']='running'
            elif failed:state['status']='stuck'
            elif limited:state['status']='budget_limited'
            elif all(r['status'] in ('completed','scientific_blocked') for r in state['jobs'].values()):state['status']='completed_with_limits' if any(r['status']=='scientific_blocked' for r in state['jobs'].values()) else 'completed'
            elif not storage['safe_to_launch']:state['status']='storage_limited'
            else:state['status']='running'
            json_dump(path,state)
            if state['status']!='running':return dict(status=state['status'],counts=state['counts'],cpu_hours=state['cpu_hours'])
            for process in processes.values():process.poll()
            time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--official',action='store_true');a=p.parse_args();print(json.dumps(run(a.full,a.official)))
