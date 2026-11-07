"""Budgeted local pair-holdout stages; previous campaign commitments retained."""
import argparse
from collections import Counter
import fcntl
import json
import subprocess
import time
from .common import ROOT,json_dump,sha256
from . import remaining_queue as q
from .disk_guard_fast import DiskGuard,guarded_command,InsufficientSpace
from .s3v2_queue import remaining_budget,cost as s3cost
from .tt_pair_data import FOLDER,TASK,SPLIT

COMPONENTS=[('context',None),('experts',None),('variants',None),('controls',None),
            ('baseline','Eadro'),('baseline','Eadro+MD'),('baseline','MULAN')]


def identifier(fold,seed,component,method=None,smoke=False):
    return f'pair_{"smoke" if smoke else "full"}_{method or component}_f{fold}_s{seed}'


def specification(fold,seed=None,component='prepare',method=None,smoke=False):
    part=json.loads(SPLIT.read_text())[f'fold_{fold}'];ids=part['test'][:5] if smoke else part['test']
    cfg=dict(task=TASK,component='train_only_preprocessing' if component=='prepare' else component,dataset='tt',fold=fold,smoke=smoke,split=part['split'])
    if component=='prepare':
        name=f'pair_prepare_f{fold}';command=['-m','exp_revision.tt_pair_data','--fold',str(fold)];shape=[1,len(ids),41]
        status='completed_generalization_preprocessing'
    else:
        name=identifier(fold,seed,component,method,smoke);cfg.update(seed=seed,scenarios=['S0_none','S3v2_all_0.10','S3v2_all_0.25','S3v2_all_0.50'])
        if method:cfg['method']=method
        command=['-m','exp_revision.tt_pair_models','--component',component,'--fold',str(fold),'--seed',str(seed)]
        if method:command+=['--method',method]
        if smoke:command+=['--smoke']
        shape=[4,len(ids),41]
        if component in ('experts','variants','controls'):shape=[{'experts':4,'variants':2,'controls':5}[component],*shape]
        status=f'completed_generalization_{component}'
    return dict(id=name,command=command,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        resource='gpu' if component=='context' else 'cpu',cores=1 if component in ('prepare','baseline') else 4,
        timeout_seconds=1800 if component=='context' else 1200 if component in ('prepare','variants','experts') else 600,
        max_attempts=2,expected_config=cfg,test_ids=ids,score_shape=shape,expected_status=[status],
        dependencies=[] if component=='prepare' else [f'pair_prepare_f{fold}'])


def charged(state,running=False,gpu=False):
    return sum(a.get('elapsed_seconds',r['spec']['timeout_seconds']+10 if running else 0)*
        (1 if gpu else r['spec']['cores'])/3600 for r in state['jobs'].values() for a in r['attempts']
        if not gpu or r['spec']['resource']=='gpu')


def budget():
    legacy=remaining_budget();s3=json.loads((ROOT/'results_v2/S3v2_models/queue_state.json').read_text())
    pending=0.
    for row in s3['jobs'].values():
        if row['status'] not in ('pending','stuck'):continue
        cfg=row['spec']['expected_config'];key=str((cfg['dataset'],cfg.get('method')))
        pending+=s3.get('forecast',{}).get('per_job_hours',{}).get(key,(row['spec']['timeout_seconds']+10)/3600)
    path=ROOT/'docs/tt_pair_compute_authorization.json';approval=json.loads(path.read_text()) if path.exists() else {}
    approved=(approval.get('approved') is True and approval.get('source')=='explicit author reply in this conversation'
              and approval.get('limits')=={'cpu':380.,'gpu':64.} and approval.get('scope')=='tt_pair_holdout_validation')
    return dict(approved=approved,cpu_limit=380. if approved else legacy['limit'],gpu_limit=64. if approved else legacy['gpu_limit'],
        prior_cpu=legacy['legacy_charged']+legacy['legacy_pending']+s3cost(s3,True)+pending,
        prior_gpu=legacy['gpu_charged']+s3cost(s3,True,gpu=True),s3v2_pending_reserved=pending)


def prerequisites(state,stage):
    if stage not in ('pilot','full'):return
    mode=True if stage=='pilot' else False
    for component,method in COMPONENTS:
        key=identifier(0,42,component,method,mode);row=state['jobs'].get(key)
        if not row or row['status']!='completed' or not q.verified(ROOT/row['output'],row['spec']):raise ValueError('incomplete prerequisite: '+key)


def run(stage='smoke',plan_only=False):
    FOLDER.mkdir(parents=True,exist_ok=True);q.FOLDER=FOLDER;q.guarded_command=guarded_command
    with (FOLDER/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=FOLDER/'queue_state.json';state=json.loads(path.read_text()) if path.exists() else dict(jobs={})
        prerequisites(state,stage)
        folds=range(5) if stage=='full' else [0];seeds=range(42,47) if stage=='full' else [42]
        jobs=[]
        for fold in folds:
            jobs.append(specification(fold))
            for seed in seeds:
                jobs.extend(specification(fold,seed,component,method,stage=='smoke') for component,method in COMPONENTS)
        for job in jobs:
            if job['id'] not in state['jobs']:state['jobs'][job['id']]=dict(spec=job,status='pending',attempts=[])
            elif not state['jobs'][job['id']]['attempts']:
                # A dry-run may precede the implementation commit. Refresh only
                # provenance for never-started jobs; never rewrite attempted specs.
                state['jobs'][job['id']]['spec']['source_commit']=job['source_commit']
        b=budget();state.update(stage=stage,budget=b,status='planned',counts=dict(Counter(r['status'] for r in state['jobs'].values())))
        if stage=='full':
            estimates={('train_only_preprocessing',None):state['jobs']['pair_prepare_f0']['attempts'][-1]['elapsed_seconds']/3600*1.5}
            for component,method in COMPONENTS:
                r=state['jobs'][identifier(0,42,component,method)];estimates[(component,method)]=r['attempts'][-1]['elapsed_seconds']/3600*1.5
            cpu=gpu=0.
            for r in state['jobs'].values():
                if r['status']!='pending':continue
                cfg=r['spec']['expected_config'];h=estimates[(cfg['component'],cfg.get('method'))]
                cpu+=h*r['spec']['cores'];gpu+=h if r['spec']['resource']=='gpu' else 0
            state['forecast']=dict(cpu=cpu,gpu=gpu,basis='matched complete-fold pilot;1.5x actual wall and declared cores')
            if b['prior_cpu']+charged(state,True)+cpu>b['cpu_limit'] or b['prior_gpu']+charged(state,True,True)+gpu>b['gpu_limit']:
                state['status']='forecast_budget_limited';json_dump(path,state);return {k:v for k,v in state.items() if k!='jobs'}
        json_dump(path,state)
        if plan_only:return {k:v for k,v in state.items() if k!='jobs'}
        if not b['approved']:
            state['status']='awaiting_budget_authorization';json_dump(path,state);return {k:v for k,v in state.items() if k!='jobs'}
        guard=DiskGuard(ROOT);processes={}
        for r in state['jobs'].values():
            if r['status']=='running' and q.alive(r['attempts'][-1]):
                a=r['attempts'][-1];guard.reserve(a['key'],ROOT/a['output'],ROOT/a['log'],child_pid=a['pid'])
        while True:
            for name,r in state['jobs'].items():
                if r['status']=='running' and not q.alive(r['attempts'][-1]):q.finish(r['spec'],r,r['attempts'][-1],processes.pop(name,None),guard)
            b=budget();storage=guard.poll();limited=False;storage_limited=not storage['safe_to_launch'];active=sum(r['status']=='running' for r in state['jobs'].values())
            failed=any(r['status']=='stuck' for r in state['jobs'].values())
            for name,r in state['jobs'].items():
                if r['status']!='pending' or failed or time.time()<r.get('retry_after',0):continue
                if any(state['jobs'][d]['status']!='completed' for d in r['spec']['dependencies']):continue
                if active>=8 or not storage['safe_to_launch']:break
                if r['spec']['resource']=='gpu' and sum(z['status']=='running' and z['spec']['resource']=='gpu' for z in state['jobs'].values())>=3:continue
                wall=(r['spec']['timeout_seconds']+10)/3600
                if b['prior_cpu']+charged(state,True)+wall*r['spec']['cores']>b['cpu_limit']:limited=True;break
                if r['spec']['resource']=='gpu' and b['prior_gpu']+charged(state,True,True)+wall>b['gpu_limit']:limited=True;continue
                r['worker_source_sha256']=sha256(ROOT/(r['spec']['command'][1].replace('.','/')+'.py'))
                try:
                    processes[name]=q.start(r['spec'],r,state,guard)
                except InsufficientSpace:
                    # poll describes existing reservations; reserve atomically
                    # checks the additional job, including other queue activity.
                    # No attempt is created when that reservation is denied.
                    storage_limited=True
                    break
                active+=1;storage=guard.poll();storage_limited=not storage['safe_to_launch']
            state.update(updated=q.now(),budget=b,cpu_hours=charged(state),gpu_hours=charged(state,gpu=True),counts=dict(Counter(r['status'] for r in state['jobs'].values())))
            state['status']='running' if active else 'stuck' if failed else 'budget_limited' if limited else 'completed' if all(r['status']=='completed' for r in state['jobs'].values()) else 'storage_limited' if storage_limited else 'dependency_blocked'
            json_dump(path,state)
            if state['status']!='running':return {k:v for k,v in state.items() if k!='jobs'}
            for process in processes.values():process.poll()
            time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','pilot','full'],default='smoke');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.stage,a.plan_only)))
