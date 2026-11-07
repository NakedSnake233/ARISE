"""Reviewed smoke gates, frozen inputs and budget forecasts for T7/T8."""
from __future__ import annotations
import argparse
import json
import math
import subprocess
from .common import ROOT,json_dump,sha256
from . import re2_data as d
from . import remaining_queue as q

NAMES={('models',True,False):"46_smoke_re2_models.json",('models',False,False):"47_full_re2_models.json",
       ('official',True,True):"48_smoke_official_legacy.json",('official',True,False):"49_smoke_official_re2.json",('official',False,False):"50_full_official_baselines.json",
       ('llm',True,True):"51_smoke_llm_legacy.json",('llm',True,False):"52_smoke_llm_re2.json",('llm',False,False):"53_full_llm_baselines.json"}
COMPONENTS=('context','evidence','Eadro','MULAN','Eadro+MD','MULAN+MD','Nezha','MicroRCA')


def state():return json.loads((q.FOLDER/"queue_state.json").read_text())


def complete(name,s=None):
    path=q.MANIFESTS/name
    if not path.exists():return False
    s=state() if s is None else s
    return all((r:=s['jobs'].get(j['id'])) and r['status']=='completed' and q.verified(ROOT/r['output'],j) for j in json.loads(path.read_text())['jobs'])


def dependencies(kind,smoke,legacy):
    required=[]
    if kind=='llm':
        approval=ROOT/'docs/llm_external_authorization.json'
        if not approval.exists() or json.loads(approval.read_text()).get('approved') is not True:
            raise ValueError('external OpenCode telemetry requests require the author approval requested after automatic review rejection')
    if legacy and smoke:required+=['43_full_re2_raw.json']
    if not legacy:
        required+=['44_smoke_re2_fold.json' if smoke else '45_full_re2_fold.json']
        if not (ROOT/'results_v2/T8_inputs/realized_rates.csv').exists():raise ValueError('RE2 masks/rates must precede model inference')
    if not smoke:
        required+=[NAMES[kind,True,False]]
        if kind in ('official','llm'):required+=[NAMES[kind,True,True]]
    if not all(complete(n) for n in required):raise ValueError('required smoke or fold preprocessing is incomplete')
    if kind=='official':
        record=ROOT/'review/T8_environment_followdoc.md'
        if not record.exists() or 'PASS' not in record.read_text():raise ValueError('independent documented environment witness pending')


def closure(kind,datasets):
    names=['exp_revision/t8_manifest.py','exp_revision/common.py','exp_revision/remaining_queue.py','exp_revision/disk_guard.py','exp_revision/disk_guard_local.py',
        'exp_revision/re2_data.py','exp_revision/fold_data.py','exp_revision/baseline_fold.py','exp_revision/baseline_settings.json',
        'scripts/care/fse2027_neural_baselines.py','scripts/care/fse2027_classical_baselines.py','scripts/care/fse2027_baseline_common.py',
        'configs/care/review-response-v1.json',str(d.CATALOG.relative_to(ROOT))]
    names+=['exp_revision/t8_masks.py','exp_revision/missingness.py']
    if kind=='models':names+=['exp_revision/re2_models.py','exp_revision/generalization_models.py','exp_revision/train_evidence.py',
        'exp_revision/train_variant_evidence.py','exp_revision/train_context_variant.py','exp_revision/train_context.py','exp_revision/masked_evidence.py',
        'exp_revision/raw_spans_v2.py','scripts/care/evidence_reranker.py','scripts/care/robust_evidence.py']
    elif kind=='official':
        names+=['exp_revision/official_baselines.py','exp_revision/rcaeval_worker.py','exp_revision/rcaeval_env_spec.json']
        names+=[str(p.relative_to(ROOT)) for p in (ROOT/'third_party/rcaeval_259ea416/RCAEval').rglob('*.py')]
        names+=['third_party/rcaeval_259ea416/lib/causallearn/utils/PCUtils/SkeletonDiscovery.py','third_party/rcaeval_259ea416/lib/causallearn/graph/GraphClass.py']
    else:names+=['exp_revision/llm_fold.py','scripts/care/fse2027_llm_baselines.py','scripts/care/fse2027_llm_client.py','scripts/care/fse2027_llm_prompts_budget_v2.json','configs/agents.yaml','docs/llm_external_authorization.json']
    for ds in datasets:
        names+=[f'splits/{ds}_kfold5.json']+[f'splits/{ds}_training_labels/fold_{f}.json' for f in range(5)]
        if ds in d.DATASETS:
            names+=[f'splits/{ds}_inner_validation.json']
            for f in range(5):
                try:folder=d.fold_folder(ds,f)
                except ValueError:continue
                names += [str((folder/n).relative_to(ROOT)) for n in ('features.npz','metadata.json')]
        else:
            for f in range(5):names += [f'cache/fold_preprocessing/{ds}/fold{f}/{n}' for n in ('features.npz','metadata.json')]
            if ds=='tt':names+=['splits/tt_inner_validation_kfold5.json']
        for scene,param in [('S0','')] if kind=='llm' else __import__('exp_revision.missingness',fromlist=['SCENARIOS']).SCENARIOS:
            for seed in range(42,47):
                path=ROOT/f'masks/{ds}/{scene}_{param or "none"}/seed{seed}.npz'
                names.append(str(path.relative_to(ROOT)))
    return {n:sha256(ROOT/n) for n in names}


def forecast(kind):
    s=state();pilots=[NAMES[kind,True,False]]+([NAMES[kind,True,True]] if kind!='models' else [])
    total={'cpu':0.,'gpu':0.}; details=[]
    for name in pilots:
        if not complete(name,s):raise ValueError('cannot forecast without all actual smoke receipts')
        for job in json.loads((q.MANIFESTS/name).read_text())['jobs']:
            row=s['jobs'][job['id']];cfg=job['expected_config'];ds=cfg['dataset']
            elapsed=sum(a.get('elapsed_seconds',0) for a in row['attempts'])
            split=json.loads((ROOT/f'splits/{ds}_kfold5.json').read_text())
            if kind=='models':factor=25 # each smoke already trains the complete outer training fold
            else:factor=sum(len(v['test']) for v in split.values())*(3 if kind=='llm' else 5)
            cost=elapsed*job['cores']*factor/3600;total[job['resource']]+=cost
            details.append(dict(dataset=ds,component=cfg.get('component'),method=cfg.get('method'),smoke_seconds=elapsed,forecast_hours=cost,resource=job['resource']))
    return dict(kind=kind,forecast_hours=total,details=details,basis='observed full-training smoke; incident-wise inference scaled by full held-out coverage')


def observed_model():
    versions=set()
    for name in (NAMES['llm',True,True],NAMES['llm',True,False]):
        manifest=json.loads((q.MANIFESTS/name).read_text())
        for job in manifest['jobs']:
            row=state()['jobs'][job['id']];meta=json.loads((ROOT/row['output']/'run_meta.json').read_text());versions.update(meta['returned_models'])
    if len(versions)!=1:raise ValueError('smoke provider model versions differ; cannot freeze one campaign')
    return next(iter(versions))


def outstanding_forecasts(s):
    reserved={'cpu':0.,'gpu':0.}
    for name in (NAMES['models',False,False],NAMES['official',False,False],NAMES['llm',False,False]):
        path=q.MANIFESTS/name
        if not path.exists():continue
        manifest=json.loads(path.read_text());projection=manifest['forecast']
        entries={(e['dataset'],e.get('method') or e['component']):e for e in projection['details']}
        for job in manifest['jobs']:
            row=s['jobs'].get(job['id'])
            if row and row['status'] in ('completed','running'):continue
            cfg=job['expected_config']; entry=entries[cfg['dataset'],cfg.get('method') or cfg['component']]
            factor=1 if projection['kind']=='models' else len(job['test_ids'])
            reserved[job['resource']]+=entry['smoke_seconds']*job['cores']*factor/3600
    return reserved


def create(kind='models',smoke=True,legacy=False):
    path=q.MANIFESTS/NAMES[kind,smoke,legacy]
    if path.exists():return None
    dependencies(kind,smoke,legacy)
    projection=None
    if not smoke:
        projection=forecast(kind);s=state();reserved=outstanding_forecasts(s)
        for resource,cost in projection['forecast_hours'].items():
            if q.charged(s,resource,True)+reserved[resource]+cost>(q.CPU_BUDGET if resource=='cpu' else q.GPU_BUDGET):
                projection['already_admitted_pending_forecast_hours']=reserved
                json_dump(ROOT/f'results_v2/T8_inputs/{kind}_budget_forecast.json',projection)
                raise ValueError('actual smoke forecast exceeds remaining '+resource+' budget; author decision required')
    datasets=('tt','aiops') if legacy else (d.DATASETS if kind=='models' or smoke else ('tt','aiops',*d.DATASETS))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();jobs=[]
    model=observed_model() if kind=='llm' and not smoke else None
    for ds in datasets:
        parts=json.loads((ROOT/f'splits/{ds}_kfold5.json').read_text())
        for fold in range(1 if smoke else 5):
            for seed in ([42] if smoke else (range(3) if kind=='llm' else range(42,47))):
                for component in (COMPONENTS if kind=='models' else ('BARO','CIRCA','RCD') if kind=='official' else ('ReAct','mABC')):
                    ids=parts[f'fold_{fold}']['test'][:3 if kind=='models' else 1] if smoke else parts[f'fold_{fold}']['test']
                    services=len(json.loads(d.CATALOG.read_text())[ds]['services']) if ds in d.DATASETS else len(__import__('exp_revision.fold_data',fromlist=['load_fold']).load_fold(ds,fold).services)
                    resource='gpu' if kind=='models' and component in ('context','Eadro','MULAN','Eadro+MD','MULAN+MD') else 'cpu'
                    cores=4 if kind=='models' and component=='evidence' else 1
                    prefix='smoke_' if smoke else 'full_'; jobid=f'{prefix}{kind}_{ds}_{component.replace("+","plus")}_f{fold}_s{seed}'
                    cfg=dict(task='T7' if kind=='llm' else 'T8',dataset=ds,fold=fold,smoke=smoke)
                    module={'models':'re2_models','official':'official_baselines','llm':'llm_fold'}[kind]
                    cmd=['-m','exp_revision.'+module,'--dataset',ds,'--fold',str(fold)]
                    if kind=='models':cfg.update(component=component,seed=seed);cmd+=['--seed',str(seed),'--component',component];status='completed_re2_scores'
                    elif kind=='official':cfg.update(component='official_baseline',method=component,seed=seed);cmd+=['--seed',str(seed),'--method',component];status='completed_official_scores'
                    else:
                        repeat=0 if smoke else seed;cfg.update(component='llm_baseline',method=component,repeat=repeat,expected_model=model)
                        cmd+=['--repeat',str(repeat),'--method',component];status='completed_llm_scores'
                        if model is not None:cmd+=['--expected-model',model]
                    if smoke:cmd+=['--smoke']
                    shape=[1 if kind=='llm' else 12,len(ids),services]
                    if kind=='models' and component=='evidence':shape=[5,*shape]
                    timeout=1800 if component=='evidence' else 900 if smoke or kind=='models' else 3600
                    jobs.append(dict(id=jobid,resource=resource,cores=cores,max_attempts=2,timeout_seconds=timeout,source_commit=commit,
                        expected_config=cfg,command=cmd,test_ids=ids,score_shape=shape,expected_status=[status]))
    json_dump(path,dict(backend='local',phase=('pilot_' if smoke else 'full_')+kind,source_commit=commit,precharged_hours=q.PRECHARGED,source_sha256=closure(kind,datasets),jobs=jobs,forecast=projection))
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('models','official','llm'),required=True);p.add_argument('--legacy',action='store_true');p.add_argument('--full',action='store_true')
    a=p.parse_args();print(create(a.kind,not a.full,a.legacy))


if __name__=='__main__':main()
