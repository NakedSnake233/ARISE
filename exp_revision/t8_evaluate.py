"""Evaluate only verified score artifacts; failed rows retain rank infinity."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,config_snapshot,dataset_dir,json_dump,sha256
from . import re2_data as d
from . import remaining_queue as q
from . import t8_manifest as manifest_module
from .metrics import ranking_metrics
from .missingness import SCENARIOS
from .train_evidence import fuse_all


def score_rows(scores,failed,ids,services,truth,cfg,method,scenes,rates=None):
    rows=[]
    if failed.shape != scores.shape[:2]:raise ValueError('failure mask shape differs')
    for j,scene in enumerate(scenes):
        scenario,param=scene.split('_',1);param='' if param=='none' else param
        for i,key in enumerate(ids):
            root,fault=truth[key];valid=not failed[j,i] and np.isfinite(scores[j,i]).all()
            order=np.argsort(-scores[j,i],kind='stable') if valid else np.arange(len(services))
            rank=int(np.flatnonzero(order==services.index(root))[0]+1) if valid else float('inf')
            rows.append(dict(task='T7' if cfg['task']=='T7' else 'T8',dataset=cfg['dataset'],method=method,split='kfold5',fold=cfg['fold'],
                seed=42+cfg['repeat'] if 'repeat' in cfg else cfg['seed'],repeat=cfg.get('repeat'),scenario=scenario,scenario_param=param,
                incident_id=key,root_service=root,fault_type=fault,root_rank=rank,top5=[services[int(k)] for k in order[:5]] if valid else [],
                status='ok' if valid else 'failed'))
            seed=42+cfg['repeat'] if 'repeat' in cfg else cfg['seed']
            rows[-1]['realized_rate']=0. if cfg['task']=='T7' else rates[(cfg['dataset'],key,seed,scenario,param)] if rates is not None else None
    return rows


def evaluate(kind,output):
    source=q.MANIFESTS/manifest_module.NAMES[kind,False,False]
    if not manifest_module.complete(source.name):raise ValueError('all admitted full score batches must complete before evaluation')
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    manifest=json.loads(source.read_text());state=manifest_module.state();batches=[]
    # Verify every score, receipt, input and mask before the first held-out
    # evaluation label read. No old evaluator or canonical result is changed.
    for relative,digest in manifest['source_sha256'].items():
        if sha256(ROOT/relative)!=digest:raise ValueError('frozen input changed: '+relative)
    for job in manifest['jobs']:
        folder=ROOT/state['jobs'][job['id']]['output'];meta=json.loads((folder/'run_meta.json').read_text())
        if not q.verified(folder,job):raise ValueError('unverified score batch')
        for group in ('input_sha256','mask_sha256','span_input_sha256'):
            for relative,digest in meta.get(group,{}).items():
                if sha256(ROOT/relative)!=digest:raise ValueError('score input changed: '+relative)
        with np.load(folder/'scores.npz') as a:
            scores=a['scores'].copy();ids=a['incident_ids'].tolist();scenes=a['scenario_keys'].tolist()
            if kind in ('official','llm') and 'failed' not in a.files:raise ValueError('missing failed prediction mask')
            failed=a['failed'].copy() if 'failed' in a.files else np.zeros(scores.shape[-3:-1],dtype=bool)
            methods=a['method_names'].tolist() if 'method_names' in a.files else None
        expected_scenes=['S0_none'] if kind=='llm' else [f'{s}_{p or "none"}' for s,p in SCENARIOS]
        if scenes!=expected_scenes:raise ValueError('score scenario order differs from the protocol')
        if failed.shape!=(len(scenes),len(ids)) or not np.isin(failed,(False,True)).all():raise ValueError('invalid failed prediction mask')
        if job['expected_config']['component']=='evidence':
            if methods!=['ARISE-evidence','ET-spec','ET-spec+gate+aug','RootFreqPrior','Random']:raise ValueError('evidence method order differs')
        elif methods is not None:raise ValueError('unexpected multi-method score archive')
        batches.append((job['expected_config'],scores,failed,ids,scenes,methods))
    truths={};catalogues={}
    for ds in {cfg['dataset'] for cfg,*_ in batches}:
        if ds in d.DATASETS:
            labels=json.loads((ROOT/f'splits/{ds}_evaluation_labels.json').read_text());services=json.loads(d.CATALOG.read_text())[ds]['services']
        else:
            folder=dataset_dir(ds);labels=json.loads((folder/'incidents.json').read_text());services=json.loads((folder/'metadata.json').read_text())['services']
        truths[ds]={r['incident_id']:(r['root_service'],r['fault_type']) for r in labels};catalogues[ds]=services
    contexts={(cfg['dataset'],cfg['fold'],cfg['seed']):(scores,ids,scenes) for cfg,scores,failed,ids,scenes,methods in batches if cfg['component']=='context'}
    rates={}
    for path in (ROOT/'results_v2/T2/realized_rates.csv',ROOT/'results_v2/T8_inputs/realized_rates.csv'):
        if path.exists():
            table=pd.read_csv(path,dtype={'scenario_param':str},keep_default_na=False)
            for r in table.itertuples(index=False):rates[r.dataset,r.incident_id,int(r.seed),r.scenario,r.scenario_param]=float(r.realized_rate)
    rows=[]
    for cfg,scores,failed,ids,scenes,methods in batches:
        if methods:
            for k,method in enumerate(methods):
                values=scores[k]
                if method=='ARISE-evidence':
                    ctx,ctx_ids,ctx_scenes=contexts[cfg['dataset'],cfg['fold'],cfg['seed']]
                    if ctx_ids!=ids or ctx_scenes!=scenes:raise ValueError('fusion incident/scenario order differs')
                    values=np.stack([fuse_all((.25,c),(.75,e)) for c,e in zip(ctx,values)]);method='ARISE'
                rows+=score_rows(values,failed,ids,catalogues[cfg['dataset']],truths[cfg['dataset']],cfg,method,scenes,rates)
        else:
            method='context-only' if cfg['component']=='context' else cfg.get('method',cfg['component'])
            rows+=score_rows(scores,failed,ids,catalogues[cfg['dataset']],truths[cfg['dataset']],cfg,method,scenes,rates)
    frame=pd.DataFrame(rows);key=['dataset','method','incident_id','seed','scenario','scenario_param']
    if frame.duplicated(key).any():raise ValueError('duplicate evaluated predictions')
    expected=len(d.DATASETS)*90*5*12*12 if kind=='models' else 548*5*12*3 if kind=='official' else 548*3*2
    if len(frame)!=expected:raise ValueError(f'full experiment coverage differs: {len(frame)} vs {expected}')
    output.mkdir(parents=True,exist_ok=True);frame.to_parquet(output/'predictions.parquet',index=False)
    per_run=[]
    for values,group in frame.groupby(['dataset','method','scenario','scenario_param','seed'],dropna=False):
        per_run.append(dict(zip(['dataset','method','scenario','scenario_param','seed'],values))|ranking_metrics(group.root_rank.to_numpy()))
    summary=pd.DataFrame(per_run);summary.to_csv(output/'per_run_metrics.csv',index=False)
    aggregated=summary.groupby(['dataset','method','scenario','scenario_param'])[['PR@1','PR@3','PR@5','MAP@5','MRR']].agg(['mean','std','count'])
    aggregated.to_csv(output/'metrics_mean_sd.csv')
    failure_rates={"|".join(k):float((g.status=='failed').mean()) for k,g in frame.groupby(['dataset','method','scenario','scenario_param'])}
    json_dump(output/'evaluation_meta.json',dict(kind=kind,rows=len(frame),failed=int((frame.status=='failed').sum()),manifest=str(source.relative_to(ROOT)),
        evaluated=dt.datetime.now(dt.timezone.utc).isoformat(),manifest_sha256=sha256(source),predictions_sha256=sha256(output/'predictions.parquet'),
        failure_policy='rank infinity, empty top5, retained in denominator',failure_rates=failure_rates,repeats=3 if kind=='llm' else 5))
    if any(v>.1 for v in failure_rates.values()):raise ValueError('failure rate above 10%; retain results and ask author')
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('models','official','llm'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(evaluate(a.kind,a.output))


if __name__=='__main__':main()
