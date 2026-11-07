"""Post-freeze labels-only scoring; full matrices and pilot comparisons separate."""
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,json_dump,sha256,dataset_dir
from .scenario_v2 import load_cases,mask_path,OUTPUT as MASK_ROOT
from .metrics import ranking_metrics
from .disk_guard_fast import DiskGuard


def collect(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    state=json.loads((ROOT/'results_v2/S3v2_models/queue_state.json').read_text())
    rows=[];coverage=[];inputs={};loaded={};masks={}
    for name,job in state['jobs'].items():
        cfg=job['spec']['expected_config'];ds=cfg['dataset']
        coverage.append(dict(job=name,**cfg,status=job['status'],output=job.get('output')))
        if job['status']!='completed':continue
        folder=ROOT/job['output'];meta=json.loads((folder/'run_meta.json').read_text())
        if sha256(folder/'scores.npz')!=meta['scores_sha256']:raise ValueError('score checksum changed')
        for p,h in meta['mask_sha256'].items():
            if sha256(ROOT/p)!=h:raise ValueError('mask changed after inference')
        inputs[str((folder/'run_meta.json').relative_to(ROOT))]=sha256(folder/'run_meta.json')
        if ds not in loaded:
            services,cases,_=load_cases(ds)
            path=ROOT/f'splits/{ds}_evaluation_labels.json' if ds.startswith('re2_') else dataset_dir(ds)/'incidents.json'
            labels={r['incident_id']:services.index(r['root_service']) for r in json.loads(path.read_text())}
            loaded[ds]=(services,{key:a for _,key,a in cases},labels,len(cases))
            inputs[str(path.relative_to(ROOT))]=sha256(path)
        services,a0s,labels,_=loaded[ds]
        with np.load(folder/'scores.npz') as z:
            ids=z['incident_ids'].tolist();scenes=z['scenario_keys'].tolist();scores=z['scores'].copy()
            if 'method_names' in z:methods=z['method_names'].tolist();failed=np.zeros(scores.shape[:-1],bool)
            else:methods=[cfg['method']];scores=scores[None];failed=z['failed'][None]
        if ids!=job['spec']['test_ids'] or scenes!=['S0_none','S3v2_all_0.10','S3v2_all_0.25','S3v2_all_0.50']:raise ValueError('identity/scene mismatch')
        for method,values,flags in zip(methods,scores,failed):
            for j,scene in enumerate(scenes):
                fraction='none' if j==0 else scene.rsplit('_',1)[1]
                if j and (ds,cfg['seed'],fraction) not in masks:
                    with np.load(mask_path(MASK_ROOT,ds,float(fraction),cfg['seed'])) as z:masks[(ds,cfg['seed'],fraction)]={key:z[key].copy() for key in z.files}
                ranks=np.argsort(-values[j],axis=-1,kind='stable')
                for i,key in enumerate(ids):
                    a0=a0s[key];k=np.ones_like(a0) if j==0 else masks[(ds,cfg['seed'],fraction)][key];a=a0*k;target=labels[key]
                    rank=float(np.flatnonzero(ranks[i]==target)[0]+1) if not flags[j,i] else float('inf')
                    stratum='root_originally_unobserved' if not a0[target].any() else 'root_newly_missing' if not a[target].any() else 'root_still_observed'
                    rows.append(dict(dataset=ds,method=method,fold=cfg['fold'],seed=cfg['seed'],incident_id=key,
                        scenario='S0' if j==0 else 'S3v2_all',scenario_param=fraction,rank_of_root=rank,failed=bool(flags[j,i]),
                        stratum=stratum,deleted=int((a0*(1-k)).sum()),original_available=int(a0.sum()),
                        eligible_services=int(a0.any(1).sum()),deleted_services=int((a0.any(1)&~a.any(1)).sum()),
                        invalid_by_design=method=='ARISE[V-a]',source_job=name))
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('no completed scores')
    if frame.duplicated(['dataset','method','seed','incident_id','scenario','scenario_param']).any():raise ValueError('duplicate prediction keys')
    frame.to_parquet(output/'predictions.parquet',index=False)
    summaries=[];paired=[]
    for keys,g in frame.groupby(['dataset','method','scenario','scenario_param'],sort=True):
        ds,method,scene,param=keys;expected=loaded[ds][3]*5
        full=(len(g)==expected and g.incident_id.nunique()==loaded[ds][3] and set(g.seed)==set(range(42,47)) and set(g.fold)==set(range(5)))
        base=dict(dataset=ds,method=method,scenario=scene,scenario_param=param,
            coverage='full_5fold_5seed' if full else 'development_pilot' if method.startswith(('ARISE-R1','ARISE-R2','ARISE-R3')) else 'partial',
            invalid_by_design=bool(g.invalid_by_design.iloc[0]))
        for stratum in ('all','root_newly_missing','root_still_observed','root_originally_unobserved'):
            part=g if stratum=='all' else g[g.stratum==stratum]
            metric=ranking_metrics(part.rank_of_root.to_numpy()) if len(part) else {k:None for k in ('PR@1','PR@3','PR@5','MAP@3','MAP@5','MRR')}
            summaries.append(dict(**base,stratum=stratum,event_seed_rows=len(part),events=part.incident_id.nunique(),
                failures=int(part.failed.sum()),failure_rate=float(part.failed.mean()) if len(part) else None,
                actual_cell_deletion_rate=float(part.deleted.sum()/max(1,part.original_available.sum())),
                actual_service_deletion_fraction=float(part.deleted_services.sum()/max(1,part.eligible_services.sum())),**metric))
        if ds in ('tt','aiops'):
            part=g[(g.fold==0)&(g.seed==42)]
            if len(part):paired.append(dict(**base,paired_scope='fold0_seed42_development_only',events=len(part),**ranking_metrics(part.rank_of_root.to_numpy())))
    metrics=pd.DataFrame(summaries);metrics.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_pilot_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'job_coverage.csv',index=False)
    receipt=dict(status='postfreeze_s3v2_evaluation',prediction_rows=len(frame),queue_counts=state['counts'],
        cpu_hours=state.get('cpu_hours'),gpu_hours=state.get('gpu_hours'),input_receipts=inputs,
        failure_policy='all algorithm failure records retained; infinite root rank means zero for all ranking metrics',
        missing_population_policy='no drops or seed selection; zero-size strata explicitly counted',
        uncertainty='descriptive only, repeated event masks are not independent incidents',
        source_sha256=sha256(Path(__file__)))
    json_dump(output/'receipt.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    key='s3v2_collection';guard=DiskGuard(ROOT);guard.reserve(key,a.output,ROOT/'logs/s3v2_collection.log')
    try:result=collect(a.output);print(json.dumps({k:v for k,v in result.items() if k!='input_receipts'}))
    finally:guard.release(key)
