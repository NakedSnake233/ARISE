"""Post-freeze descriptive development comparison; never imported by training."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,json_dump,sha256,dataset_dir
from . import remaining_queue as q
from .metrics import ranking_metrics
from .disk_guard_fast import DiskGuard


def evaluate():
    s=json.loads((q.FOLDER/'queue_state.json').read_text())
    rows=[r for r in s['jobs'].values() if r['spec']['expected_config'].get('component')=='arise_missing_v2' and r['status']=='completed']
    if not rows:raise ValueError('no completed R1 jobs')
    validated=[];inputs={}
    for r in rows:
        job=r['spec'];folder=ROOT/r['output']
        if not q.verified(folder,job):raise ValueError('R1 verification failed')
        meta=json.loads((folder/'run_meta.json').read_text())
        if sha256(folder/'router.json')!=meta['router_sha256']:raise ValueError('router changed')
        for p in [folder/'scores.npz',folder/'run_meta.json',folder/'router.json']:inputs[str(p.relative_to(ROOT))]=sha256(p)
        validated.append((r,folder))
    parts=[];coverage=[];routers={}
    base=pd.read_parquet(ROOT/'results_v2/T2/predictions.parquet')
    variants=pd.read_parquet(ROOT/'results_v2/T3/predictions.parquet')
    for name in ['T2','T3']:inputs[f'results_v2/{name}/predictions.parquet']=sha256(ROOT/f'results_v2/{name}/predictions.parquet')
    for r,folder in validated:
        cfg=r['spec']['expected_config'];ds=cfg['dataset'];seed=cfg['seed'];fold=cfg['fold']
        path=dataset_dir(ds)/'incidents.json';labels={v['incident_id']:v['root_service'] for v in json.loads(path.read_text())}
        services=json.loads((dataset_dir(ds)/'metadata.json').read_text())['services'];inputs[str(path.relative_to(ROOT))]=sha256(path)
        with np.load(folder/'scores.npz') as a:scores=a['scores'].copy();ids=a['incident_ids'].tolist();scenes=a['scenario_keys'].tolist()
        records=[]
        for values,scene in zip(scores,scenes):
            scenario,param=scene.split('_',1);param='' if param=='none' else param
            ranks=np.argsort(-values,axis=1,kind='stable')
            for key,order in zip(ids,ranks):
                records.append(dict(dataset=ds,method='ARISE-R1',seed=seed,fold=fold,incident_id=key,scenario=scenario,scenario_param=param,
                    rank_of_root=int(np.flatnonzero(order==services.index(labels[key]))[0]+1)))
        frame=pd.DataFrame(records);parts.append(frame)
        for original,methods in [(base,['ARISE','ET-spec','ET-spec+gate+aug','Eadro','Eadro+MD','context-only']),
                                 (variants,['ARISE[V-d]','ARISE[V-b]'])]:
            comparison=original[(original.dataset==ds)&(original.seed==seed)&original.incident_id.isin(ids)&original.method.isin(methods)&(original.scenario!='S4')].copy()
            comparison['scenario_param']=comparison.scenario_param.fillna('')
            if len(comparison)!=len(ids)*len(scenes)*len(methods):raise ValueError('baseline paired coverage mismatch')
            comparison['fold']=fold;parts.append(comparison[frame.columns])
        coverage.append(dict(dataset=ds,fold=fold,seed=seed,events=len(ids),job=r['spec']['id']))
        routers[r['spec']['id']]=json.loads((folder/'router.json').read_text())
    all_rows=pd.concat(parts,ignore_index=True)
    key=['dataset','method','seed','incident_id','scenario','scenario_param']
    if all_rows.duplicated(key).any():raise ValueError('duplicate paired rows')
    rate_path=ROOT/'results_v2/T2/realized_rates.csv'
    rates=pd.read_csv(rate_path,keep_default_na=False,dtype={'scenario_param':str})
    inputs[str(rate_path.relative_to(ROOT))]=sha256(rate_path)
    rate_keys=['dataset','seed','incident_id','scenario','scenario_param']
    all_rows=all_rows.merge(rates[rate_keys+['deleted','original_available']],on=rate_keys,how='left',validate='many_to_one')
    if all_rows[['deleted','original_available']].isna().any().any():raise ValueError('missing actual deletion denominator')
    summaries=[]
    for keys,g in all_rows.groupby(['dataset','method','scenario','scenario_param','seed'],dropna=False):
        summaries.append(dict(zip(['dataset','method','scenario','scenario_param','seed'],keys))|ranking_metrics(g.rank_of_root.to_numpy())|
            dict(deleted_cells=int(g.deleted.sum()),original_available_cells=int(g.original_available.sum()),
                 actual_deletion_rate=float(g.deleted.sum()/g.original_available.sum()) if g.original_available.sum() else 0.))
    metrics=pd.DataFrame(summaries)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S');output=ROOT/'results_v2'/('R1_development_'+stamp)
    guard=DiskGuard(ROOT);lease='r1_eval_'+stamp;guard.reserve(lease,output,ROOT/'logs'/(lease+'.log'))
    try:
        output.mkdir();metrics.to_csv(output/'metrics.csv',index=False)
        json_dump(output/'receipt.json',dict(time=q.now(),coverage=coverage,input_sha256=inputs,
            limitation='development comparison on previously inspected benchmark; partial folds/seeds explicitly listed',router=routers))
        print('OUTPUT',output)
        selected=metrics[(metrics.scenario.isin(['S0','S1']))|((metrics.scenario=='S2')&(metrics.scenario_param=='0.50'))]
        print(selected[['dataset','method','scenario','scenario_param','actual_deletion_rate','PR@1','MAP@5']].to_string(index=False))
    finally:guard.release(lease)
    return output


if __name__=='__main__':evaluate()
