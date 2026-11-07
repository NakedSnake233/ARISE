"""Original official algorithms on new masks, retaining every failure flag."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .common import ROOT,json_dump,sha256
from .fold_data import load_fold
from . import re2_data
from .s3v2_infer import shared_masks,SCENES


def run(dataset,fold,seed,method,output):
    start=dt.datetime.now(dt.timezone.utc).isoformat();clock=time.perf_counter()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    is_re2=dataset in re2_data.DATASETS
    data=re2_data.load_fold(dataset,fold) if is_re2 else load_fold(dataset,fold)
    masks,hashes=shared_masks(data,seed);ids=data.test_ids;paths={};tasks=[]
    if is_re2:
        locations=re2_data.raw_locations(dataset);rows={r['incident_id']:r for r in re2_data.event_rows(dataset)}
        paths={key:locations[key]/'native_metrics.npz' for key in ids}
    else:
        columns=[f'v{s:03d}_c{c:03d}' for s in range(len(data.services)) for c in range(len(data.metadata['metric_names']))]
        column_services=[s for s in data.services for _ in data.metadata['metric_names']]
        for i,(key,pos) in enumerate(zip(ids,data.test_indices)):
            path=output/f'input_{i:03d}.npz';values=data.metrics[pos].transpose(1,0,2).reshape(data.metrics.shape[2],-1)
            np.savez_compressed(path,values=values,times=np.arange(len(values))*data.metadata['step'],columns=columns,column_services=column_services);paths[key]=path
    for scene,mask in zip(SCENES,masks):
        for i,key in enumerate(ids):tasks.append(dict(incident_id=key,scenario=scene,path=str(paths[key]),metric_keep=mask[i,:,0].tolist(),
            inject_time=rows[key]['start']+data.metadata['before'] if is_re2 else data.metadata['before']))
    payload=dict(services=data.services,seed=seed,tasks=tasks,official_dataset={'re2_ob':'online-boutique','re2_ss':'sock-shop','re2_tt':'train-ticket'}.get(dataset,'generic'))
    json_dump(output/'worker_input.json',payload)
    actual='CIRCA' if method=='CIRCA+rank-prune' else method
    worker='exp_revision.rcaeval_rank_worker' if method=='CIRCA+rank-prune' else 'exp_revision.rcaeval_worker'
    runtime=ROOT/'cache/rcaeval_runtime'/('rcd' if method=='RCD' else 'default')/'bin/python'
    if method=='CIRCA+rank-prune':
        assert json.loads((ROOT/'docs/circa_preprocessing_authorization.json').read_text())['approved']
    subprocess.run([str(runtime),'-m',worker,'--method',actual,'--input',str(output/'worker_input.json'),'--output',str(output/'official_audit.json')],
        cwd=ROOT,env={**os.environ,'PYTHONHASHSEED':'42'},check=True)
    records=json.loads((output/'official_audit.json').read_text())
    if [(r['incident_id'],r['scenario']) for r in records]!=[(r['incident_id'],r['scenario']) for r in tasks]:raise ValueError('official output identity mismatch')
    scores=np.asarray([r['scores'] for r in records]).reshape(4,len(ids),len(data.services))
    failed=np.asarray([r['failed'] for r in records]).reshape(4,len(ids))
    if not np.isfinite(scores).all():raise ValueError('nonfinite official score')
    from .s3v2_official_parity import check
    parity_receipt=check(dataset,fold,seed,method,ids,scores,failed)
    parity=parity_receipt['max_abs_difference']
    rates=failed.mean(1);limited=bool((rates>.1).any())
    np.savez_compressed(output/'scores.npz',scores=scores,failed=failed,incident_ids=ids,scenario_keys=SCENES)
    json_dump(output/'run_meta.json',dict(status='limited_s3v2_official' if limited else 'completed_s3v2_official',errors=[],
        config=dict(dataset=dataset,fold=fold,seed=seed,component='official',method=method),start_time=start,
        end_time=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-clock,
        scores_sha256=sha256(output/'scores.npz'),mask_sha256=hashes,s0_max_abs_difference=parity,
        s0_reference=parity_receipt,failure_rates=dict(zip(SCENES,rates.tolist())),scientific_stop=limited,
        input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths.values()},source_sha256=sha256(Path(__file__))))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--method',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(run(a.dataset,a.fold,a.seed,a.method,a.output))
