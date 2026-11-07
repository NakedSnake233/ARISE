"""Cheap R2 selection and inference, reusing frozen supervised R1 experts."""
import argparse
import json
import time
import datetime as dt
import numpy as np
from .common import ROOT,json_dump,sha256,config_snapshot
from . import remaining_queue as q
from .arise_missing_v3 import select_and_fit,infer


def run(ds,fold,seed,output):
    clock=time.perf_counter();start=dt.datetime.now(dt.timezone.utc).isoformat()
    if output.exists() and any(output.iterdir()):raise FileExistsError('occupied immutable output')
    output.mkdir(parents=True,exist_ok=True)
    state=json.loads((q.FOLDER/'queue_state.json').read_text());row=state['jobs'][f'r1_{ds}_f{fold}_s{seed}']
    folder=ROOT/row['output']
    if row['status']!='completed':raise ValueError('R1 prerequisite incomplete')
    prior_meta=json.loads((folder/'run_meta.json').read_text())
    if prior_meta['config']!=row['spec']['expected_config'] or prior_meta['scores_sha256']!=sha256(folder/'scores.npz'):raise ValueError('R1 prerequisite invalid')
    metadata_path=ROOT/f'cache/fold_preprocessing/{ds}/fold{fold}/metadata.json'
    metadata=json.loads(metadata_path.read_text());services=metadata['services']
    train_path=ROOT/f'splits/{ds}_training_labels/fold_{fold}.json'
    train={r['incident_id']:r for r in json.loads(train_path.read_text())}
    with np.load(folder/'calibration.npz') as z:
        p=z['scores'];a=z['availability'];y=z['labels'];ids=z['event_ids'];view_names=z['views']
    split=json.loads((folder/'inner_split.json').read_text());held=split['calibration_ids']
    if ids.tolist()!=held*len(view_names):raise ValueError('calibration row order mismatch')
    if not np.array_equal(y,[services.index(train[k]['root_service']) for k in ids]):raise ValueError('training labels mismatch')
    groups=np.array([str(train[k]['overlap_group']) for k in ids]);views=np.repeat(view_names,len(held))
    router,selection,oof=select_and_fit(p,a,y,ids,views,groups)
    json_dump(output/'router.json',router);json_dump(output/'selection.json',selection)
    np.savez_compressed(output/'selection_oof.npz',**oof,labels=y,ids=ids,views=views,groups=groups)
    frozen_router=sha256(output/'router.json')
    print(json.dumps({'stage':'router_frozen','selected':selection['selected'],'eligible':selection['eligible'],'candidates':selection['candidates']}),flush=True)
    # Outer scores decoded only after selection and router freeze. Preflight checks bytes only.
    if not q.verified(folder,row['spec']):raise ValueError('R1 prerequisite shape invalid')
    with np.load(folder/'scores.npz') as z:experts=z['expert_scores'];test_ids=z['incident_ids'].tolist();scenes=z['scenario_keys'].tolist()
    if test_ids!=metadata['test_ids']:raise ValueError('test identity mismatch')
    feature_path=ROOT/f'cache/fold_preprocessing/{ds}/fold{fold}/features.npz'
    with np.load(feature_path) as z:a0=z['availability'][len(metadata['train_ids']):]
    batches=[];inputs={str(path.relative_to(ROOT)):sha256(path) for path in [metadata_path,train_path,feature_path,folder/'scores.npz',folder/'run_meta.json',folder/'calibration.npz',folder/'inner_split.json']}
    for scene,pred in zip(scenes,experts):
        path=ROOT/f'masks/{ds}/{scene}/seed{seed}.npz';inputs[str(path.relative_to(ROOT))]=sha256(path)
        with np.load(path) as z:availability=a0*np.stack([z[k] for k in test_ids])
        batches.append(infer(pred,availability,router))
    if sha256(output/'router.json')!=frozen_router:raise ValueError('router changed after freeze')
    np.savez_compressed(output/'scores.npz',scores=np.stack(batches),incident_ids=test_ids,scenario_keys=scenes)
    cfg=dict(task='R2',component='arise_missing_v3',method='ARISE-R2',dataset=ds,fold=fold,seed=seed,smoke=False)
    meta=config_snapshot('python -m exp_revision.arise_missing_v3_run',cfg)
    meta.update(status='completed_arise_missing_v3',errors=[],start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-clock,
        scores_sha256=sha256(output/'scores.npz'),router_sha256=frozen_router,input_sha256=inputs,
        source_sha256={n:sha256(ROOT/n) for n in ['exp_revision/arise_missing_v3.py','exp_revision/arise_missing_v3_run.py']},
        evaluation_scope='inspected benchmark development only',selected_candidate=selection['selected'])
    json_dump(output/'run_meta.json',meta)
    return output

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['tt','aiops'],required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(run(a.dataset,a.fold,a.seed,a.output))
