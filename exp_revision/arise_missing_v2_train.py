"""Train ARISE-R1 using outer-training labels only; output frozen test scores."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .common import ROOT, json_dump, sha256, config_snapshot, stable_seed
from .arise_missing_v2 import EXPERTS, fit_router, infer_router, probabilities

SCENES = ['S0_none','S1_metric','S1_log','S1_trace','S2_0.10','S2_0.25','S2_0.50','S2_0.75','S3_trace','S3_all','S3c_trace','S3c_all']
VIEW_NAMES = ['complete','no_metric','no_log','no_trace','random_010','random_025','random_050','random_075','random_service_trace','random_service_all']


def inner_split(data):
    fit, held = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(data.train_indices, groups=data.train_groups))
    groups = np.asarray(data.train_groups)
    if set(groups[fit]) & set(groups[held]) or not len(fit) or not len(held): raise ValueError('invalid inner groups')
    return fit, held


def context_split(data, fit):
    """Checkpoint validation is nested inside fit, excluding router calibration."""
    if data.dataset == 'aiops': return np.asarray(fit), np.asarray([], dtype=int)
    groups=np.asarray(data.train_groups)
    train,val=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(fit,groups=groups[fit]))
    train,val=np.asarray(fit)[train],np.asarray(fit)[val]
    if set(groups[train])&set(groups[val]):raise ValueError('context checkpoint groups overlap')
    return train,val


def training_views(a0, ids, seed, stream):
    """Label-free masks. Restore only from actually observed modalities."""
    a0=np.asarray(a0,dtype=np.uint8)
    if a0.ndim!=3 or a0.shape[-1]!=3 or len(ids)!=len(a0) or not np.isin(a0,(0,1)).all(): raise ValueError('invalid availability')
    views=[a0.copy()]
    for m in range(3):
        mask=a0.copy();mask[:,:,m]=0;views.append(mask)
    for view,p in enumerate((.10,.25,.50,.75)):
        mask=np.zeros_like(a0)
        for i,key in enumerate(ids):
            rng=np.random.default_rng(stable_seed('ARISE-R1',seed,stream,key,'cells',view))
            mask[i]=a0[i]*(rng.random(a0[i].shape)>=p)
            for service in np.flatnonzero((mask[i].sum(1)==0)&(a0[i].sum(1)>0)):
                mask[i,service,rng.choice(np.flatnonzero(a0[i,service]))]=1
        views.append(mask)
    for scope in ('trace','all'):
        mask=a0.copy()
        for i,key in enumerate(ids):
            rng=np.random.default_rng(stable_seed('ARISE-R1',seed,stream,key,'service',scope))
            eligible=np.flatnonzero(a0[i,:,2]>0) if scope=='trace' else np.flatnonzero(a0[i].sum(1)>0)
            if len(eligible):
                service=rng.choice(eligible)
                if scope=='trace':mask[i,service,2]=0
                else:mask[i,service,:]=0
        views.append(mask)
    return views


def materialize(evidence,indices,seed,stream):
    from .masked_evidence import transform
    d=evidence.data;ids=[d.train_ids[i] for i in indices]
    if len(indices)==0 or min(indices)<0 or max(indices)>=len(d.train_ids): raise ValueError('training-only cohort required')
    masks=training_views(d.availability[indices],ids,seed,stream)
    spec=[];generic=[]
    for name,mask in zip(VIEW_NAMES,masks):
        x,names,a=evidence.recompute(indices,mask,'edge_gated')
        spec.append(transform(x,names,a,'edge_gated'))
        generic.append(transform(x,names,a,'edge_gated',generic=True))
        print(json.dumps(dict(stage='features',stream=stream,view=name,events=len(indices))),flush=True)
    return spec,generic,masks


def fit_experts(material,labels,indices,seed):
    from .train_evidence import fit,model_for
    spec,generic,_=material;models=[]
    for values,aug in ((spec,False),(spec,True),(generic,True)):
        x=np.concatenate([v[indices] for v in values]) if aug else values[0][indices]
        y=np.tile(labels[indices],len(values)) if aug else labels[indices]
        model=model_for(None);model.set_params(random_state=seed,n_jobs=4)
        models.append(fit(model,x,y))
    return models


def predict_experts(models,specific,generic,context):
    from .train_evidence import predict
    return np.stack([predict(models[0],specific),predict(models[1],specific),predict(models[2],generic),context],axis=1)


def final_context(data,seed):
    from . import re2_data
    if data.dataset in re2_data.DATASETS:
        from . import remaining_queue as q
        state=json.loads((q.FOLDER/'queue_state.json').read_text())
        row=state['jobs'][f'full_models_{data.dataset}_context_f{data.fold}_s{seed}']
        if row['status']!='completed':raise ValueError('final context prerequisite pending')
        folder=ROOT/row['output']
        if not q.verified(folder,row['spec']):raise ValueError('context failed verification')
        with np.load(folder/'scores.npz') as z:
            assert z['incident_ids'].tolist()==data.test_ids and z['scenario_keys'].tolist()==SCENES
            values=z['scores'].copy()
    else:
        from .infer_evidence_masks import context_scores
        values,folder=context_scores(data.dataset,data.fold,seed,data.test_ids)
    return values,folder


def train(dataset,fold,seed,output):
    start=dt.datetime.now(dt.timezone.utc).isoformat();clock=time.perf_counter()
    if output.exists() and any(output.iterdir()):raise FileExistsError('immutable output occupied')
    output.mkdir(parents=True,exist_ok=True)
    from .masked_evidence import MaskedEvidence,transform
    from . import re2_data
    if dataset in re2_data.DATASETS:
        from .re2_models import Re2Evidence
        evidence=Re2Evidence(re2_data.load_fold(dataset,fold))
    else:evidence=MaskedEvidence(dataset,fold)
    d=evidence.data
    # Check final context prerequisite before any fitting. It is never used for inner calibration.
    context,context_folder=final_context(d,seed)
    fit,held=inner_split(d)
    split=dict(fit_ids=[d.train_ids[i] for i in fit],calibration_ids=[d.train_ids[i] for i in held],
        fit_groups=sorted(set(d.train_groups[i] for i in fit),key=str),calibration_groups=sorted(set(d.train_groups[i] for i in held),key=str))
    json_dump(output/'inner_split.json',split)
    material=materialize(evidence,d.train_indices,seed,'fit')
    calibration=materialize(evidence,held,seed,'calibration')
    models=fit_experts(material,d.train_labels,fit,seed)
    # Separate context model sees only fit labels; original test-context model is not used here.
    from .train_context_variant import fit_context
    profile=copy.copy(d)
    if dataset in re2_data.DATASETS:profile.dataset='tt'
    directory=output/'inner_context';directory.mkdir()
    context_fit,context_validation=context_split(d,fit)
    context_infer,context_training=fit_context(profile,context_fit,context_validation,seed,'full','cpu',directory)
    split.update(context_fit_ids=[d.train_ids[i] for i in context_fit],context_checkpoint_ids=[d.train_ids[i] for i in context_validation])
    json_dump(output/'inner_split.json',split)
    cal_scores=[]
    for spec,gen,mask in zip(*calibration):
        cal_scores.append(predict_experts(models,spec,gen,context_infer(held,mask)))
    cal_scores=np.concatenate(cal_scores);cal_a=np.concatenate(calibration[2])
    y=np.tile(d.train_labels[held],len(VIEW_NAMES));ids=[d.train_ids[i] for _ in VIEW_NAMES for i in held]
    router=fit_router(cal_scores,cal_a,y,ids)
    router.update(training_only=True,outer_test_used=False,inner_context_training=context_training)
    json_dump(output/'router.json',router)
    np.savez_compressed(output/'calibration.npz',scores=cal_scores,availability=cal_a,labels=y,event_ids=ids,views=VIEW_NAMES)
    np.savez_compressed(output/'training_masks.npz',fit=np.stack(material[2]),calibration=np.stack(calibration[2]))
    # Router is frozen before refitting or any held-out prediction.
    models=fit_experts(material,d.train_labels,d.train_indices,seed)
    for name,model in zip(EXPERTS[:3],models):joblib.dump(model,output/(name+'.joblib'),compress=3)
    del material,calibration,context_infer
    batches=[];experts=[];weights=[];hashes={}
    for j,scene in enumerate(SCENES):
        path=ROOT/f'masks/{dataset}/{scene}/seed{seed}.npz';hashes[str(path.relative_to(ROOT))]=sha256(path)
        with np.load(path) as z:mask=np.stack([z[k] for k in d.test_ids])
        x,names,a=evidence.recompute(d.test_indices,mask,'edge_gated')
        scores=predict_experts(models,transform(x,names,a,'edge_gated'),transform(x,names,a,'edge_gated',generic=True),context[j])
        final,w=infer_router(scores,a,router);batches.append(final);experts.append(scores);weights.append(w)
        print(json.dumps(dict(stage='test_prediction',scenario=scene,elapsed_seconds=time.perf_counter()-clock)),flush=True)
    np.savez_compressed(output/'scores.npz',scores=np.stack(batches),expert_scores=np.stack(experts),mixture_weights=np.stack(weights),
        incident_ids=d.test_ids,scenario_keys=SCENES,expert_names=EXPERTS)
    cfg=dict(task='R1',component='arise_missing_v2',method='ARISE-R1',dataset=dataset,fold=fold,seed=seed,smoke=False)
    meta=config_snapshot('python -m exp_revision.arise_missing_v2_train',cfg)
    sources=['exp_revision/arise_missing_v2.py','exp_revision/arise_missing_v2_train.py','exp_revision/masked_evidence.py','exp_revision/train_context_variant.py',
        'scripts/care/robust_evidence.py','scripts/care/evidence_reranker.py','configs/care/review-response-v1.json']
    inputs=dict(evidence.inputs)
    for path in [context_folder/'scores.npz',context_folder/'run_meta.json',ROOT/f'splits/{dataset}_training_labels/fold_{fold}.json']:
        inputs[str(path.relative_to(ROOT))]=sha256(path)
    meta.update(status='completed_arise_missing_v2',errors=[],start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-clock,scores_sha256=sha256(output/'scores.npz'),input_sha256=inputs,mask_sha256=hashes,
        source_sha256={n:sha256(ROOT/n) for n in sources},router_sha256=sha256(output/'router.json'),
        evaluation_scope='development on previously inspected benchmark; independent confirmation outstanding',
        context_refit='frozen original full-fold context predictions; inner calibration context fitted separately on disjoint training groups')
    json_dump(output/'run_meta.json',meta)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['tt','aiops','re2_ob','re2_ss','re2_tt'],required=True)
    p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(train(a.dataset,a.fold,a.seed,a.output))


if __name__=='__main__':main()
