"""ARISE-R3: full training-group OOF calibration plus matched augmentation controls."""
import argparse
import datetime as dt
import json
import time
from pathlib import Path
import joblib
import numpy as np
from .common import ROOT,json_dump,sha256,config_snapshot
from .arise_missing_v4 import masks,crossfit_plan,expert_view_indices,VIEW_NAMES
from .arise_missing_v2_train import SCENES,inner_split,final_context
from .arise_missing_v3 import fit_rank,infer


def materialize(evidence,seed,stream):
    from .masked_evidence import transform
    d=evidence.data;indices=d.train_indices;views=masks(d.availability[indices],d.train_ids,seed,stream)
    specific=[];generic=[]
    for name,mask in zip(VIEW_NAMES,views):
        x,names,a=evidence.recompute(indices,mask,'edge_gated')
        specific.append(transform(x,names,a,'edge_gated'));generic.append(transform(x,names,a,'edge_gated',generic=True))
        print(json.dumps(dict(stage='features',stream=stream,view=name)),flush=True)
    return specific,generic,views


def fit_models(material,labels,fit_indices,seed):
    from .train_evidence import model_for,fit
    specific,generic,_=material;models=[]
    for features,kind in [(specific,'clean'),(specific,'standard'),(generic,'standard'),(specific,'block'),(generic,'block')]:
        views=expert_view_indices(kind)
        x=np.concatenate([features[i][fit_indices] for i in views]);y=np.tile(labels[fit_indices],len(views))
        model=model_for(None);model.set_params(random_state=seed,n_jobs=4)
        models.append(fit(model,x,y))
    return models


def predict_models(models,specific,generic,context):
    from .train_evidence import predict
    clean=predict(models[0],specific)
    standard=np.stack([clean,predict(models[1],specific),predict(models[2],generic),context],axis=1)
    block=np.stack([clean,predict(models[3],specific),predict(models[4],generic),context],axis=1)
    return standard,block


def train(ds,fold,seed,output):
    clock=time.perf_counter();start=dt.datetime.now(dt.timezone.utc).isoformat()
    if output.exists() and any(output.iterdir()):raise FileExistsError('immutable output occupied')
    output.mkdir(parents=True,exist_ok=True)
    from .masked_evidence import MaskedEvidence,transform
    from .train_context_variant import fit_context
    e=MaskedEvidence(ds,fold);d=e.data;plan=crossfit_plan(d)
    receipts=[]
    for index,part in enumerate(plan):
        receipt={'part':index}
        for name,indices in part.items():
            receipt[name+'_ids']=[d.train_ids[i] for i in indices]
            receipt[name+'_groups']=sorted(set(str(d.train_groups[i]) for i in indices))
        receipts.append(receipt)
    json_dump(output/'crossfit_plan.json',receipts)
    fit_material=materialize(e,seed,'fit');cal=materialize(e,seed,'calibration')
    shape=(len(VIEW_NAMES),len(d.train_ids),4,len(d.services))
    standard=np.zeros(shape);block=np.zeros(shape);coverage=np.zeros(len(d.train_ids),dtype=int)
    for index,part in enumerate(plan):
        models=fit_models(fit_material,d.train_labels,part['fit'],seed)
        ctxdir=output/f'context_part{index}';ctxdir.mkdir()
        ctx,ctxmeta=fit_context(d,part['context_fit'],part['context_validation'],seed,'full','cpu',ctxdir)
        receipts[index]['context_training']=ctxmeta
        held=part['held']
        for view,(specific,generic,mask) in enumerate(zip(*cal)):
            context=ctx(held,mask[held])
            standard[view,held],block[view,held]=predict_models(models,specific[held],generic[held],context)
        coverage[held]+=1
        del models,ctx
        print(json.dumps(dict(stage='crossfit_completed',part=index,held_events=len(held),elapsed_seconds=time.perf_counter()-clock)),flush=True)
    if not np.all(coverage==1):raise ValueError('OOF coverage invalid')
    json_dump(output/'crossfit_plan.json',receipts)
    ids=np.tile(np.asarray(d.train_ids),len(VIEW_NAMES));views=np.repeat(VIEW_NAMES,len(d.train_ids));y=np.tile(d.train_labels,len(VIEW_NAMES));a=np.concatenate(cal[2])
    p_std=standard.reshape(-1,4,len(d.services));p_block=block.reshape(-1,4,len(d.services))
    _,held_small=inner_split(d);small=np.isin(ids,np.asarray(d.train_ids)[held_small])
    routers={'ARISE-R3':fit_rank(p_block,a,y,ids,views,True),
        'ARISE-R3-no-block':fit_rank(p_std,a,y,ids,views,True),
        'ARISE-R3-small-cal':fit_rank(p_block[small],a[small],y[small],ids[small],views[small],True)}
    json_dump(output/'routers.json',routers);frozen=sha256(output/'routers.json')
    np.savez_compressed(output/'training_oof.npz',standard=standard,block=block,availability=np.stack(cal[2]),labels=d.train_labels,event_ids=d.train_ids,view_names=VIEW_NAMES,coverage=coverage,small_calibration_indices=held_small)
    np.savez_compressed(output/'training_masks.npz',fit=np.stack(fit_material[2]),calibration=np.stack(cal[2]))
    print(json.dumps(dict(stage='routers_frozen',events=len(d.train_ids),small_events=len(held_small))),flush=True)
    models=fit_models(fit_material,d.train_labels,d.train_indices,seed)
    for name,model in zip(['clean','standard_specific','standard_generic','block_specific','block_generic'],models):joblib.dump(model,output/(name+'.joblib'),compress=3)
    del fit_material,cal
    # Decode final context test arrays only after all three routers are frozen.
    context,context_folder=final_context(d,seed)
    scores={k:[] for k in routers};mask_hashes={}
    for index,scene in enumerate(SCENES):
        path=ROOT/f'masks/{ds}/{scene}/seed{seed}.npz';mask_hashes[str(path.relative_to(ROOT))]=sha256(path)
        with np.load(path) as z:mask=np.stack([z[k] for k in d.test_ids])
        x,names,availability=e.recompute(d.test_indices,mask,'edge_gated')
        standard_test,block_test=predict_models(models,transform(x,names,availability,'edge_gated'),transform(x,names,availability,'edge_gated',generic=True),context[index])
        for method,router in routers.items():scores[method].append(infer(standard_test if method=='ARISE-R3-no-block' else block_test,availability,router))
        print(json.dumps(dict(stage='test_prediction',scenario=scene,elapsed_seconds=time.perf_counter()-clock)),flush=True)
    if sha256(output/'routers.json')!=frozen:raise ValueError('router mutated')
    np.savez_compressed(output/'scores.npz',scores=np.stack(scores['ARISE-R3']),no_block_scores=np.stack(scores['ARISE-R3-no-block']),small_cal_scores=np.stack(scores['ARISE-R3-small-cal']),incident_ids=d.test_ids,scenario_keys=SCENES)
    cfg=dict(task='R3',component='arise_missing_v4',method='ARISE-R3',dataset=ds,fold=fold,seed=seed,smoke=False)
    meta=config_snapshot('python -m exp_revision.arise_missing_v4_train',cfg)
    inputs=dict(e.inputs)
    for path in [ROOT/f'splits/{ds}_training_labels/fold_{fold}.json',context_folder/'scores.npz',context_folder/'run_meta.json']:inputs[str(path.relative_to(ROOT))]=sha256(path)
    meta.update(status='completed_arise_missing_v4',errors=[],start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-clock,
        scores_sha256=sha256(output/'scores.npz'),router_sha256=frozen,input_sha256=inputs,mask_sha256=mask_hashes,
        source_sha256={n:sha256(ROOT/n) for n in ['exp_revision/arise_missing_v4.py','exp_revision/arise_missing_v4_train.py','exp_revision/arise_missing_v3.py']},
        evaluation_scope='inspected benchmark development only; supervised OOF conditional on outer-training preprocessing',
        oof_events=len(d.train_ids),crossfit_parts=3,small_calibration_events=len(held_small),
        context_refit='original frozen final context; inner models group-isolated, final refit approximation disclosed')
    json_dump(output/'run_meta.json',meta)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['tt','aiops'],required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(train(a.dataset,a.fold,a.seed,a.output))
