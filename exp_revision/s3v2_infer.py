"""Checkpoint-only inference on frozen S3v2 masks; no held-out labels or fit."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import joblib
import numpy as np
import torch
from .common import ROOT, json_dump, sha256, stable_seed
from .scenario_v2 import OUTPUT as MASK_ROOT, mask_path
from .missingness_v2 import RATES
from . import baseline_fold as b
from . import re2_data
from .fold_data import load_fold
from .masked_evidence import MaskedEvidence, transform
from .re2_models import Re2Evidence
from .infer_evidence_masks import completed_folder
from .train_context import ContextEncoder
from .train_evidence import predict, fuse_all
from .train_variant_evidence import T3_VARIANTS, T6_VARIANTS, remove_group
from .arise_missing_v2 import EXPERTS, infer_router
from .arise_missing_v3 import infer as infer_rank
from .arise_missing_v4_train import predict_models

SCENES = ['S0_none'] + [f'S3v2_all_{q:.2f}' for q in RATES]
RE2_BASELINES = ('Eadro','MULAN','Eadro+MD','MULAN+MD','Nezha','MicroRCA')


def completed(state, key):
    row = state['jobs'][key]
    if row['status'] != 'completed': raise ValueError('checkpoint incomplete: '+key)
    folder = ROOT/row['output']
    meta = json.loads((folder/'run_meta.json').read_text())
    if sha256(folder/'scores.npz') != meta['scores_sha256']: raise ValueError('reference scores changed')
    return folder


def shared_masks(data, seed):
    manifest = json.loads((MASK_ROOT/'masks_manifest.json').read_text())
    if manifest['datasets'][data.dataset]['services'] != data.services: raise ValueError('service catalogue mismatch')
    masks = [np.ones_like(data.availability[data.test_indices])]; hashes = {}
    for fraction in RATES:
        path=mask_path(MASK_ROOT,data.dataset,fraction,seed)
        expected=manifest['datasets'][data.dataset]['files'][str(path.relative_to(MASK_ROOT))]
        if sha256(path)!=expected: raise ValueError('frozen S3v2 mask changed')
        with np.load(path) as a: k=np.stack([a[key] for key in data.test_ids])
        if k.shape!=masks[0].shape or not np.isin(k,(0,1)).all():raise ValueError('invalid shared mask')
        masks.append(k);hashes[str(path.relative_to(ROOT))]=expected
    return masks,hashes


def reference_s0(folder, ids, method=None, key='scores'):
    with np.load(folder/'scores.npz') as z:
        stored_ids=z['incident_ids'].tolist(); indices=[stored_ids.index(k) for k in ids]
        values=z[key if key in z else 'probabilities'].copy()
        if method is not None: values=values[z['method_names'].tolist().index(method)]
        if values.ndim==3:
            values=values[z['scenario_keys'].tolist().index('S0_none')]
        if values.ndim!=2:raise ValueError('unrecognized reference shape')
        return values[indices]


class FrozenInputs:
    def __init__(self):self.hashes={}
    def path(self,path):
        self.hashes[str(path.relative_to(ROOT))]=sha256(path)
        return path
    def tree(self,path):
        model=joblib.load(self.path(path))
        if hasattr(model,'n_jobs'):model.set_params(n_jobs=1)
        return model
    def checkpoint(self,path):return torch.load(self.path(path),map_location='cpu',weights_only=False)
    def json(self,path):return json.loads(self.path(path).read_text())


def context_infer(data,masks,folder,inputs):
    device='cuda' if data.dataset!='aiops' else 'cpu'
    c=inputs.checkpoint(folder/'model.pt');model=ContextEncoder(**c['model_args']).to(device)
    model.load_state_dict(c['state_dict']);model.eval()
    adjacency=torch.as_tensor(c['adjacency'],device=device)
    arrays=[getattr(data,n)[data.test_indices] for n in ('metrics','traces','logs')]
    batches=[]
    with torch.no_grad():
        for k in masks:
            tensors=[torch.as_tensor(a*k[:,:,m,None,None],dtype=torch.float32,device=device) for a,m in zip(arrays,(0,2,1))]
            chunks=[]
            for idx in np.array_split(np.arange(len(data.test_ids)),max(1,int(np.ceil(len(data.test_ids)/16)))):
                logits,_,_=model(*(a[idx] for a in tensors),adjacency)
                chunks.append(logits.softmax(-1).cpu().numpy())
            batches.append(np.concatenate(chunks))
    return np.stack(batches)


def baseline_infer(data,masks,folder,method,inputs,is_re2):
    meta=inputs.json(folder/'run_meta.json')
    if is_re2:fit,val=re2_data.inner_indices(data)
    elif method.removesuffix('+MD') in ('Eadro','MULAN'):fit,val=b.inner_indices(data)
    else:fit,val=data.train_indices,np.array([],int)
    bundle=b.make_bundle(data,fit,val)
    config=json.loads(b.CONFIG.read_text())['tt' if is_re2 else data.dataset][method]['configuration']
    base=method.removesuffix('+MD')
    device='cuda' if data.dataset!='aiops' and base in ('Eadro','MULAN') else 'cpu'
    if base in ('Eadro','MULAN','TraceAnomaly'):
        c=inputs.checkpoint(folder/'model.pt')
        if c['configuration']!=config:raise ValueError('baseline configuration changed')
        if base=='TraceAnomaly':
            model=b.TraceVAE(data.traces.shape[-1],config['hidden_dim'],config['latent_dim'])
        else:
            widths={name:len(values[0]) for name,values in c['scaler'].items()}
            model=(b.EadroAdapter(widths,config['hidden_dim'],config['attention_heads'],config['dropout'])
                   if base=='Eadro' else b.MulanAdapter(widths,config['hidden_dim']))
            graph=c['adjacency']
        model.load_state_dict(c['state_dict']);model.to(device);model.eval()
    else:
        fitted=inputs.tree(folder/'model.joblib')
        if is_re2:graph=b.training_adjacency(bundle).astype(np.float32)
        else:graph=fitted['adjacency'];fitted=fitted['model']
    result=[]
    for mask in masks:
        features={name:getattr(data,name).copy() for name in b.MODALITIES}
        for name,m in b.MODALITIES.items():features[name][data.test_indices]*=mask[:,:,m,None,None]
        if base in ('Eadro','MULAN'):
            values,_=b.descriptors({n:a[data.test_indices] for n,a in features.items()},bundle.split_bin,c['scaler'])
            with torch.no_grad():
                tensors={n:torch.as_tensor(a,device=device) for n,a in values.items()}
                if base=='Eadro':logits=model(tensors,torch.as_tensor(graph,device=device)[None].expand(len(data.test_ids),-1,-1))
                else:logits,_=model(tensors)
                score=logits.softmax(-1).cpu().numpy()
            if base=='MULAN':
                propagated=np.stack([b.random_walk(graph,row,damping=.85,reverse=True) for row in score])
                score=(1-config['graph_mix'])*b.row_unit_interval(score)+config['graph_mix']*b.row_unit_interval(propagated)
        elif base=='TraceAnomaly':
            post=(features['traces'][data.test_indices,:,bundle.split_bin:]-c['center'])/c['scale']
            flat=torch.as_tensor(post.reshape(-1,post.shape[-1]),dtype=torch.float32)
            with torch.no_grad():
                reconstructed,_,_=model(flat)
                errors=(reconstructed-flat).square().mean(-1).numpy().reshape(post.shape[:3])
            score=(errors.mean(-1) if config['aggregation']=='mean' else errors.max(-1)
                   if config['aggregation']=='max' else np.quantile(errors,.90,axis=-1))
        else:
            masked=copy.copy(bundle);masked.features=features
            score=b.classical_scores(masked,method,config,fitted,graph)[data.test_indices]
        result.append(score)
    return np.stack(result)


def run(dataset,fold,seed,output):
    clock=time.perf_counter();start=dt.datetime.now(dt.timezone.utc).isoformat()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    state=json.loads((ROOT/'results_v2/remaining_queue/queue_state.json').read_text())
    is_re2=dataset in re2_data.DATASETS
    data=re2_data.load_fold(dataset,fold) if is_re2 else load_fold(dataset,fold)
    masks,mask_hashes=shared_masks(data,seed);inputs=FrozenInputs();scores={};parity={};references={}
    def add(method,values,folder=None,reference_method=None,key='scores'):
        values=np.asarray(values)
        if values.shape!=(4,len(data.test_ids),len(data.services)) or not np.isfinite(values).all():raise ValueError('invalid scores '+method)
        if folder is not None:
            inputs.path(folder/'scores.npz');inputs.path(folder/'run_meta.json')
            ref=reference_s0(folder,data.test_ids,reference_method,key)
            difference=float(np.max(np.abs(values[0]-ref)));parity[method]=difference
            references[method]=str(folder.relative_to(ROOT))
            if difference>1e-5:raise ValueError(f'S0 parity failed {method}: {difference}')
        scores[method]=values
        print(json.dumps(dict(method=method,s0_difference=parity.get(method),seconds=time.perf_counter()-clock)),flush=True)
    if is_re2:
        context_folder=completed(state,f'full_models_{dataset}_context_f{fold}_s{seed}')
        evidence_folder=completed(state,f'full_models_{dataset}_evidence_f{fold}_s{seed}')
    else:
        context_folder=completed_folder('results_v2/T1_context_local_full' if dataset=='tt' else 'results_v2/T1_context_full',dataset,fold,seed)
        evidence_folder=completed_folder('results_v2/T1_evidence_full',dataset,fold,seed)
    context=context_infer(data,masks,context_folder,inputs)
    add('context-only',context,context_folder)
    if is_re2:
        view=copy.copy(data);view.train_ids=[]
        view.train_labels=np.asarray([],dtype=int);view.train_groups=[];view.train_fault_types=[]
        for name in (*b.MODALITIES,'availability'):setattr(view,name,getattr(data,name)[data.test_indices])
        e=Re2Evidence(view)
    else:e=MaskedEvidence(dataset,fold)
    cache={}
    def features(j,pipeline):
        key=(j,pipeline)
        if key not in cache:cache[key]=e.recompute(e.data.test_indices,masks[j],pipeline)
        return cache[key]
    specific=inputs.tree(evidence_folder/'specific.joblib');generic=inputs.tree(evidence_folder/'generic.joblib')
    meta=inputs.json(evidence_folder/'run_meta.json')
    beta=meta['beta_selection']['selected_beta'] if is_re2 else meta['selected_beta']
    et_folder=evidence_folder if is_re2 else completed_folder('results_v2/T4_et_spec_full',dataset,fold,seed)
    et=inputs.tree(et_folder/('et.joblib' if is_re2 else 'model.joblib'))
    branches=[];specs=[];ets=[]
    for j in range(4):
        x,n,a=features(j,'family_gated')
        spec=predict(specific,transform(x,n,a,'family_gated'))
        gen=predict(generic,transform(x,n,a,'family_gated',generic=True))
        branches.append(fuse_all((1-beta,spec),(beta,gen)));specs.append(spec)
        ets.append(predict(et,x if is_re2 else transform(x,n,a,'recompute_ungated',append_flags=False)))
    branch=np.stack(branches)
    arise=np.stack([fuse_all((.25,c),(.75,bv)) for c,bv in zip(context,branch)])
    if is_re2:
        add('ARISE-evidence',branch,evidence_folder,'ARISE-evidence')
        add('ARISE',arise)  # both frozen component S0 checks below establish composition parity.
        add('ET-spec',ets,evidence_folder,'ET-spec');add('ET-spec+gate+aug',specs,evidence_folder,'ET-spec+gate+aug')
    else:
        ref=ROOT/f'results_v2/T1_arise_s0/{dataset}/{dataset}_fold{fold}_seed{seed}'
        add('ARISE',arise,ref)
        add('ET-spec',ets,et_folder);add('ET-spec+gate+aug',specs,evidence_folder,key='specific')
    prior=np.bincount(data.train_labels,minlength=len(data.services)).astype(float)
    add('RootFreqPrior',np.broadcast_to(prior,(4,len(data.test_ids),len(data.services))))
    random=[]
    for key in data.test_ids:
        order=np.random.default_rng(stable_seed(dataset,key,'Random',seed)).permutation(len(data.services));row=np.zeros(len(data.services));row[order]=np.arange(len(data.services),0,-1);random.append(row)
    add('Random',np.repeat(np.array(random)[None],4,axis=0))
    for method in RE2_BASELINES if is_re2 else b.METHODS:
        folder=completed(state,f'full_models_{dataset}_{method.replace("+","plus")}_f{fold}_s{seed}' if is_re2 else
                         f'full_baseline_{dataset}_{method.replace("+","_")}_fold{fold}_seed{seed}')
        add(method,baseline_infer(data,masks,folder,method,inputs,is_re2),folder)
    if not is_re2:
        for task,settings in (('T3',T3_VARIANTS),('T6',T6_VARIANTS)):
            folder=completed(state,f'full_{task.lower()}_evidence_{dataset}_fold{fold}_seed{seed}')
            model_meta=inputs.json(folder/'run_meta.json')
            for index,(method,setting) in enumerate(settings.items()):
                # Recovery receipts may refer to split jobs; use their actual recorded model paths.
                paths=[]
                for name in ('specific','generic'):
                    candidates=[ROOT/p for p in model_meta['model_sha256'] if p.endswith(f'variant{index}/{name}.joblib')]
                    path=folder/f'variant{index}/{name}.joblib'
                    if not path.exists():
                        if len(candidates)!=1:raise ValueError('missing/ambiguous recovered variant '+method)
                        path=candidates[0]
                    paths.append(path)
                models=[inputs.tree(p) for p in paths]
                beta=model_meta['beta_selections'][method]['selected_beta'];values=[]
                for j in range(4):
                    x,n,a=features(j,setting['pipeline']);x=remove_group(x,n,setting['drop'])
                    pair=[predict(model,transform(x,n,a,setting['pipeline'],generic=g,append_flags=setting['flags'])) for model,g in zip(models,(False,True))]
                    values.append(fuse_all((.25,context[j]),(.75,fuse_all((1-beta,pair[0]),(beta,pair[1])))))
                add(method,values,folder,method)
        for variant,method in __import__('exp_revision.train_context_variant',fromlist=['VARIANTS']).VARIANTS.items():
            folder=completed(state,f'full_context_variant_{dataset}_{variant}_fold{fold}_seed{seed}')
            changed=context_infer(data,masks,folder,inputs)
            add(method,[fuse_all((.25,c),(.75,bv)) for c,bv in zip(changed,branch)],folder)
        if fold==0 and seed==42:
            folder=completed(state,f'r1_{dataset}_f0_s42')
            models=[inputs.tree(folder/(name+'.joblib')) for name in EXPERTS[:3]]
            router=inputs.json(folder/'router.json');expert=[];availability=[]
            for j in range(4):
                x,n,a=features(j,'edge_gated');availability.append(a)
                spec=transform(x,n,a,'edge_gated');gen=transform(x,n,a,'edge_gated',generic=True)
                expert.append(np.stack([predict(models[0],spec),predict(models[1],spec),predict(models[2],gen),context[j]],axis=1))
            add('ARISE-R1',[infer_router(p,a,router)[0] for p,a in zip(expert,availability)],folder)
            r2=completed(state,f'r2_{dataset}_f0_s42');router2=inputs.json(r2/'router.json')
            add('ARISE-R2',[infer_rank(p,a,router2) for p,a in zip(expert,availability)],r2)
            r3=completed(state,f'r3_{dataset}_f0_s42');routers=inputs.json(r3/'routers.json')
            models=[inputs.tree(r3/(name+'.joblib')) for name in ('clean','standard_specific','standard_generic','block_specific','block_generic')]
            results={k:[] for k in routers}
            for j in range(4):
                x,n,a=features(j,'edge_gated')
                standard,block=predict_models(models,transform(x,n,a,'edge_gated'),transform(x,n,a,'edge_gated',generic=True),context[j])
                for method,router in routers.items():results[method].append(infer_rank(standard if method=='ARISE-R3-no-block' else block,a,router))
            for method,key in [('ARISE-R3','scores'),('ARISE-R3-no-block','no_block_scores'),('ARISE-R3-small-cal','small_cal_scores')]:add(method,results[method],r3,key=key)
    np.savez_compressed(output/'scores.npz',scores=np.stack(list(scores.values())),method_names=list(scores),scenario_keys=SCENES,incident_ids=data.test_ids)
    json_dump(output/'run_meta.json',dict(status='completed_s3v2_frozen_models',errors=[],config=dict(dataset=dataset,fold=fold,seed=seed,component='frozen_models'),
        start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-clock,
        scores_sha256=sha256(output/'scores.npz'),input_sha256={**inputs.hashes,**e.inputs},mask_sha256=mask_hashes,
        s0_max_abs_differences=parity,references=references,training=False,source_sha256=sha256(Path(__file__))))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(run(a.dataset,a.fold,a.seed,a.output))
