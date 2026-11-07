"""Run frozen ARISE and baseline functions on supplied, isolated RE2 folds."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
from functools import lru_cache
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from .common import ROOT, config_snapshot, json_dump, sha256, stable_seed
from . import re2_data as data_module
from .baseline_fold import (MODALITIES,make_bundle,fit_neural,descriptors,fit_classical,
                            classical_scores, row_unit_interval,random_walk,training_adjacency)
from .generalization_models import CachedTrainer
from .masked_evidence import MaskedEvidence, transform, extract
from .missingness import SCENARIOS
from .raw_spans_v2 import rpc_features
from .train_context_variant import fit_context
from .train_evidence import fit, model_for, predict, fuse_all

SCENE_KEYS=[f"{s}_{p or 'none'}" for s,p in SCENARIOS]
EVIDENCE_METHODS=["ARISE-evidence","ET-spec","ET-spec+gate+aug","RootFreqPrior","Random"]


class Re2Evidence(MaskedEvidence):
    def __init__(self,data):
        self.data=data; self.ids=data.train_ids+data.test_ids
        self.rows={r["incident_id"]:r for r in data_module.event_rows(data.dataset)}
        self.locations=data_module.raw_locations(data.dataset); self.inputs={}
        raw=[]
        catalogue=json.loads(data_module.CATALOG.read_text())[data.dataset]["metric_names"]
        keep=[catalogue.index(c) for c in data.metadata["metric_names"]]
        for key in self.ids:
            with np.load(self.locations[key]/"raw.npz") as a: raw.append(a["metrics"][:,:,keep])
        self.raw_metrics=np.stack(raw)
        rows=[dict(incident_id=k,calls=data.metadata["calls"][k]) for k in self.ids]
        x,names=extract({name:getattr(data,name) for name in MODALITIES},data.metadata,rows,self.raw_metrics)
        rpc=[]
        for i,key in enumerate(self.ids):
            values,rpc_names=self.rpc(key,np.asarray(data.availability[i,:,2],dtype=np.uint8).tobytes());rpc.append(values)
        self.cached=np.concatenate([x,np.stack(rpc)],axis=-1).astype(np.float32); self.names=names+rpc_names

    @lru_cache(maxsize=8)
    def table(self,key):
        path=self.locations[key]/"spans.npz"; self.inputs[str(path.relative_to(ROOT))]=sha256(path)
        with np.load(path) as a: return {n:a[n].copy() for n in a.files}

    @lru_cache(maxsize=16000)
    def rpc(self,key,mask_bytes):
        return rpc_features(self.table(key),self.rows[key],self.data.services,self.data.metadata["before"],np.frombuffer(mask_bytes,dtype=np.uint8))


def masks(data,seed,ids):
    values=[]; hashes={}
    for scene in SCENE_KEYS:
        path=ROOT / f"masks/{data.dataset}/{scene}/seed{seed}.npz"
        with np.load(path) as a: values.append(np.stack([a[k] for k in ids]))
        hashes[str(path.relative_to(ROOT))]=sha256(path)
    return values,hashes


def begin(dataset,fold,seed,smoke,component,output):
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    data=data_module.load_fold(dataset,fold)
    ids=data.test_ids[:3] if smoke else data.test_ids
    indices=data.test_indices[:len(ids)]
    meta=config_snapshot("python -m exp_revision.re2_models",dict(task="T8",dataset=dataset,fold=fold,seed=seed,
        smoke=smoke,component=component,scenarios=SCENE_KEYS,model_profile="frozen TrainTicket",alpha=.75))
    return data,ids,indices,meta,dt.datetime.now(dt.timezone.utc).isoformat(),time.perf_counter()


def record(output,data,meta,start,clock,hashes,**extra):
    inputs=[data_module.fold_folder(data.dataset,data.fold)/"features.npz", data_module.fold_folder(data.dataset,data.fold)/"metadata.json",
            ROOT/f"splits/{data.dataset}_training_labels/fold_{data.fold}.json",ROOT/f"splits/{data.dataset}_inner_validation.json"]
    meta.update(status="completed_re2_scores",errors=[],start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-clock,mask_sha256=hashes,scores_sha256=sha256(output/"scores.npz"),
        input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in inputs},**extra)
    json_dump(output/"run_meta.json",meta)


def context(dataset,fold,seed,output,smoke=False):
    data,ids,indices,meta,start,clock=begin(dataset,fold,seed,smoke,"context",output)
    fit_ids,validation=data_module.inner_indices(data)
    profile=copy.copy(data); profile.dataset="tt"
    # Only configuration dispatch uses the TT profile; the supplied tensors,
    # labels, fit/validation IDs and incident graph are the actual RE2 fold.
    infer,training=fit_context(profile,fit_ids,validation,seed,"full","cuda",output)
    scene_masks,hashes=masks(data,seed,ids)
    values=np.stack([infer(indices,k) for k in scene_masks])
    if not np.isfinite(values).all(): raise ValueError("nonfinite context scores")
    np.savez_compressed(output/"scores.npz",scores=values,probabilities=values,incident_ids=ids,scenario_keys=SCENE_KEYS)
    record(output,data,meta,start,clock,hashes,training=training,model_sha256=sha256(output/"model.pt"))


def evidence(dataset,fold,seed,output,smoke=False):
    data,ids,indices,meta,start,clock=begin(dataset,fold,seed,smoke,"evidence",output)
    e=Re2Evidence(data); trainer=CachedTrainer(e,seed)
    setting=dict(augment=True,pipeline="family_gated",flags=True,drop=None)
    models,selection=trainer.train_variant(setting)
    for name,model in zip(("specific","generic"),models): joblib.dump(model,output/f"{name}.joblib",compress=3)
    trainer.save_views(output); json_dump(output/"inner_beta.json",selection)
    et=model_for(None); et.set_params(random_state=seed,n_estimators=256,min_samples_leaf=3,max_features=.7,class_weight="balanced",n_jobs=4)
    fit(et,e.cached[data.train_indices],data.train_labels); joblib.dump(et,output/"et.joblib",compress=3)
    prior=np.bincount(data.train_labels,minlength=len(data.services)).astype(float)
    random_scores=[]
    for key in ids:
        order=np.random.default_rng(stable_seed(dataset,key,"Random",seed)).permutation(len(data.services))
        row=np.zeros(len(data.services));row[order]=np.arange(len(data.services),0,-1);random_scores.append(row)
    scene_masks,hashes=masks(data,seed,ids); values=[]
    for k in scene_masks:
        x,names,a=e.recompute(indices,k,"family_gated")
        spec=predict(models[0],transform(x,names,a,"family_gated"))
        gen=predict(models[1],transform(x,names,a,"family_gated",generic=True))
        branch=fuse_all((1-selection["selected_beta"],spec),(selection["selected_beta"],gen))
        values.append(np.stack([branch,predict(et,x),spec,np.repeat(prior[None],len(ids),axis=0),np.stack(random_scores)]))
    scores=np.stack(values,axis=1)
    if not np.isfinite(scores).all(): raise ValueError("nonfinite evidence scores")
    np.savez_compressed(output/"scores.npz",scores=scores,method_names=EVIDENCE_METHODS,incident_ids=ids,scenario_keys=SCENE_KEYS)
    record(output,data,meta,start,clock,hashes,beta_selection=selection,span_input_sha256=e.inputs,
        model_sha256={name:sha256(output/name) for name in ("specific.joblib","generic.joblib","et.joblib")})


def baseline(dataset,fold,seed,method,output,smoke=False):
    data,ids,indices,meta,start,clock=begin(dataset,fold,seed,smoke,method,output)
    fit_ids,validation=data_module.inner_indices(data); bundle=make_bundle(data,fit_ids,validation)
    config=json.loads((ROOT/"exp_revision/baseline_settings.json").read_text())["tt"][method]["configuration"]
    graph=training_adjacency(bundle).astype(np.float32)
    if method.removesuffix("+MD") in ("Eadro","MULAN"):
        model,scaler,graph,history=fit_neural(data,bundle,method,config,seed,"cuda")
        torch.save(dict(state_dict=model.state_dict(),configuration=config,scaler=scaler,adjacency=graph),output/"model.pt")
    else:
        model=fit_classical(bundle,method,config); joblib.dump(model,output/"model.joblib",compress=3);history=[]
    scene_masks,hashes=masks(data,seed,ids); values=[]
    for k in scene_masks:
        all_masks=np.ones_like(data.availability); all_masks[indices]=k
        features={name:getattr(data,name)*all_masks[:,:,m,None,None] for name,m in MODALITIES.items()}
        if method.removesuffix("+MD") in ("Eadro","MULAN"):
            prepared,_=descriptors({name:a[indices] for name,a in features.items()},bundle.split_bin,scaler)
            inputs={name:torch.as_tensor(a,device="cuda") for name,a in prepared.items()}
            with torch.no_grad():
                if method.removesuffix("+MD")=="Eadro": logits=model(inputs,torch.as_tensor(graph,device="cuda")[None].expand(len(ids),-1,-1))
                else: logits,_=model(inputs)
                score=logits.softmax(-1).cpu().numpy()
            if method.removesuffix("+MD")=="MULAN":
                propagated=np.stack([random_walk(graph,row,damping=.85,reverse=True) for row in score])
                score=(1-config["graph_mix"])*row_unit_interval(score)+config["graph_mix"]*row_unit_interval(propagated)
        else:
            masked=copy.copy(bundle);masked.features=features
            score=classical_scores(masked,method,config,model,graph)[indices]
        values.append(score)
    scores=np.stack(values)
    if not np.isfinite(scores).all():raise ValueError("nonfinite baseline scores")
    np.savez_compressed(output/"scores.npz",scores=scores,incident_ids=ids,scenario_keys=SCENE_KEYS)
    record(output,data,meta,start,clock,hashes,configuration=config,training_history=history)


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=data_module.DATASETS,required=True);p.add_argument("--fold",type=int,required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--component",required=True);p.add_argument("--smoke",action="store_true");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.component=="context":context(a.dataset,a.fold,a.seed,a.output,a.smoke)
    elif a.component=="evidence":evidence(a.dataset,a.fold,a.seed,a.output,a.smoke)
    else:baseline(a.dataset,a.fold,a.seed,a.component,a.output,a.smoke)


if __name__=="__main__":main()
