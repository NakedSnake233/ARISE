"""Original methods and explicitly named ET controls on pair-held-out TT."""
import argparse
import copy
import json
from pathlib import Path
import joblib
import numpy as np
from .common import ROOT, json_dump, sha256
from . import tt_pair_data as pd
from . import generalization_models as gm
from . import generalization_data as gd
from .s3v2_infer import shared_masks, SCENES
from .train_variant_evidence import VariantTrainer, T3_VARIANTS

CONTROLS=['ET-no-RPC-observed','ET-RPC-only','ET-no-RPC','ET-metrics-only','ET-label-permuted']
ORIGINAL_NEURAL_SCORES=gm.neural_scores


def neural_scores(model,data,features,scaler,adjacency,method,cfg):
    # +MD changes training, not the architecture or test-time graph path.
    return ORIGINAL_NEURAL_SCORES(model,data,features,scaler,adjacency,method.removesuffix('+MD'),cfg)


def pair_permutation(data, seed):
    groups=sorted(set(data.train_groups));mapping={g:np.flatnonzero(np.asarray(data.train_groups)==g) for g in groups}
    values=[]
    for g in groups:
        v=np.unique(data.train_labels[mapping[g]])
        if len(v)!=1 or len(mapping[g])!=3:raise ValueError('permutation requires three same-root repetitions per train pair')
        values.append(v[0])
    permuted=np.random.default_rng(seed).permutation(values);out=data.train_labels.copy()
    for g,label in zip(groups,permuted):out[mapping[g]]=label
    if not np.array_equal(np.sort(out),np.sort(data.train_labels)):raise ValueError('label permutation changed root counts')
    return out


def control_columns(names):
    n=np.arange(len(names));observed={'rpc:server:observed','rpc:exclusive:observed'}
    if not observed.issubset(names):raise ValueError('expected original RPC observed fields')
    return {CONTROLS[0]:np.asarray([i for i,x in enumerate(names) if x not in observed]),
            CONTROLS[1]:np.asarray([i for i,x in enumerate(names) if x.startswith('rpc:')]),
            CONTROLS[2]:np.asarray([i for i,x in enumerate(names) if not x.startswith('rpc:')]),
            CONTROLS[3]:np.asarray([i for i,x in enumerate(names) if x.startswith(('metric:','raw_metric:'))]),
            CONTROLS[4]:n}


def activate(smoke=False):
    pd.activate()
    original_load=gd.load_cohort
    def load(dataset,task,fold,_smoke=False):
        data,folder=original_load(dataset,task,fold,False)
        if smoke:
            data=copy.copy(data);data.test_ids=data.test_ids[:5]
            stop=len(data.train_ids)+len(data.test_ids)
            for key in ('metrics','traces','logs','availability'):setattr(data,key,getattr(data,key)[:stop])
        return data,folder
    gm.load_cohort=load;gm.split_file=pd.split_file;gm.training_file=pd.training_file
    gm.neural_scores=neural_scores
    gm.scenarios=lambda task:list(SCENES)
    def masks(data,task,seed):
        values,hashes=shared_masks(data,seed);return dict(zip(SCENES,values)),hashes
    gm.get_masks=masks


def variants(fold,seed,output):
    start,clock=gm.begin(output);data,folder=gm.load_cohort('tt',pd.TASK,fold)
    evidence=gm.CohortEvidence(data,folder,pd.TASK,seed);trainer=VariantTrainer(evidence,seed)
    masks,hashes=gm.get_masks(data,pd.TASK,seed);all_scores=[];selections={};model_hashes={}
    methods=['ARISE[V-b]','ARISE[V-d]']
    for index,method in enumerate(methods):
        setting=T3_VARIANTS[method];pair,selection=trainer.train_variant(setting);selections[method]=selection
        for name,model in zip(('specific','generic'),pair):
            p=output/f'variant{index}_{name}.joblib';joblib.dump(model,p,compress=3);model_hashes[p.name]=sha256(p)
        values=[]
        for scene in SCENES:
            specific,generic=trainer.score_pair(pair,data.test_indices,masks[scene],setting)
            beta=selection['selected_beta'];values.append(gm.fuse_all((1-beta,specific),(beta,generic)))
        all_scores.append(np.stack(values))
    trainer.save_views(output);scores=np.stack(all_scores)
    np.savez_compressed(output/'scores.npz',scores=scores,method_names=methods,scenario_keys=SCENES,incident_ids=data.test_ids)
    return gm.record(output,data,pd.TASK,seed,False,'variants',scores,start,clock,evidence.inputs,hashes,
        configuration=dict(methods=methods,score_role='evidence branch; final fusion uses matching newly trained context'),beta_selection=selections,model_sha256=model_hashes)


def controls(fold,seed,output):
    start,clock=gm.begin(output);data,folder=gm.load_cohort('tt',pd.TASK,fold)
    evidence=gm.CohortEvidence(data,folder,pd.TASK,seed);columns=control_columns(evidence.names)
    masks,hashes=gm.get_masks(data,pd.TASK,seed);test_x=[]
    for scene in SCENES:
        x,names,a=evidence.recompute(data.test_indices,masks[scene],'family_gated')
        test_x.append(gm.transform(x,names,a,'recompute_ungated',append_flags=False))
    shuffled=pair_permutation(data,seed);all_scores=[];models={}
    for index,name in enumerate(CONTROLS):
        selected=columns[name];model=gm.model_for(None);model.set_params(random_state=seed)
        labels=shuffled if name=='ET-label-permuted' else data.train_labels
        gm.fit(model,evidence.cached[data.train_indices][:,:,selected],labels)
        p=output/f'control{index}.joblib';joblib.dump(model,p,compress=3);models[p.name]=sha256(p)
        all_scores.append(np.stack([gm.predict(model,x[:,:,selected]) for x in test_x]))
    scores=np.stack(all_scores)
    np.savez_compressed(output/'scores.npz',scores=scores,method_names=CONTROLS,scenario_keys=SCENES,incident_ids=data.test_ids)
    json_dump(output/'controls.json',dict(columns={name:[evidence.names[i] for i in indices] for name,indices in columns.items()},
        permuted_training_labels=shuffled.tolist(),training_ids=data.train_ids,training_groups=data.train_groups,
        permutation_unit='whole service/fault pair; same label for its three repeats; original root counts preserved'))
    return gm.record(output,data,pd.TASK,seed,False,'controls',scores,start,clock,evidence.inputs,hashes,
        configuration=dict(methods=CONTROLS),model_sha256=models,control_receipt_sha256=sha256(output/'controls.json'))


def run(component,fold,seed,output,method=None,smoke=False):
    activate(smoke)
    args=('tt',pd.TASK,fold,seed,output)
    if component=='context':gm.train_context(*args,smoke=False)
    elif component=='experts':gm.train_experts(*args,smoke=False)
    elif component=='baseline':gm.train_baseline(*args,method=method,smoke=False)
    elif component=='variants':variants(fold,seed,output)
    elif component=='controls':controls(fold,seed,output)
    else:raise ValueError(component)
    path=output/'run_meta.json';meta=json.loads(path.read_text());meta['config']['smoke']=smoke
    meta['source_sha256'].update({str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'exp_revision/tt_pair_data.py']})
    meta['protocol_note']='pair-disjoint outer and inner cohorts; outer-training preprocessing fixed during inner model selection; original settings'
    meta['evaluation_scope']='first5testevents_smoke' if smoke else 'complete_outer_test_fold'
    json_dump(path,meta);return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--component',required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method');p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(run(a.component,a.fold,a.seed,a.output,a.method,a.smoke))
