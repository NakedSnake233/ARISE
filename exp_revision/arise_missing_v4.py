"""Cross-fit coverage and label-free whole-service outage augmentation."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from .common import stable_seed
from .arise_missing_v2_train import training_views,context_split,VIEW_NAMES as OLD_VIEWS

VIEW_NAMES=OLD_VIEWS+['service_block_025','service_block_050']


def masks(a0,ids,seed,stream):
    a0=np.asarray(a0)
    if a0.ndim!=3 or a0.shape[-1]!=3 or not np.isin(a0,(0,1)).all():raise ValueError("invalid original availability")
    a0=a0.astype(np.uint8)
    values=training_views(a0,ids,seed,stream)
    for rate in (.25,.5):
        mask=a0.copy()
        for i,key in enumerate(ids):
            observed=np.flatnonzero(a0[i].sum(1)>0)
            if not len(observed):continue
            rng=np.random.default_rng(stable_seed('ARISE-R3',seed,stream,key,'service_block',rate))
            selected=rng.choice(observed,size=int(np.ceil(rate*len(observed))),replace=False)
            mask[i,selected]=0
        values.append(mask)
    return values


def crossfit_plan(data):
    groups=np.asarray(data.train_groups);indices=np.asarray(data.train_indices)
    if not np.array_equal(indices,np.arange(len(data.train_ids))):raise ValueError('training index order changed')
    if len(set(groups.tolist()))<3:raise ValueError('need3 groups')
    seen=np.zeros(len(indices),dtype=int);plan=[]
    for fit,held in GroupKFold(3).split(indices,groups=groups):
        context_fit,validation=context_split(data,fit)
        if set(groups[fit])&set(groups[held]):raise ValueError('held group enters expert fit')
        if set(groups[held])&(set(groups[context_fit])|set(groups[validation])):raise ValueError('held group enters context')
        seen[held]+=1
        plan.append(dict(fit=fit,held=held,context_fit=context_fit,context_validation=validation))
    if not np.all(seen==1):raise ValueError('incomplete or duplicated OOF coverage')
    return plan


def expert_view_indices(kind):
    if kind=='clean':return [0]
    if kind=='standard':return list(range(len(OLD_VIEWS)))
    if kind=='block':return list(range(len(VIEW_NAMES)))
    raise ValueError('unknown expert')


def main():
    p=argparse.ArgumentParser();p.add_argument('--synthetic-smoke',type=Path,required=True);a=p.parse_args()
    x=np.ones((8,8,3),dtype=np.uint8);views=masks(x,[str(i) for i in range(8)],42,'synthetic')
    assert len(views)==12 and (views[-1].sum((1,2))==12).all()
    from types import SimpleNamespace
    plan=crossfit_plan(SimpleNamespace(dataset='aiops',train_ids=list(range(8)),train_indices=np.arange(8),train_groups=list(range(8))))
    assert sorted(np.concatenate([p['held'] for p in plan]).tolist())==list(range(8))
    a.synthetic_smoke.write_text(json.dumps({'benchmark_result':False,'views':VIEW_NAMES,'parts':len(plan),'status':'group_mask_spine_passed'}))

if __name__=='__main__':main()
