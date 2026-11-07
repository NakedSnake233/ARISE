"""Rank-aligned, calibration-guarded routing. All selection labels are training-only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from .arise_missing_v2 import probabilities, simplex_grid, fit_router as fit_r1, infer_router as infer_r1

CANDIDATES=('r1','rank_global','rank_coverage')


def event_mean(values, ids):
    ids=np.asarray(ids)
    return np.stack([values[ids==key].mean(axis=0) for key in sorted(set(ids.tolist()))]).mean(axis=0)


def rank_losses(scores, labels):
    """Stable catalogue ties, matching final evaluator."""
    p=probabilities(scores); y=np.asarray(labels,dtype=int)
    if len(p)!=len(y) or (y<0).any() or (y>=p.shape[-1]).any():raise ValueError('invalid labels')
    order=np.argsort(-p,axis=-1,kind='stable')
    target=y.reshape((len(y),)+(1,)*(p.ndim-1))
    rank=np.argmax(order==target,axis=-1)+1
    err=(rank!=1).astype(float)
    return err+.05*(1-1/rank),err


def coverage_regimes(a, reference):
    a=np.asarray(a)
    if a.ndim!=3 or a.shape[-1]!=3 or not np.isin(a,(0,1)).all():raise ValueError('invalid availability')
    ref=np.asarray(reference,dtype=float)
    if ref.shape!=(3,) or not np.isfinite(ref).all() or (ref<0).any():raise ValueError('invalid reference')
    coverage=a.mean(axis=1);present=coverage>0
    ratios=np.minimum(coverage/np.maximum(ref,1e-8),1)
    level=np.where(present,ratios,1).min(axis=1)
    bands=np.minimum((level*3).astype(int),2)
    return (~present).astype(int)@np.array([1,2,4])*3+bands


def guard_candidates(errors,anchor,views):
    keep=np.ones(errors.shape[1],dtype=bool)
    for view in sorted(set(views.tolist())):
        rows=views==view
        keep &= errors[rows].mean(0)<=errors[rows,anchor].mean()+1e-12
    return keep


def fit_rank(scores,a,y,ids,views,local=True):
    p=probabilities(scores);ids=np.asarray(ids);views=np.asarray(views)
    if p.ndim!=3 or p.shape[1]!=4 or len(ids)!=len(p) or len(views)!=len(p):raise ValueError('shape mismatch')
    if len(set(zip(ids.tolist(),views.tolist())))!=len(ids):raise ValueError('duplicate event/view')
    if not (views=='complete').any():raise ValueError('complete training reference missing')
    reference=np.median(np.asarray(a)[views=='complete'].mean(1),axis=0)
    regimes=coverage_regimes(a,reference);grid=simplex_grid()
    mixed=np.einsum('nes,ke->nks',p,grid)
    loss,errors=rank_losses(mixed,y);glob=event_mean(loss,ids)
    singles=np.flatnonzero((grid==1).any(1));anchor=int(singles[np.argmin(glob[singles])])
    keep=guard_candidates(errors,anchor,views)
    g=int(np.argmin(np.where(keep,glob,np.inf)))
    rules={}
    if local:
        for key in sorted(set(regimes.tolist())):
            use=regimes==key;n=len(set(ids[use].tolist()))
            objective=(n*event_mean(loss[use],ids[use])+8*glob)/(n+8)
            valid=guard_candidates(errors[use],g,views[use])
            chosen=int(np.argmin(np.where(valid,objective,np.inf)))
            rules[str(key)]={'weights':grid[chosen].tolist(),'events':n}
    return {'kind':'rank_coverage' if local else 'rank_global','reference':reference.tolist(),
        'global_weights':grid[g].tolist(),'anchor_weights':grid[anchor].tolist(),'regimes':rules,
        'objective':'Top1 error + .05*(1-MRR); event-balanced', 'calibration_guard':'per-view Top1 nonregression relative to anchor/global; no test guarantee'}


def infer(scores,a,router):
    if router['kind']=='r1':return infer_r1(scores,a,router['router'])[0]
    p=probabilities(scores);r=coverage_regimes(a,router['reference'])
    if p.ndim!=3 or p.shape[1]!=4 or len(r)!=len(p):raise ValueError('shape mismatch')
    w=np.asarray([router['regimes'].get(str(k),{}).get('weights',router['global_weights']) for k in r])
    if not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(1),1):raise ValueError('invalid weights')
    out=np.einsum('nes,ne->ns',p,w)
    out[np.asarray(a).sum((1,2))==0]=1/p.shape[-1]
    return probabilities(out)


def fit_candidate(kind,p,a,y,ids,views):
    if kind=='r1':return {'kind':'r1','router':fit_r1(p,a,y,ids)}
    return fit_rank(p,a,y,ids,views,kind=='rank_coverage')


def select_and_fit(p,a,y,ids,views,groups):
    ids=np.asarray(ids);views=np.asarray(views);groups=np.asarray(groups);y=np.asarray(y)
    if len(set(groups.tolist()))<3:raise ValueError('at least3 calibration groups required')
    # Every augmented row for an incident must remain in a single overlap group.
    if any(len(set(groups[ids==k].tolist()))!=1 for k in set(ids.tolist())):raise ValueError('event crosses groups')
    predictions={k:np.zeros((len(y),p.shape[-1])) for k in CANDIDATES};splits=[]
    for train,valid in GroupKFold(3).split(p,groups=groups):
        splits.append({'fit_groups':sorted(set(groups[train].tolist())), 'validation_groups':sorted(set(groups[valid].tolist()))})
        for kind in CANDIDATES:
            model=fit_candidate(kind,p[train],a[train],y[train],ids[train],views[train])
            predictions[kind][valid]=infer(p[valid],a[valid],model)
    summaries={}
    for k,pred in predictions.items():
        loss,err=rank_losses(pred,y)
        summaries[k]={'objective':float(event_mean(loss,ids)), 'view_top1':{v:float(1-err[views==v].mean()) for v in sorted(set(views.tolist()))}}
    base=summaries['r1']['view_top1'];eligible=['r1']
    for kind in CANDIDATES[1:]:
        values=summaries[kind]['view_top1']
        if all(values[v]>=base[v]-1e-12 for v in ('complete','no_metric','no_log','no_trace')) and min(values.values())>=min(base.values())-1e-12:
            eligible.append(kind)
    selected=min(eligible,key=lambda k:summaries[k]['objective'])
    model=fit_candidate(selected,p,a,y,ids,views)
    record={'selected':selected,'eligible':eligible,'candidates':summaries,'splits':splits,
        'selection_data_only':True,'outer_labels_used':False,'warning':'OOF is for selecting router, not independent accuracy'}
    return model,record,predictions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--synthetic-smoke',type=Path,required=True);args=parser.parse_args()
    p=np.tile(np.eye(3)[[0,1,2,0]][:,None],(1,4,1));a=np.ones((4,3,3),dtype=int)
    r=fit_rank(p,a,[0,1,2,0],['a','b','c','d'],['complete']*4)
    assert np.array_equal(infer(p,a,r).argmax(1),[0,1,2,0])
    args.synthetic_smoke.write_text(json.dumps({'benchmark_result':False,'router':r}))

if __name__=='__main__':main()
