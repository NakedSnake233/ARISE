import numpy as np
import pytest
from exp_revision.arise_missing_v3 import fit_rank,infer,rank_losses,coverage_regimes,select_and_fit


def fixture():
    rng=np.random.default_rng(7);n=12;s=5
    views=np.repeat(['complete','no_metric','no_log','no_trace','random'],n)
    ids=np.tile(np.array([f'i{i}' for i in range(n)]),5);y=np.tile(np.arange(n)%s,5)
    a=np.ones((len(y),s,3),int)
    for i in range(3):a[(i+1)*n:(i+2)*n,:,i]=0
    a[4*n:]=rng.integers(0,2,size=a[4*n:].shape)
    p=rng.random((len(y),4,s));p[np.arange(len(y)),0,y]+=1
    return p,a,y,ids,views


def test_rank_loss_matches_catalogue_ties():
    _,err=rank_losses(np.ones((2,3)),[0,1]);assert err.tolist()==[0,1]
    loss,_=rank_losses(np.array([[.1,.8,.1]]),[1]);assert loss[0]==0


def test_per_view_calibration_nonregression_and_uniform():
    p,a,y,ids,v=fixture();r=fit_rank(p,a,y,ids,v)
    _,err=rank_losses(infer(p,a,r),y)
    anchor=np.einsum('nes,e->ns',p/p.sum(-1,keepdims=True),r['anchor_weights']);_,base=rank_losses(anchor,y)
    for k in set(v):assert err[v==k].mean()<=base[v==k].mean()+1e-12
    assert np.allclose(infer(p,np.zeros_like(a),r),1/p.shape[-1])


def test_reference_uses_complete_training_only():
    p,a,y,ids,v=fixture();r=fit_rank(p,a,y,ids,v)
    assert r['reference']==[1,1,1]
    assert coverage_regimes(np.ones((1,10,3),int),[1,1,1])[0]==2
    sparse=np.zeros((1,10,3),int);sparse[:,:5]=1
    assert coverage_regimes(sparse,[1,1,1])[0]==1
    assert coverage_regimes(sparse,[.5,.5,.5])[0]==2


def test_group_oof_isolation_and_selection_guard():
    p,a,y,ids,v=fixture();r,rec,oof=select_and_fit(p,a,y,ids,v,ids)
    for part in rec['splits']:assert not set(part['fit_groups'])&set(part['validation_groups'])
    b=rec['candidates']['r1']['view_top1'];z=rec['candidates'][rec['selected']]['view_top1']
    for k in ['complete','no_metric','no_log','no_trace']:assert z[k]>=b[k]-1e-12
    assert min(z.values())>=min(b.values())-1e-12
    assert all(np.isfinite(z).all() and np.allclose(z.sum(1),1) for z in oof.values())


def test_row_order_and_probability_scale_invariance():
    p,a,y,ids,v=fixture();r=fit_rank(p,a,y,ids,v);ix=np.random.default_rng(2).permutation(len(y))
    assert fit_rank(p[ix],a[ix],y[ix],ids[ix],v[ix])==r
    assert np.allclose(infer(p,a,r),infer(p*np.arange(1,5)[None,:,None],a,r))


def test_reject_invalid_scores_and_duplicate_rows():
    p,a,y,ids,v=fixture()
    with pytest.raises(ValueError):fit_rank(-p,a,y,ids,v)
    with pytest.raises(ValueError):fit_rank(np.concatenate([p,p]),np.concatenate([a,a]),np.tile(y,2),np.tile(ids,2),np.tile(v,2))
