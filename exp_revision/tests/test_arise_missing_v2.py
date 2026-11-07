from types import SimpleNamespace
import copy
import numpy as np
import pytest
from exp_revision.arise_missing_v2 import fit_router,infer_router,probabilities,regimes,simplex_grid
from exp_revision.arise_missing_v2_train import training_views,inner_split,materialize,SCENES
from exp_revision.missingness import SCENARIOS


def test_probability_and_uniform_no_observation():
    scores=np.zeros((2,4,3));a=np.zeros((2,3,3),dtype=np.uint8)
    router=fit_router(scores,a,[0,1],['a','b'])
    p,w=infer_router(scores,a,router)
    assert np.allclose(p,1/3) and np.allclose(w.sum(1),1)
    scores[0,0,0]=100
    p,_=infer_router(scores,a,router);assert np.allclose(p,1/3)


def test_grid_and_zero_weight_experts():
    grid=simplex_grid();assert grid.shape==(35,4)
    assert np.allclose(grid.sum(1),1) and (grid>=0).all()
    assert np.array_equal(grid[0],[1,0,0,0])
    p=np.zeros((20,4,3));p[:,0,0]=1;p[:,1:,1]=1
    a=np.ones((20,3,3),dtype=np.uint8)
    router=fit_router(p,a,np.zeros(20,int),[str(i) for i in range(20)])
    assert router['global_weights']==[1,0,0,0]
    score,_=infer_router(p,a,router);assert (score.argmax(1)==0).all()


def test_regime_uses_only_current_availability():
    a=np.ones((2,4,3),dtype=np.uint8);a[1,:,0]=0
    assert regimes(a)[0]!=regimes(a)[1]
    assert np.array_equal(regimes(a),regimes(a[:,::-1]))
    with pytest.raises(ValueError):regimes(a.astype(float)+.1)


def test_unseen_regime_uses_global_and_service_permutation_is_equivariant():
    rng=np.random.default_rng(4);p=rng.random((20,4,5));a=np.ones((20,5,3),dtype=np.uint8)
    router=fit_router(p,a,np.arange(20)%5,[str(i) for i in range(20)])
    a[:,:,2]=0
    out,w=infer_router(p,a,router)
    assert np.allclose(w,np.tile(router['global_weights'],(20,1)))
    order=[4,2,0,1,3];permuted,_=infer_router(p[:,:,order],a[:,order],router)
    assert np.allclose(permuted,out[:,order])


def test_masks_restore_available_modalities_and_never_create_observations():
    a=np.array([[[1,0,0],[0,1,1],[0,0,0]],[[0,1,0],[1,0,1],[0,0,0]]],dtype=np.uint8)
    views=training_views(a,['x','y'],42,'fit')
    assert len(views)==10
    for v in views:assert (v<=a).all()
    for v in views[4:8]:assert (v.sum(2)[a.sum(2)>0]>0).all()
    for x,y in zip(views,training_views(a,['x','y'],42,'fit')):assert np.array_equal(x,y)
    reordered=training_views(a[::-1],['y','x'],42,'fit')
    for x,y in zip(views,reordered):assert np.array_equal(x,y[::-1])


def test_training_calibration_streams_are_independent():
    a=np.ones((30,15,3),dtype=np.uint8);ids=list(map(str,range(30)))
    x=training_views(a,ids,42,'fit');y=training_views(a,ids,42,'calibration')
    assert not np.array_equal(x[6],y[6])


def test_split_never_breaks_groups_or_uses_test_labels():
    d=SimpleNamespace(train_indices=np.arange(40),train_groups=[str(i//2) for i in range(40)])
    fit,held=inner_split(d)
    assert set(fit)|set(held)==set(range(40)) and not set(fit)&set(held)
    assert not {d.train_groups[i] for i in fit}&{d.train_groups[i] for i in held}
    assert np.array_equal(fit,inner_split(d)[0])


def test_all_training_views_recompute_observable_features():
    calls=[]
    class Evidence:
        data=SimpleNamespace(train_ids=['a','b'],availability=np.ones((2,3,3),dtype=np.uint8))
        def recompute(self,indices,mask,pipeline):
            calls.append((tuple(indices),pipeline,mask.copy()))
            return np.ones((len(indices),3,1))*mask[:,:,0,None],['metric:x:positive_z'],mask
    e=Evidence();spec,generic,masks=materialize(e,np.array([0,1]),42,'fit')
    assert len(calls)==10 and all(c[1]=='edge_gated' for c in calls)
    assert np.all(spec[1][:,:,0]==0)
    with pytest.raises((ValueError,IndexError)):materialize(e,np.array([2]),42,'fit')


def test_expert_fits_exclude_calibration_rows(monkeypatch):
    from exp_revision import arise_missing_v2_train as runner,train_evidence
    seen=[]
    class Model:
        def set_params(self,**kwargs):return self
    monkeypatch.setattr(train_evidence,'model_for',lambda _:Model())
    monkeypatch.setattr(train_evidence,'fit',lambda model,x,y:seen.append((x.copy(),y.copy())) or model)
    values=[np.arange(4)[:,None,None]+10*v for v in range(10)]
    labels=np.array([0,1,2,3]);runner.fit_experts((values,values,None),labels,np.array([0,2]),42)
    assert len(seen)==3 and np.array_equal(seen[0][1],[0,2])
    for _,y in seen:assert set(y)=={0,2}
    assert len(seen[1][0])==20


def test_scenarios_match_frozen_context_order():
    assert SCENES==[f'{s}_{p or "none"}' for s,p in SCENARIOS]


def test_invalid_scores_are_not_silently_repaired():
    for x in [np.array([[np.nan,1]]),np.array([[-1,2]])]:
        with pytest.raises(ValueError):probabilities(x)


def test_context_checkpoint_and_router_calibration_are_three_way_disjoint():
    from exp_revision.arise_missing_v2_train import context_split
    d=SimpleNamespace(dataset='tt',train_indices=np.arange(80),train_groups=[str(i//2) for i in range(80)])
    fit,held=inner_split(d);ctx_fit,ctx_val=context_split(d,fit)
    group_sets=[{d.train_groups[i] for i in a} for a in [ctx_fit,ctx_val,held]]
    assert all(not group_sets[i]&group_sets[j] for i in range(3) for j in range(i))
    assert set(ctx_fit)|set(ctx_val)==set(fit)
    d.dataset='aiops';ctx_fit,ctx_val=context_split(d,fit)
    assert np.array_equal(ctx_fit,fit) and len(ctx_val)==0


def test_router_does_not_treat_repeated_views_as_independent_incidents():
    rng=np.random.default_rng(31);p=rng.random((6,4,3));a=np.ones((6,3,3),dtype=np.uint8)
    y=np.array([0,0,1,1,2,2]);ids=['a','a','b','b','c','c']
    r=fit_router(p,a,y,ids)
    order=np.array([0,1,0,1,0,1,2,3,4,5])
    rr=fit_router(p[order],a[order],y[order],np.asarray(ids)[order])
    assert r['global_weights']==rr['global_weights']
    for k in r['regimes']:
        assert r['regimes'][k]['weights']==rr['regimes'][k]['weights']
        assert r['regimes'][k]['objective']==pytest.approx(rr['regimes'][k]['objective'])
        assert r['regimes'][k]['unique_events']==rr['regimes'][k]['unique_events']
