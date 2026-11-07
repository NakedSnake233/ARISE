from types import SimpleNamespace
import numpy as np
import pytest
from exp_revision.arise_missing_v4 import masks,crossfit_plan,expert_view_indices,VIEW_NAMES
from exp_revision.arise_missing_v2_train import training_views


def test_group_coverage_including_nested_checkpoint_groups():
    for ds in ['tt','aiops']:
        d=SimpleNamespace(dataset=ds,train_ids=[str(i) for i in range(24)],train_indices=np.arange(24),train_groups=np.repeat(np.arange(12),2))
        plan=crossfit_plan(d);assert np.array_equal(np.sort(np.concatenate([p['held'] for p in plan])),np.arange(24))
        for p in plan:
            g=lambda x:set(d.train_groups[p[x]].tolist())
            assert not g('held')&g('fit')
            assert not g('held')&(g('context_fit')|g('context_validation'))
            assert not g('context_fit')&g('context_validation')
            assert g('context_fit')|g('context_validation')==g('fit')


def test_outage_count_and_original_availability():
    a=np.ones((3,8,3),dtype=int);a[:,0]=0;a[:,1,1]=0
    v=masks(a,['a','b','c'],42,'fit')
    assert len(v)==12
    for rate,mask in zip([.25,.5],v[-2:]):
        removed=(a.sum(2)>0)&(mask.sum(2)==0)
        assert (removed.sum(1)==np.ceil(7*rate)).all()
        assert (mask<=a).all()
    assert all((x<=a).all() for x in v)


def test_masks_preserve_existing_views_and_are_independent():
    a=np.ones((4,15,3),dtype=int);ids=['a','b','c','d']
    v=masks(a,ids,42,'fit');base=training_views(a,ids,42,'fit')
    assert all(np.array_equal(x,y) for x,y in zip(v[:10],base))
    assert any(not np.array_equal(x,y) for x,y in zip(v[-2:],masks(a,ids,42,'calibration')[-2:]))
    perm=[3,0,2,1];rev=np.argsort(perm)
    assert all(np.array_equal(x,y[rev]) for x,y in zip(v,masks(a[perm],[ids[i] for i in perm],42,'fit')))


def test_empty_observation_and_invalid_input():
    a=np.zeros((2,8,3),dtype=int)
    assert not np.stack(masks(a,['a','b'],42,'fit')).any()
    with pytest.raises(ValueError):masks(np.full_like(a,2),['a','b'],42,'fit')


def test_matched_augmentation_membership_and_training_labels(monkeypatch):
    from exp_revision import train_evidence as base
    from exp_revision.arise_missing_v4_train import fit_models
    records=[]
    class Model:
        def set_params(self,**kw):return self
    monkeypatch.setattr(base,'model_for',lambda depth:Model())
    monkeypatch.setattr(base,'fit',lambda model,x,y:records.append((x.copy(),y.copy())) or model)
    n=9;fit=np.array([0,2,3,6]);labels=np.arange(n)%3
    values=[np.broadcast_to(np.arange(n)[:,None,None],(n,3,2)).copy() for _ in VIEW_NAMES]
    fit_models((values,values,[None]*12),labels,fit,42)
    assert [len(x) for x,y in records]==[4,40,40,48,48]
    for (x,y),kind in zip(records,['clean','standard','standard','block','block']):
        assert set(x[:,0,0])==set(fit)
        assert np.array_equal(y,np.tile(labels[fit],len(expert_view_indices(kind))))
