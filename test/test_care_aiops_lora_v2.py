"""Guard the new diagnosis interface against metadata leaks and rank mistakes."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/care'))
from build_aiops_lora_v2 import prompt_for
from evaluate_aiops_lora_v2 import metrics


@pytest.fixture(scope='module')
def prepared():
    cfg=json.loads((ROOT/'configs/care/aiops-paper128-lora-v2.json').read_text())
    d=ROOT/cfg['data'];e=ROOT/cfg['evidence'];s=ROOT/cfg['sft']
    return (json.loads((d/'incidents.json').read_text()),json.loads((d/'metadata.json').read_text()),
            np.load(d/'features.npz'),np.load(e/'full.npz')['x'],json.loads((e/'full-names.json').read_text()),
            json.loads((s/'fpg.json').read_text()),s)


def test_real_inputs_do_not_depend_on_root_or_fault_metadata(prepared):
    rows,meta,features,x,names,fpg,_=prepared
    for i in [0,next(i for i,r in enumerate(rows) if r['split']=='test')]:
        row=rows[i]
        poisoned=dict(row,root_service='INJECTED_SECRET',root_entity='INJECTED_SECRET',fault_type='INJECTED_SECRET',source='INJECTED_SECRET')
        a=prompt_for(x[i],names,meta['services'],row,features,meta,fpg['edges'],42,i)
        b=prompt_for(x[i],names,meta['services'],poisoned,features,meta,fpg['edges'],42,i)
        assert a==b and 'INJECTED_SECRET' not in b[0]
        content=json.loads(a[0])
        assert sorted(a[1].values())==sorted(meta['services'])
        assert len(content['options'])==10


def test_real_frozen_training_targets_and_history_exclude_test(prepared):
    rows,meta,_,_,_,fpg,s=prepared
    train_ids={r['incident_id'] for r in rows if r['split']=='train'}
    test_ids={r['incident_id'] for r in rows if r['split']=='test'}
    assert train_ids.isdisjoint(test_ids) and len(train_ids)==102 and len(test_ids)==26
    assert set(fpg['training_ids'])==train_ids
    for edge in fpg['edges']:
        for supports in edge['relations'].values():
            assert all(x['incident_id'] in train_ids for x in supports)
    train=json.loads((s/'train.json').read_text());prov=json.loads((s/'provenance.json').read_text())
    assert len(train)==len(prov)==102
    for ex,p in zip(train,prov):
        options={v['option']:v['service'] for v in json.loads(ex['input'])['options']}
        assert options[ex['output']]==rows[p['index']]['root_service']
        assert rows[p['index']]['split']=='train'
    test=json.loads((s/'test_inputs.json').read_text())
    assert len(test)==78 and all('output' not in r for r in test)
    assert all(rows[r['index']]['split']=='test' for r in test)


def test_metric_denominators_are_incidents_and_rank_is_reciprocal():
    p=np.array([[.9,.1,0],[.9,.1,0],[.9,.1,0]])
    result=metrics(p,np.array([0,1,2]))
    assert result['correct']==1 and result['PR@1']==pytest.approx(1/3)
    assert result['MAP@5']==pytest.approx((1+1/2+1/3)/3)
    assert metrics(7*p,np.array([0,1,2]))==result
