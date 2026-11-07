"""Behavioral checks for evidence correction: trace attribution and isolation."""
import json
import gzip
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/care'))
from trace_residual import interval_union, trace_records, extract_one
from evidence_reranker import fuse
from contrast_prompt import build_prompt
from prepare_aiops_rpc import records, operation
from train_base import CAREBase
import pandas as pd
import torch


def span(sid,service,start,duration,kind,parent=None):
    return {'traceID':'trace','spanID':sid,'processID':service,'startTime':start,
            'duration':duration,'tags':[{'key':'span.kind','value':kind}],
            'references':([] if parent is None else [{'refType':'CHILD_OF','traceID':'trace','spanID':parent}])}


def test_exclusive_time_subtracts_union_and_clips_to_parent():
    assert interval_union([(0,40),(30,70),(80,120)],10,100)==80
    trace={'processes':{'a':{'serviceName':'a'},'b':{'serviceName':'b'}},'spans':[
        span('server','a',0,100000,'server'),
        span('client','a',10000,50000,'client','server'),
        span('client2','a',40000,50000,'client','server'),
        span('remote','b',20000,10000,'server','client')]}
    local,edges=trace_records(trace,{'a','b'})
    assert local[0]==(0.,'a',100.,20.)
    assert edges==[(.01,'a','b',40.)]


def test_trace_features_are_label_and_injection_metadata_independent(tmp_path):
    traces=[]
    for stamp,gap in [(s,1) for s in [1,2,3]]+[(s,100) for s in [11,12,13]]:
        traces.append({'processes':{'a':{'serviceName':'a'},'b':{'serviceName':'b'}},'spans':[
            span('client','a',stamp*1000000,(gap+1)*1000,'client'),
            span('server','b',stamp*1000000,1000,'server','client')]})
    with gzip.open(tmp_path/'traces.jsonl.gz','wt') as f:
        for trace in traces:f.write(json.dumps(trace)+'\n')
    row={'source':str(tmp_path/'incident.json'),'start':0,'end':20}
    x,names=extract_one(row,['a','b'],10)
    poisoned=dict(row,root_service='a',fault_type='leaked',fault_start=-999,incident_id='root=b')
    xx,nn=extract_one(poisoned,['a','b'],10)
    assert np.array_equal(x,xx) and names==nn
    # The link alone cannot distinguish its two endpoints.
    assert x[0,names.index('rpc:incident:hot_fraction')]==1
    assert x[1,names.index('rpc:incident:hot_fraction')]==1
    assert np.isfinite(x).all()


def test_candidate_fusion_preserves_mass_and_external_ranking():
    base=np.array([[.4,.3,.15,.1,.04,.01]])
    ev=np.array([[0.,0.,1.,0.,0.,0.]])
    for weight in [0.,.25,1.]:
        result=fuse(base,ev,weight,3)
        assert np.array_equal(result[:,3:],base[:,3:])
        assert result.sum()==pytest.approx(base.sum())
    assert fuse(base,ev,1.,3).argmax()==2
    assert np.allclose(fuse(base,np.zeros_like(ev),1.,3),base)


def test_prompt_cannot_reveal_candidate_score_order():
    # Empty telemetry suffices to test presentation independently of extraction.
    a=build_prompt(np.empty((3,0)),[],['a','b','c'],['c','a','b'],[],'anonymous')
    b=build_prompt(np.empty((3,0)),[],['a','b','c'],['b','c','a'],[],'anonymous')
    assert a==b
    assert a['candidates']==['a','b','c']


def test_exported_rpc_requires_matching_operation_and_unambiguous_link():
    def s(sid,parent,service,op,duration):
        return {'trace_id':'t','span_id':sid,'parent_span':parent,'cmdb_id':service,
                'operation_name':op,'timestamp':1000,'duration':duration}
    frame=pd.DataFrame([s('a','root','frontend-0','/hipstershop.CurrencyService/Convert',10000),
                        s('b','a','currencyservice-1','grpc.hipstershop.CurrencyService/Convert',1000)])
    local,edges,audit=records(frame)
    assert edges==[(1.,'frontend','currencyservice',9.)]
    assert audit['matched_pairs']==1 and len(local)==1
    bad=frame.copy();bad.loc[1,'operation_name']='Unrelated/Operation'
    assert records(bad)[1]==[]
    extra=pd.concat([frame,pd.DataFrame([s('c','a','currencyservice-2','hipstershop.CurrencyService/Convert',1000)])])
    assert records(extra)[1]==[]


@pytest.mark.parametrize('ablation,index',[('no_metrics',0),('no_traces',1),('no_logs',2)])
def test_ablated_modality_values_cannot_change_base_prediction(ablation,index):
    torch.manual_seed(42);model=CAREBase(2,2,2,hidden=16,heads=8,ablation=ablation).eval()
    inputs=[torch.randn(2,3,7,2) for _ in range(3)];adj=torch.eye(3,dtype=torch.bool)
    first=model(*inputs,adj)[0]
    inputs[index]=torch.randn_like(inputs[index])*100
    assert torch.equal(first,model(*inputs,adj)[0])
