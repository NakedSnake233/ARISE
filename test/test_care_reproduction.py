import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/care'))
from prepare_aiops import normalize, split_incidents, service_name
from build_sft import build_fpg, retrieve
from train_base import CAREBase, ranking_metrics
from collect_trainticket import chaos_manifest
from correct import correct, path_search
from paper_split import freeze, apply_manifest
import collect_trainticket as collector
from prepare_trainticket import parse_log_timestamp


def test_training_statistics_do_not_use_test_values():
    data=np.array([[[[1.],[3.]]],[[[100.],[200.]]]],dtype='float32')
    _,stats=normalize(data,np.array([True,False]),2)
    assert np.array(stats['mean']).item()==2.
    assert np.array(stats['std']).item()==1.


def test_overlapping_incidents_stay_in_same_split():
    rows=[{'cloud':'a','start':0,'end':20},{'cloud':'a','start':10,'end':30}]
    rows += [{'cloud':'a','start':i*100,'end':i*100+30} for i in range(1,20)]
    result=split_incidents(rows,42)
    assert result[0]['split']==result[1]['split']
    assert {r['split'] for r in result}=={'train','validation','test'}


def test_fpg_never_admits_test_edges_and_requires_time_order():
    def row(split,name,source,target):
        return {'split':split,'incident_id':name,'fault_type':'cpu','start':0,'end':20,
                'calls':[{'source':source,'target':target,'count':3}]}
    def obs(onset):return {'anomalous':True,'onset':onset,'metrics':[]}
    rows=[row('train','tr','a','b'),row('test','te','b','a')]
    ev=[{'a':obs(15),'b':obs(1)},{'a':obs(15),'b':obs(1)}]
    fpg=build_fpg(rows,ev,delta=2)
    assert len(fpg)==1 and fpg[0]['source']=='a'
    assert fpg[0]['relations']['PROPAGATES_TO']==[]
    assert retrieve(fpg,['a'],exclude='tr')==[]
    assert retrieve(fpg,['a'])[0]['training_incident_ids']==['tr']


def test_gat_all_nodes_have_finite_gradients_and_attention():
    torch.manual_seed(2)
    model=CAREBase(3,3,4,hidden=16,heads=8)
    adj=torch.eye(3,dtype=torch.bool);adj[0,1]=adj[1,0]=True
    m=torch.randn(2,3,7,3);t=torch.randn_like(m);logs=torch.rand(2,3,7,4)
    logits,contrast,attention=model(m,t,logs,adj)
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,2]))+.1*contrast
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.allclose(attention.sum(1),torch.ones(2,3),atol=1e-6)
    assert torch.all(attention[:,~adj]==0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_single_culprit_map_definition():
    scores=ranking_metrics(np.array([[.9,.1,.0],[.6,.3,.1]]),np.array([0,2]))
    assert scores['PR@1']==.5
    assert scores['PR@3']==1.
    assert scores['MAP@3']==pytest.approx((1+1/3)/2)


def test_chaos_targets_only_dedicated_namespace_and_recovers():
    cfg=json.loads((ROOT/'configs/care/collection.json').read_text())
    for fault in cfg['faults']:
        manifest=chaos_manifest('ts-travel-service',fault,'care-test',cfg)
        assert manifest['spec']['selector']['namespaces']==['care-tt']
        assert manifest['spec']['selector']['labelSelectors']['care.experiment']=='trainticket'
        assert manifest['spec']['duration']=='120s'
    cfg['namespace']='production'
    with pytest.raises(ValueError):chaos_manifest('ts-travel-service','cpu_load','care-test',cfg)


def test_service_mapping_is_explicit():
    assert service_name('currencyservice2-0')=='currencyservice'
    assert service_name('node-5') is None
    assert service_name('unknown2-0') is None


def test_edge_gate_nullifies_attention_and_semantic_score():
    evidence={'a':{'anomalous':True,'onset':20,'health':.1},
              'b':{'anomalous':True,'onset':0,'health':.1}}
    facts=[{'source':'a','target':'b','relation':'PROPAGATES_TO','fact_id':'f'}]
    proposal={'root_cause':'a','causal_scores':{'a':1.,'b':0.},
              'propagation_edges':[{'source':'a','target':'b','confidence':1.,'fact_id':'f'},
                                   {'source':'b','target':'a','confidence':1.,'fact_id':'fake'}]}
    result=correct({'a':.4,'b':.6},evidence,facts,proposal,{('a','b'):1.,('b','a'):1.},delta=1.)
    assert all(edge['confidence']==0 for edge in result['edges'])
    assert len(result['rejected_proposals'])==1
    assert not result['changed']


def test_path_uses_product_and_never_zero_or_reverse_edge():
    graph={('a','b'):.8,('b','c'):.8,('a','c'):.3,('c','a'):0.}
    assert path_search(graph,'a','c')['nodes']==['a','b','c']
    assert path_search(graph,'a','c')['confidence']==pytest.approx(.64)
    assert path_search(graph,'c','a') is None


def test_frozen_paper_split_ignores_labels_and_prevents_overlap_leakage():
    rows=[{'incident_id':str(i),'cloud':'c','start':i*100,'end':i*100+50,'root_service':'a'} for i in range(150)]
    rows[1]['start']=10
    entries=freeze(rows,42)
    assert len(entries)==128
    assert sum(e['split']=='train' for e in entries)==102
    assert sum(e['split']=='test' for e in entries)==26
    assert freeze([{**r,'root_service':'changed'} for r in rows],42)==entries
    manifest={'counts':{'train':102,'test':26},'incidents':entries}
    selected=apply_manifest(rows,manifest)
    pair=[r for r in selected if r['incident_id'] in ('0','1')]
    assert not pair or len(pair)==2 and pair[0]['split']==pair[1]['split']
    broken=[{**r,'end':r['end']+1} for r in rows]
    with pytest.raises(ValueError):apply_manifest(broken,manifest)


def test_overlapping_metric_patches_match_independent_encoding():
    torch.manual_seed(42)
    model=CAREBase(2,3,4,hidden=16,metric_patch_size=3,metric_patch_stride=1)
    x=torch.randn(2,3,7,2,requires_grad=True)
    expected=torch.stack([model.metrics(x[:,:,i:i+3]) for i in range(5)]).mean(0)
    actual=model.encode_metrics(x)
    assert torch.allclose(actual,expected,atol=1e-6)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_trace_health_requires_real_recent_application_span(monkeypatch):
    trace={'traceID':'t','processes':{'p1':{'serviceName':'jaeger-all-in-one'},
                                  'p2':{'serviceName':'ts-train-service'}},
           'spans':[{'processID':'p1','startTime':20_000_000}]}
    monkeypatch.setattr(collector,'query_traces',lambda *args:[trace])
    with pytest.raises(RuntimeError,match='refuse fault injection'):
        collector.check_target_traces('ts-train-service',10,30)
    trace['spans'].append({'processID':'p2','startTime':5_000_000})
    with pytest.raises(RuntimeError):collector.check_target_traces('ts-train-service',10,30)
    trace['spans'].append({'processID':'p2','startTime':20_000_000})
    assert collector.check_target_traces('ts-train-service',10,30)['trace_count']==1


def test_saturated_trace_query_is_partitioned_without_time_gaps(monkeypatch):
    intervals=[]
    def query(url,**params):
        start,end=params['start']/1e6,params['end']/1e6
        if end-start>2:return {'data':[{}]*1000}
        intervals.append((start,end));return {'data':[{'traceID':str(start)}]}
    monkeypatch.setattr(collector,'get_json',query)
    assert len(collector.query_traces('ts-train-service',0,8))==4
    assert intervals==[(0,2),(2,4),(4,6),(6,8)]


def test_capture_excludes_jaeger_and_reads_final_application_spans(monkeypatch,tmp_path):
    import threading,gzip
    stop=threading.Event();stop.set();queried=[]
    monkeypatch.setattr(collector,'get_json',lambda *a,**kw:{'data':['jaeger-all-in-one','ts-train-service']})
    def query(service,start,end):
        queried.append(service)
        return [{'traceID':'t','processes':{'p':{'serviceName':service}},
                 'spans':[{'processID':'p','startTime':20_000_000}]}]
    monkeypatch.setattr(collector,'query_traces',query)
    errors=[];path=tmp_path/'traces.gz'
    assert collector.capture_traces(stop,path,10,errors)==1
    assert queried==['ts-train-service'] and not errors
    with gzip.open(path,'rt') as f:assert json.loads(f.readline())['traceID']=='t'


def test_kubernetes_nanosecond_logs_keep_utc_and_bin_membership():
    value=parse_log_timestamp('2026-09-23T18:01:20.872027356Z')
    expected=parse_log_timestamp('2026-09-23T18:01:20.872027+00:00')
    offset=parse_log_timestamp('2026-09-24T02:01:20.872027356+08:00')
    assert value==expected==offset
    start=parse_log_timestamp('2026-09-23T18:01:20Z')
    assert int((value-start)//5)==0
    with pytest.raises(ValueError):parse_log_timestamp('2026-09-23T18:01:20.872027356')
