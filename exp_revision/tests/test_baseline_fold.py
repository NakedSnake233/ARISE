from types import SimpleNamespace
import copy
import numpy as np
import pytest
from exp_revision import baseline_fold as b


def toy():
    rng=np.random.default_rng(9)
    data=SimpleNamespace(dataset="toy",fold=0,services=["a","b","c"],
        train_ids=["fit","validation"],test_ids=["test"],train_labels=np.array([0,1]),
        train_indices=np.array([0,1]),test_indices=np.array([2]),
        availability=np.ones((3,3,3),dtype=np.uint8),
        metadata={"before":2,"step":1,"calls":{
            "fit":[{"source":"a","target":"b","count":3}],
            "validation":[{"source":"b","target":"c","count":8}],
            "test":[{"source":"c","target":"a","count":999}]}})
    for name in b.MODALITIES: setattr(data,name,rng.uniform(0,4,(3,3,4,2)).astype(np.float32))
    return data


def test_fit_graph_excludes_validation_and_test():
    d=toy(); bundle=b.make_bundle(d,np.array([0]),np.array([1]))
    graph=b.training_adjacency(bundle)
    expected=np.eye(3);expected[0,1]+=np.log(4)
    np.testing.assert_allclose(graph,expected)
    d.metadata["calls"]["test"][0]["count"]=9999999
    d2=b.make_bundle(d,np.array([0]),np.array([1]));d2.labels[2]=2
    np.testing.assert_array_equal(b.training_adjacency(d2),graph)
    for method,cfg in [("Nezha",{"event_threshold":1.5}),
                       ("SBLD",{"active_threshold":1.,"formula":"tarantula"})]:
        first=b.fit_classical(bundle,method,cfg);second=b.fit_classical(d2,method,cfg)
        if isinstance(first,dict):
            for name in first:np.testing.assert_array_equal(first[name],second[name])
        else:np.testing.assert_array_equal(first,second)


def test_descriptor_scaler_is_fit_only_and_missing_is_recomputed():
    d=toy(); features={name:getattr(d,name) for name in b.MODALITIES}
    _,scaler=b.descriptors(features,2,fit=np.array([0]))
    changed=copy.deepcopy(features)
    for value in changed.values(): value[2]+=900
    _,second=b.descriptors(changed,2,fit=np.array([0]))
    for name in features:
        np.testing.assert_array_equal(scaler[name][0],second[name][0])
        np.testing.assert_array_equal(scaler[name][1],second[name][1])
    removed={name:np.zeros_like(value[2:]) for name,value in features.items()}
    values,_=b.descriptors(removed,2,scaler)
    for name,value in values.items():
        np.testing.assert_allclose(value,np.broadcast_to(-scaler[name][0]/scaler[name][1],value.shape),rtol=1e-6)
        assert np.any(value!=0)


def test_md_restore_only_original_available_modalities():
    class DeleteAll:
        def random(self,shape):return np.zeros(shape)
        def choice(self,values):return values[-1]
    a0=np.array([[[1,0,0],[1,1,0],[0,0,0]]],dtype=np.uint8)
    effective=a0*b.dropout_mask(a0,DeleteAll())
    np.testing.assert_array_equal(effective,[[[1,0,0],[0,1,0],[0,0,0]]])


@pytest.mark.parametrize("method,cfg",[
    ("Nezha",{"event_threshold":1.5,"service_aggregation":"max","graph_mix":0.}),
    ("SBLD",{"active_threshold":1.,"formula":"tarantula","service_aggregation":"sum"}),
    ("LogFaultFlagger",{"C":1.,"max_iter":2000}),
    ("MicroRCA",{"damping":.95,"direct_mix":.75,"reverse":True}),
    ("MicroScope",{"anomaly_threshold":2.,"damping":.65,"direct_mix":.25,"reverse":True}),
    ("MicroRank",{"anomaly_quantile":.75,"damping":.65,"spectrum":"tarantula"})])
def test_classical_cached_fits_match_original_scoring_with_same_graph(method,cfg):
    d=toy();bundle=b.make_bundle(d,np.array([0,1]),np.array([],dtype=int))
    graph=b.training_adjacency(bundle)
    fitted=b.fit_classical(bundle,method,cfg)
    original=b.classical.adjacency_for_incident
    b.classical.adjacency_for_incident=lambda *_:graph.copy()
    try: expected=b.classical.METHODS[method](bundle,cfg)
    finally: b.classical.adjacency_for_incident=original
    actual=b.classical_scores(bundle,method,cfg,fitted,graph)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
