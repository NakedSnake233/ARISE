import numpy as np
from exp_revision.t8_evaluate import score_rows
from exp_revision.metrics import ranking_metrics
from exp_revision.re2_data import span_table
import pandas as pd


def test_failed_finite_fallback_is_never_a_correct_prediction():
    scores=np.asarray([[[10.,0.],[10.,0.]]])
    truth={'a':('s1','cpu'),'b':('s1','mem')}
    cfg=dict(task='T7',dataset='toy',fold=0,repeat=0)
    rows=score_rows(scores,np.asarray([[False,True]]),['a','b'],['s1','s2'],truth,cfg,'mABC',['S0_none'])
    assert rows[0]['root_rank']==1 and np.isinf(rows[1]['root_rank'])
    assert rows[1]['top5']==[] and rows[1]['status']=='failed'
    assert ranking_metrics(np.asarray([r['root_rank'] for r in rows]))['PR@1']==.5


def test_trace_identity_is_per_trace_and_keeps_unknown_parent_dependency():
    frame=pd.DataFrame(dict(traceID=['t1','t1','t2'],spanID=['p','c','p'],parentSpanID=['','p',''],
        serviceName=['istio-init','frontendservice','s2'],operationName=['grpc.op','op','op'],
        startTime=[1e6,1e6+10,2e6],duration=[100,50,40]))
    table=span_table(frame,['frontend','s2'])
    assert table['service'].tolist()==[-1,0,1]
    assert table['parent'].tolist()==[0] and table['child'].tolist()==[1]
    assert table['operation'][0]==table['operation'][1]
