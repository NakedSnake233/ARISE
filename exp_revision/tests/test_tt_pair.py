import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from exp_revision.common import dataset_dir
from exp_revision.tt_pair_data import make_cohorts
from exp_revision.tt_pair_models import pair_permutation, control_columns, neural_scores, gm


class PairChecks(unittest.TestCase):
    def test_queue_handles_atomic_storage_denial_without_attempt(self):
        from exp_revision import tt_pair_queue as queue
        from exp_revision.common import ROOT
        from unittest.mock import MagicMock
        guard=MagicMock();guard.poll.return_value={'safe_to_launch':True}
        limits=dict(approved=True,cpu_limit=380.,gpu_limit=64.,prior_cpu=0.,prior_gpu=0.)
        with tempfile.TemporaryDirectory(prefix='PLACEHOLDER_storage_check_',dir=ROOT/'cache') as tmp:
            folder=Path(tmp)
            with patch.object(queue,'FOLDER',folder),patch.object(queue.q,'FOLDER',folder),\
                 patch.object(queue,'budget',return_value=limits),patch.object(queue,'DiskGuard',return_value=guard),\
                 patch.object(queue.q,'start',side_effect=queue.InsufficientSpace('capacity reserved by another queue')):
                result=queue.run('smoke')
            self.assertEqual(result['status'],'storage_limited')
            state=json.loads((folder/'queue_state.json').read_text())
            self.assertTrue(all(r['status']=='pending' and not r['attempts'] for r in state['jobs'].values()))

    def test_pair_outer_and_inner_isolation_balanced(self):
        rows=json.loads((dataset_dir('tt')/'incidents.json').read_text());lookup={r['incident_id']:r for r in rows};parts=make_cohorts(rows)
        pair=lambda k:(lookup[k]['root_service'],lookup[k]['fault_type'])
        alltests=[]
        for p in parts.values():
            self.assertEqual(len(p['train']),120);self.assertEqual(len(p['test']),30)
            self.assertFalse(set(map(pair,p['train']))&set(map(pair,p['test'])))
            inner=p['context_inner'];self.assertFalse(set(map(pair,inner['fit']))&set(map(pair,inner['validation'])))
            self.assertEqual(set(inner['fit'])|set(inner['validation']),set(p['train']))
            for field,n in [('root_service',3),('fault_type',6)]:
                from collections import Counter
                self.assertEqual(set(Counter(lookup[k][field] for k in p['test']).values()),{n})
            alltests+=p['test']
        self.assertEqual(len(set(alltests)),150)

    def test_permutation_preserves_pair_blocks_and_counts(self):
        d=SimpleNamespace(train_labels=np.repeat(np.arange(10),3),train_groups=np.repeat(np.arange(10),3))
        a=pair_permutation(d,42);np.testing.assert_array_equal(np.sort(a),d.train_labels)
        self.assertFalse(np.array_equal(a,d.train_labels))
        self.assertTrue(all(len(set(a[i:i+3]))==1 for i in range(0,30,3)))
        np.testing.assert_array_equal(a,pair_permutation(d,42))

    def test_control_feature_exclusion(self):
        names=['rpc:server:observed','rpc:exclusive:observed','rpc:incoming:degree','metric:a:x','raw_metric:a:x','log:a']
        c=control_columns(names)
        self.assertEqual(c['ET-no-RPC-observed'].tolist(),[2,3,4,5])
        self.assertEqual(c['ET-RPC-only'].tolist(),[0,1,2])
        self.assertEqual(c['ET-no-RPC'].tolist(),[3,4,5])
        self.assertEqual(c['ET-metrics-only'].tolist(),[3,4])

    def test_eadro_md_uses_same_graph_forward(self):
        import torch
        torch.manual_seed(42)
        widths={name:4 for name in ('metrics','logs','traces')}
        model=gm.baseline.EadroAdapter(widths,8,2,0.);model.eval()
        data=SimpleNamespace(metadata={'before':60,'step':5},test_ids=['a','b'])
        prepared={name:np.ones((2,3,4),np.float32) for name in widths}
        with patch.object(gm.baseline,'descriptors',return_value=(prepared,None)):
            a=neural_scores(model,data,{},None,np.eye(3,dtype=np.float32),'Eadro',{})
            b=neural_scores(model,data,{},None,np.eye(3,dtype=np.float32),'Eadro+MD',{})
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(2,3))
        np.testing.assert_allclose(a.sum(1),1,atol=1e-6)

    def test_collector_fusion_and_smoke_separation(self):
        from exp_revision import tt_pair_collect as c
        from exp_revision.common import ROOT,sha256
        import pandas as pd
        records=json.loads((dataset_dir('tt')/'incidents.json').read_text())[:2]
        services=json.loads((dataset_dir('tt')/'metadata.json').read_text())['services']
        ids=[r['incident_id'] for r in records];truth=[services.index(r['root_service']) for r in records]
        ctx=np.zeros((4,2,len(services)));ev=np.zeros_like(ctx)
        for i,y in enumerate(truth):
            other=(y+1)%len(services);ctx[:,i,y]=.95;ctx[:,i,other]=.05
            ev[:,i,y]=.05;ev[:,i,other]=.95
        with tempfile.TemporaryDirectory(prefix='PLACEHOLDER_pair_check_',dir=ROOT/'cache') as tmp:
            root=Path(tmp);jobs={}
            for component in ['context','experts']:
                folder=root/component;folder.mkdir();payload=dict(scores=ctx if component=='context' else np.stack([ev]*4),incident_ids=ids,scenario_keys=c.SCENES)
                if component=='experts':payload.update(method_names=['ET-spec','ET-spec+gate+aug','RootFreqPrior','Random'],evidence=ev)
                np.savez_compressed(folder/'scores.npz',**payload)
                (folder/'run_meta.json').write_text(json.dumps(dict(scores_sha256=sha256(folder/'scores.npz'),mask_sha256={})))
                jobs[component]=dict(status='completed',output=str(folder.relative_to(ROOT)),spec=dict(expected_config=dict(component=component,fold=0,seed=42,smoke=True),test_ids=ids))
            state=dict(jobs=jobs,counts={'completed':2},stage='smoke',status='completed');(root/'queue_state.json').write_text(json.dumps(state))
            output=root/'smoke_result'
            with patch.object(c,'FOLDER',root):c.collect(output)
            result=pd.read_parquet(output/'predictions.parquet')
            self.assertTrue((result[result.method=='ARISE'].rank_of_root==2).all())
            self.assertTrue((result[result.method=='context-only'].rank_of_root==1).all())
            self.assertTrue(result.smoke.all());self.assertFalse((root/'final_evaluation.json').exists())

if __name__=='__main__':unittest.main()
