import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from exp_revision import s3v2_queue as queue, s3v2_official_parity as parity
from exp_revision.common import sha256

class OfficialChecks(unittest.TestCase):
    def test_later_fold_failure_blocks_only_same_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'fold1';folder.mkdir();(folder/'run_meta.json').write_text(json.dumps({'scientific_stop':True}))
            def row(ds,method,status):return dict(status=status,spec=dict(expected_config=dict(dataset=ds,method=method,component='official')))
            state={'jobs':{'done':row('tt','RCD','completed'),'same':row('tt','RCD','pending'),'other_ds':row('aiops','RCD','pending'),'other_method':row('tt','BARO','pending'),'inflight':row('tt','RCD','running')}}
            state['jobs']['done']['output']='fold1'
            with patch.object(queue,'ROOT',root):queue.refresh_scientific_blocks(state)
            self.assertEqual(state['jobs']['same']['status'],'scientific_blocked')
            self.assertEqual(state['jobs']['other_ds']['status'],'pending')
            self.assertEqual(state['jobs']['other_method']['status'],'pending')
            self.assertEqual(state['jobs']['inflight']['status'],'running')
    def test_circa_historical_name_and_smoke_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'saved';folder.mkdir()
            scores=np.array([[[1.,0.]]]);failed=np.zeros((1,1),bool)
            np.savez_compressed(folder/'scores.npz',scores=scores,failed=failed,incident_ids=['a'])
            (folder/'run_meta.json').write_text(json.dumps({'scores_sha256':sha256(folder/'scores.npz')}))
            state={'jobs':{'full_official_tt_CIRCAplusrankprune_f0_s42':dict(status='completed',output='saved'),'smoke_official_re2_ss_RCD_f0_s42':dict(status='completed',output='saved')}}
            p=root/'results_v2/remaining_queue';p.mkdir(parents=True);(p/'queue_state.json').write_text(json.dumps(state))
            with patch.object(parity,'ROOT',root):
                self.assertEqual(parity.check('tt',0,42,'CIRCA+rank-prune',['a'],scores,failed)['scope'],'full_reference')
                new=np.array([[[0.,1.],[1.,0.]]]);flags=np.zeros((1,2),bool)
                self.assertEqual(parity.check('re2_ss',0,42,'RCD',['b','a'],new,flags)['compared_events'],1)
                with self.assertRaises(ValueError):parity.check('re2_ss',0,42,'RCD',['a','b'],new,flags)
if __name__=='__main__':unittest.main()
