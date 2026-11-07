import tempfile
import unittest
from pathlib import Path
import numpy as np
from exp_revision.s3v2_infer import reference_s0,SCENES
from exp_revision.s3v2_queue import spec

class InferenceChecks(unittest.TestCase):
    def test_reference_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            np.savez_compressed(p/'scores.npz',scores=np.arange(12).reshape(2,2,3),scenario_keys=['S0_none','old'],incident_ids=['b','a'])
            np.testing.assert_array_equal(reference_s0(p,['a','b']),[[3,4,5],[0,1,2]])
    def test_method_axis_and_no_old_missing_scenes(self):
        self.assertEqual(SCENES,['S0_none','S3v2_all_0.10','S3v2_all_0.25','S3v2_all_0.50'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            np.savez_compressed(p/'scores.npz',scores=np.arange(24).reshape(2,2,2,3),method_names=['a','b'],scenario_keys=['old','S0_none'],incident_ids=['x','y'])
            np.testing.assert_array_equal(reference_s0(p,['y'],'b'),[[21,22,23]])
    def test_plan_scope(self):
        self.assertEqual(spec('tt',0,42)['score_shape'][0],35)
        self.assertEqual(spec('tt',1,42)['score_shape'][0],30)
        self.assertEqual(spec('re2_tt',0,42)['score_shape'][0],13)
        self.assertEqual(spec('aiops',0,42)['resource'],'cpu')
        self.assertEqual(spec('tt',0,42)['resource'],'gpu')
if __name__=='__main__':unittest.main()
