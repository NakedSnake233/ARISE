import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exp_revision.missingness_v2 import RATES, make_service_mask
from exp_revision.scenario_v2 import generate, load_cases, mask_path, rule_scores, top1
from exp_revision.scenario_v2_counts import counts


class ServiceOutageTests(unittest.TestCase):
    def test_zero_inclusive_stratum_counts(self):
        rows = [dict(dataset='tt',scenario_param='0.10',incident_id='a',seed=seed,
                     stratum=stratum,method='fully_missing_first')
                for seed,stratum in ((42,'root_newly_missing'),(43,'root_still_observed'))]
        result = {r['stratum']:r for r in counts(rows)}
        self.assertEqual(result['root_originally_unobserved']['event_seed_rows'],0)
        self.assertEqual(result['all']['events'],1)
        self.assertEqual(result['all']['event_seed_rows'],2)
        self.assertEqual(result['root_newly_missing']['fraction_of_event_seed_rows'],.5)
        with self.assertRaises(ValueError):
            counts(rows+rows)

    def test_independent_of_roots_and_event_order(self):
        self.assertNotIn('root', inspect.signature(make_service_mask).parameters)
        a0 = np.ones((10, 3), np.uint8)
        forward = {key: make_service_mask('ds', key, a0, .25, 42)[0] for key in ('a', 'b')}
        for key in ('b', 'a'):
            np.testing.assert_array_equal(forward[key], make_service_mask('ds', key, a0, .25, 42)[0])
        with self.assertRaises(TypeError):
            make_service_mask('ds', 'a', a0, .25, 42, root=0)
        included = [not make_service_mask('ds', 'a', a0, .25, s)[0][0].any() for s in range(100)]
        self.assertTrue(any(included) and not all(included))

    def test_actual_deletion_and_originally_absent_services(self):
        a0 = np.array([[1,1,1], [1,0,0], [0,0,0], [0,1,1], [0,0,1]], np.uint8)
        for fraction in RATES:
            k, rates = make_service_mask('ds', 'e', a0, fraction, 42)
            self.assertEqual(rates['eligible_services'], 4)
            self.assertEqual(rates['deleted_services'], 2)
            self.assertEqual(rates['realized_service_fraction'], .5)
            self.assertEqual(rates['realized_rate'], (a0 * (1-k)).sum()/a0.sum())
            np.testing.assert_array_equal(k[2], [1,1,1])
            self.assertTrue(np.all((a0*k) <= a0))
            self.assertEqual(np.sum(a0.any(1) & ~(a0*k).any(1)), 2)

    def test_zero_or_one_observable_service(self):
        for n in (0,1):
            a0 = np.zeros((5,3), np.uint8); a0[:n,0] = 1
            k, rates = make_service_mask('ds', 'e', a0, .1, 42)
            self.assertEqual(rates['deleted_services'], n)
            self.assertTrue(rates['insufficient_eligible_services'])
            self.assertEqual(rates['realized_rate'], float(n))
            self.assertFalse((a0*k).any())

    def test_invalid_inputs(self):
        for a0 in (np.ones((3,2)), np.full((3,3), .5), np.full((3,3), 256)):
            with self.assertRaises(ValueError):
                make_service_mask('ds','e',a0,.25,42)
        with self.assertRaises(ValueError):
            make_service_mask('ds','e',np.ones((3,3)),.75,42)

    def test_missing_only_rule_not_guaranteed_correct(self):
        a0 = np.ones((10,3),np.uint8)
        k,_ = make_service_mask('ds','e',a0,.25,42)
        scores = rule_scores(a0,k,np.zeros(10))['fully_missing_first']
        values = [top1(scores,r) for r in range(10)]
        self.assertAlmostEqual(np.mean([r[0] for r in values]), .1)
        self.assertAlmostEqual(np.mean([r[1] for r in values]), .1)
        missing = np.flatnonzero(~k.any(1))
        self.assertEqual(len(missing),3)
        self.assertAlmostEqual(top1(scores,int(missing[0]))[1],1/3)
        self.assertEqual(top1(scores,int(np.flatnonzero(k.any(1))[0]))[1],0.)

    def test_generation_without_labels_and_label_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); splits = root/'splits'; splits.mkdir()
            keys = [f'event{i}' for i in range(5)]
            split = {f'fold_{f}':dict(train=[k for k in keys if k!=keys[f]],test=[keys[f]]) for f in range(5)}
            (splits/'tt_kfold5.json').write_text(json.dumps(split))
            (splits/'tt_unlabeled_events.json').write_text(json.dumps([dict(incident_id=k) for k in keys]))
            for f in range(5):
                folder = root/f'cache/fold_preprocessing/tt/fold{f}'; folder.mkdir(parents=True)
                (folder/'metadata.json').write_text(json.dumps(dict(services=[f's{i}' for i in range(10)],
                    train_ids=split[f'fold_{f}']['train'],test_ids=split[f'fold_{f}']['test'])))
                np.savez_compressed(folder/'features.npz',availability=np.ones((5,10,3),np.uint8))
            output = root/'out'
            self.assertEqual(generate(output,('tt',),root),{'tt':5})
            path = mask_path(output,'tt',.25,42); original = path.read_bytes()
            # Generation succeeded with no label files. Arbitrary evaluation labels
            # added/permuted afterwards cannot alter any mask or manifest.
            label_path = root/'data/care/trainticket-prepared/incidents.json'
            label_path.parent.mkdir(parents=True)
            for roots in ([0,1,2,3,4], [9,8,7,6,5]):
                label_path.write_text(json.dumps([dict(incident_id=key, root_service=f's{r}')
                    for key, r in zip(keys, roots)]))
                generate(output,('tt',),root)
                self.assertEqual(original,path.read_bytes())
            with np.load(path) as archive:
                masks = {key:archive[key].copy() for key in archive.files}
            masks[keys[0]][0] = 1-masks[keys[0]][0]
            np.savez_compressed(path,**masks)
            with self.assertRaises(FileExistsError):
                generate(output,('tt',),root)
            meta_path = root/'cache/fold_preprocessing/tt/fold0/metadata.json'
            meta = json.loads(meta_path.read_text()); meta['test_ids'] = ['wrong']
            meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_cases('tt',root)


if __name__ == '__main__':
    unittest.main()
