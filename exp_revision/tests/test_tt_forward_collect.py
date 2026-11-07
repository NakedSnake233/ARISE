"""Synthetic receipt/aggregation checks; no model training or real result edits."""
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from exp_revision import tt_forward_collect as c
from exp_revision.common import json_dump, sha256


class ForwardCollectionChecks(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, full=False):
        with tempfile.TemporaryDirectory(prefix='PLACEHOLDER_forward_collector_') as directory:
            root = Path(directory); folder = root / 'results'; folder.mkdir()
            split = root / 'split.json'; masks_root = root / 'masks'; masks_root.mkdir()
            services = [f'service{i}' for i in range(41)]
            records = [dict(incident_id=f'event{i:03}', root_service=services[i % 10],
                            fault_type=f'fault{i % 5}',
                            start=float(i * 300), end=float(i * 300 + 240)) for i in range(150)]
            all_ids = [r['incident_id'] for r in records]; ids = all_ids[90:]
            part = dict(train=all_ids[:90], test=ids, split='chronological_90_60')
            json_dump(split, {'fold_0': part}); json_dump(root / 'incidents.json', records)
            original = np.ones((150, 41, 3), dtype=np.uint8)
            files = {}; mask_hashes = {}
            for seed in sorted(c.SEEDS):
                mask_hashes[seed] = {}
                for count, scene in enumerate(c.SCENES[1:], 1):
                    path = c.mask_path(masks_root, 'tt', float(scene.rsplit('_', 1)[1]), seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    values = original.copy(); values[:, :count * 4] = 0
                    np.savez_compressed(path, **dict(zip(all_ids, values)))
                    files[str(path.relative_to(masks_root))] = sha256(path)
                    mask_hashes[seed][str(path.relative_to(root))] = sha256(path)
            json_dump(masks_root / 'masks_manifest.json', {'datasets': {'tt': {'services': services, 'files': files}}})
            jobs = {}; completed = {'forward_prepare', 'forward_context_s42', 'forward_experts_s42'}
            for name, (component, method, seed) in c.expected_jobs().items():
                cfg = dict(task='TFWD', dataset='tt', fold=0, smoke=False, component=component,
                           split='chronological_90_60')
                if seed is not None:
                    cfg.update(seed=seed, scenarios=c.SCENES)
                if method:
                    cfg['method'] = method
                shape = [1, 60, 41] if seed is None else [4, 4, 60, 41] if component == 'experts' else [4, 60, 41]
                status = 'completed_generalization_' + ('preprocessing' if seed is None else component)
                spec = dict(expected_config=cfg, score_shape=shape, test_ids=ids, expected_status=[status])
                row = dict(spec=spec, status='completed' if full or name in completed else 'pending')
                jobs[name] = row
                if row['status'] != 'completed':
                    continue
                out = folder / name; out.mkdir(); row['output'] = str(out.relative_to(root))
                ctx = np.zeros((4, 60, 41)); evidence = np.zeros_like(ctx)
                for i, record in enumerate(records[90:]):
                    target = services.index(record['root_service']); other = (target + 1) % 41
                    ctx[:, i, target] = .95; ctx[:, i, other] = .05
                    evidence[:, i, target] = .05; evidence[:, i, other] = .95
                values = dict(incident_ids=ids)
                if seed is None:
                    values['scores'] = original[90:, :, 0][None].astype(float)
                else:
                    values.update(scenario_keys=c.SCENES, scores=ctx)
                if component == 'experts':
                    values.update(scores=np.stack([evidence] * 4), evidence=evidence, method_names=c.EXPERTS)
                np.savez_compressed(out / 'scores.npz', **values)
                meta = dict(config=cfg, status=status, errors=[], start_time='synthetic-start', end_time='synthetic-end',
                            scores_sha256=sha256(out / 'scores.npz'), input_sha256={str(split.relative_to(root)): sha256(split)})
                if seed is None:
                    np.savez_compressed(out / 'features.npz', availability=original)
                    json_dump(out / 'metadata.json', dict(train_ids=part['train'], test_ids=ids, services=services))
                    meta.update(features_sha256=sha256(out / 'features.npz'), metadata_sha256=sha256(out / 'metadata.json'))
                else:
                    meta['mask_sha256'] = mask_hashes[seed]
                json_dump(out / 'run_meta.json', meta)
            state = dict(jobs=jobs, status='completed' if full else 'running')
            json_dump(folder / 'queue_state.json', state)
            source = root / 'synthetic_source.txt'; source.write_text('Synthetic fixture only')
            json_dump(folder / 'queue_manifest.json', dict(jobs={name: row['spec'] for name, row in jobs.items()},
                source_sha256={str(source.relative_to(root)): sha256(source)}))
            with patch.object(c, 'ROOT', root), patch.object(c, 'FOLDER', folder), patch.object(c, 'SPLIT', split), \
                    patch.object(c, 'MASK_ROOT', masks_root), patch.object(c, 'dataset_dir', return_value=root), \
                    patch.object(c, 'load_cases', return_value=(services, [(0, key, a) for key, a in zip(all_ids, original)], {})):
                yield root, folder, state

    def test_partial_fusion_and_actual_deletion(self):
        with self.fixture() as (root, folder, _):
            output = root / 'report'; c.collect(output)
            frame = pd.read_parquet(output / 'predictions.parquet')
            self.assertTrue((frame[frame.method == 'context-only'].rank_of_root == 1).all())
            self.assertTrue((frame[frame.method == 'ARISE'].rank_of_root == 2).all())
            metrics = pd.read_csv(output / 'metrics.csv')
            self.assertEqual(set(metrics.coverage), {'partial'})
            selected = metrics[(metrics.method == 'ARISE') & (metrics.scenario_param == '0.10') & (metrics.stratum == 'all')].iloc[0]
            self.assertAlmostEqual(selected.actual_cell_deletion_rate, 4 / 41)
            self.assertAlmostEqual(selected.actual_service_deletion_fraction, 4 / 41)
            self.assertFalse((folder / 'final_evaluation.json').exists())

    def test_complete_coverage_adapted_names_and_publication(self):
        with self.fixture(full=True) as (root, folder, _):
            output = root / 'report'; c.collect(output)
            frame = pd.read_parquet(output / 'predictions.parquet')
            self.assertEqual(len(frame), 9 * 4 * 60 * 5)
            self.assertEqual(set(frame.method), c.EXPECTED_METHODS)
            self.assertEqual(set(frame.groupby(['method', 'scenario_param']).size()), {300})
            self.assertTrue((folder / 'final_evaluation.json').exists())
            self.assertEqual(json.loads((output / 'receipt.json').read_text())['independent_test_events'], 60)

    def test_terminal_incomplete_cannot_publish(self):
        with self.fixture() as (root, folder, state):
            state['status'] = 'completed'; json_dump(folder / 'queue_state.json', state)
            with self.assertRaisesRegex(ValueError, 'terminal queue has incomplete'):
                c.collect(root / 'report')
            self.assertFalse((folder / 'final_evaluation.json').exists())

    def test_nonfinite_evidence_rejected_even_when_primary_scores_valid(self):
        with self.fixture() as (root, folder, _):
            artifact = folder / 'forward_experts_s42'; path = artifact / 'scores.npz'
            with np.load(path) as z:
                arrays = {key: z[key].copy() for key in z.files}
            arrays['evidence'][0, 0, 0] = np.nan
            np.savez_compressed(path, **arrays)
            receipt = artifact / 'run_meta.json'; meta = json.loads(receipt.read_text())
            meta['scores_sha256'] = sha256(path); json_dump(receipt, meta)
            with self.assertRaisesRegex(ValueError, 'invalid evidence branch'):
                c.collect(root / 'report')

    def test_reordered_ids_rejected(self):
        with self.fixture() as (root, folder, _):
            artifact = folder / 'forward_context_s42'; path = artifact / 'scores.npz'
            with np.load(path) as z:
                arrays = {key: z[key].copy() for key in z.files}
            arrays['incident_ids'] = arrays['incident_ids'][::-1]
            np.savez_compressed(path, **arrays)
            receipt = artifact / 'run_meta.json'; meta = json.loads(receipt.read_text())
            meta['scores_sha256'] = sha256(path); json_dump(receipt, meta)
            with self.assertRaisesRegex(ValueError, 'unverified completed artifact'):
                c.collect(root / 'report')

    def test_changed_frozen_source_rejected(self):
        with self.fixture() as (root, _, _):
            (root / 'synthetic_source.txt').write_text('changed source')
            with self.assertRaisesRegex(ValueError, 'frozen source/configuration changed'):
                c.collect(root / 'report')

    def test_changed_queue_spec_rejected(self):
        with self.fixture() as (root, folder, state):
            state['jobs']['forward_context_s42']['spec']['score_shape'] = [4, 59, 41]
            json_dump(folder / 'queue_state.json', state)
            with self.assertRaisesRegex(ValueError, 'queue specifications differ'):
                c.collect(root / 'report')


if __name__ == '__main__':
    unittest.main()
