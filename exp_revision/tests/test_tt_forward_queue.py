"""Budget interruption and pilot-admission invariants for the forward queue."""
import unittest
from unittest.mock import patch
from exp_revision import tt_forward_queue as queue


class ForwardQueueTests(unittest.TestCase):
    def row(self, status, attempts, resource='gpu', cores=4):
        return dict(status=status, attempts=attempts,
                    spec=dict(resource=resource, cores=cores, timeout_seconds=1800, max_attempts=2))

    def test_running_deadline_and_unused_retry_are_both_reserved(self):
        state = dict(jobs={'x': self.row('running', [dict(start_epoch=0)])})
        self.assertAlmostEqual(queue.maximum_remaining(state), 2*1810*4/3600)
        self.assertAlmostEqual(queue.maximum_remaining(state, True), 2*1810/3600)
        # Actual failed work remains charged in addition to the unused retry.
        state['jobs']['x'] = self.row('pending', [dict(elapsed_seconds=900)])
        self.assertAlmostEqual(queue.prior.charged(state) + queue.maximum_remaining(state), (900+1810)*4/3600)

    def test_completed_work_does_not_reserve_unused_retry(self):
        state = dict(jobs={'x': self.row('completed', [dict(elapsed_seconds=900)])})
        self.assertEqual(queue.maximum_remaining(state), 0)
        self.assertAlmostEqual(queue.prior.charged(state), 1.)
        self.assertAlmostEqual(queue.prior.charged(state, gpu=True), .25)

    def test_bad_receipt_blocks_expansion_even_with_completed_status(self):
        jobs = {queue.identifier(c, 42, m): dict(status='completed', output='unused', spec={})
                for c, m in queue.COMPONENTS}
        with patch.object(queue.q, 'verified', return_value=False):
            self.assertFalse(queue.pilot_ready(dict(jobs=jobs)))
        with patch.object(queue.q, 'verified', return_value=True):
            self.assertTrue(queue.pilot_ready(dict(jobs=jobs)))

    def test_fixed_matrix_fits_existing_remaining_allowance(self):
        jobs = [queue.specification('prepare')]
        jobs += [queue.specification(c, s, m) for s in range(42, 47) for c, m in queue.COMPONENTS]
        self.assertEqual(len({j['id'] for j in jobs}), 26)
        self.assertEqual(sum(j['resource']=='gpu' for j in jobs), 5)
        self.assertTrue(all(len(j['test_ids']) == 60 and j['expected_config']['dataset']=='tt' for j in jobs))
        state = dict(jobs={j['id']: dict(spec=j, status='pending', attempts=[]) for j in jobs})
        self.assertAlmostEqual(queue.maximum_remaining(state), 39.31111111111111)
        self.assertAlmostEqual(queue.maximum_remaining(state, True), 5.027777777777778)
        self.assertIn('--prepare', jobs[0]['command'])


if __name__ == '__main__':
    unittest.main()
