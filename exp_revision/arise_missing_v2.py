"""ARISE-R1: low-capacity availability-conditioned expert mixture.

No labels or fault types are accepted by inference. Synthetic entry is only a
plumbing check; it is never a benchmark result.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

EXPERTS = ('clean_specific', 'robust_specific', 'robust_generic', 'context')


def probabilities(scores):
    p = np.asarray(scores, dtype=float)
    if p.ndim < 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('invalid expert scores')
    total = p.sum(-1, keepdims=True)
    return np.divide(p, total, out=np.full_like(p, 1 / p.shape[-1]), where=total > 0)


def regimes(availability):
    a = np.asarray(availability)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isin(a, (0, 1)).all():
        raise ValueError('expected binary event/service/modality availability')
    missing = (a.sum(1) == 0).astype(int) @ np.asarray([1, 2, 4])
    band = np.minimum((a.mean((1, 2)) * 3).astype(int), 2)
    return missing * 3 + band


def simplex_grid():
    # Stable ties prefer clean expert, then robust-specific, robust-generic.
    return np.asarray([v for v in itertools.product(range(4, -1, -1), repeat=4) if sum(v) == 4], dtype=float) / 4


def fit_router(scores, availability, labels, event_ids):
    p = probabilities(scores)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 3 or p.shape[1] != len(EXPERTS) or len(y) != len(p) or len(event_ids) != len(p):
        raise ValueError('calibration shape mismatch')
    if (y < 0).any() or (y >= p.shape[-1]).any(): raise ValueError('invalid calibration labels')
    r = regimes(availability)
    if len(r) != len(y): raise ValueError('calibration availability mismatch')
    grid = simplex_grid()
    truth_prob = p[np.arange(len(y)), :, y]
    loss = -np.log(np.maximum(truth_prob @ grid.T, 1e-6))
    event_ids = np.asarray(event_ids)
    # Count incidents equally even when a regime contains unequal numbers of views.
    def event_mean(values, ids):
        return np.stack([values[ids == k].mean(0) for k in sorted(set(ids.tolist()))]).mean(0)
    global_loss = event_mean(loss, event_ids)
    global_index = int(np.argmin(global_loss))
    selected = {}
    for key in sorted(set(r.tolist())):
        use = r == key
        count = len(set(np.asarray(event_ids)[use].tolist()))
        local_loss = (count * event_mean(loss[use], event_ids[use]) + 8 * global_loss) / (count + 8)
        index = int(np.argmin(local_loss))
        selected[str(key)] = dict(weights=grid[index].tolist(), unique_events=count,
            observations=int(use.sum()), objective=float(local_loss[index]))
    return dict(method='ARISE-R1', experts=list(EXPERTS), grid_step=.25, pseudo_events=8,
        global_weights=grid[global_index].tolist(), regimes=selected,
        objective='negative log likelihood; event-regime loss shrunk to global',
        calibration_events=len(set(event_ids)))


def infer_router(scores, availability, router):
    p = probabilities(scores); r = regimes(availability)
    if p.ndim != 3 or p.shape[1] != 4 or len(r) != len(p): raise ValueError('inference shape mismatch')
    weights = np.asarray([router['regimes'].get(str(k), {}).get('weights', router['global_weights']) for k in r])
    if weights.shape != (len(p), 4) or (weights < 0).any() or not np.allclose(weights.sum(1), 1):
        raise ValueError('invalid mixture weights')
    result = np.einsum('nes,ne->ns', p, weights)
    result[np.asarray(availability).sum((1, 2)) == 0] = 1 / p.shape[-1]
    return probabilities(result), weights


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--synthetic-smoke', type=Path, required=True)
    args = parser.parse_args()
    a = np.ones((4, 3, 3), dtype=np.uint8); p = np.tile(np.eye(3)[[0, 1, 2, 0]][:, None], (1, 4, 1))
    router = fit_router(p, a, [0, 1, 2, 0], ['a', 'b', 'c', 'd'])
    out, _ = infer_router(p, a, router)
    assert np.array_equal(out.argmax(1), [0, 1, 2, 0])
    args.synthetic_smoke.write_text(json.dumps(dict(status='synthetic_plumbing_only', benchmark_result=False, router=router)))


if __name__ == '__main__': main()
