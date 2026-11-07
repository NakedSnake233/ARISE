"""Root-independent whole-service outages. Legacy targeted masks stay frozen."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import stable_seed

VERSION = "S3v2_all"
RATES = (0.10, 0.25, 0.50)


def make_service_mask(dataset: str, incident_id: str, a0: np.ndarray,
                      fraction: float, seed: int) -> tuple[np.ndarray, dict]:
    """Uniform subsets of observable services; no root or root-derived stratum."""
    a0 = np.asarray(a0)
    if a0.ndim != 2 or a0.shape[1] != 3 or not np.isin(a0, (0, 1)).all():
        raise ValueError("A0 must be binary service-by-three-modality availability")
    if fraction not in RATES:
        raise ValueError(f"fraction must be one of {RATES}")
    eligible = np.flatnonzero(a0.any(axis=1))
    n = len(eligible)
    count = min(n, max(2, int(np.ceil(fraction * n))))
    rng = np.random.default_rng(stable_seed(VERSION, dataset, incident_id,
                                           f"{fraction:.2f}", int(seed)))
    chosen = rng.choice(eligible, size=count, replace=False)
    mask = np.ones(a0.shape, dtype=np.uint8)
    mask[chosen] = 0
    available = int(a0.sum())
    deleted = int((a0 * (1 - mask)).sum())
    return mask, dict(requested_service_fraction=fraction,
        eligible_services=n, deleted_services=count,
        realized_service_fraction=count / n if n else 0.0,
        original_available=available, deleted=deleted,
        realized_rate=deleted / available if available else 0.0,
        insufficient_eligible_services=n < 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", required=True, type=Path)
    args = parser.parse_args()
    mask, rates = make_service_mask("synthetic", "event", np.ones((10, 3), np.uint8), .25, 42)
    args.smoke.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.smoke, mask=mask)
    print(rates)


if __name__ == "__main__":
    main()
