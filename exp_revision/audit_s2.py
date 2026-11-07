"""Show the unavoidable realized-rate effect of S2 recovery on candidate A0."""

from __future__ import annotations

import json

import numpy as np

from .common import load_bundle
from .missingness import make_mask


def audit(dataset: str, seed: int = 42) -> dict:
    bundle = load_bundle(dataset)
    counts = np.bincount(bundle.availability.sum(-1).astype(int).ravel(), minlength=4)
    rates = {}
    expectations = {}
    for p in (.10, .25, .50, .75):
        removed, total = 0, 0
        for i, row in enumerate(bundle.rows):
            k, _ = make_mask({"dataset": dataset, "incident_id": row["incident_id"]},
                             bundle.availability[i], "S2", f"{p:.2f}", seed)
            removed += int(((bundle.availability[i] == 1) & (k == 0)).sum())
            total += int(bundle.availability[i].sum())
        rates[f"{p:.2f}"] = removed / total
        expectations[f"{p:.2f}"] = float(sum(counts[m] * (m * p - p**m) for m in (1, 2, 3)) / total)
    return {"dataset": dataset, "seed": seed, "A0_source": "nonzero prepared tensor slice; raw audit pending",
            "available_modality_count": counts.tolist(), "realized_rates": rates, "expected_rates_under_restoration": expectations}


def main() -> None:
    print(json.dumps([audit("tt"), audit("aiops")], indent=2))


if __name__ == "__main__": main()
