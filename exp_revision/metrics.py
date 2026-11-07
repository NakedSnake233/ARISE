from __future__ import annotations

import numpy as np


def ranking_metrics(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=float)
    finite = np.isfinite(ranks) & (ranks >= 1)
    reciprocal = np.divide(1.0, ranks, out=np.zeros_like(ranks), where=finite)
    return {
        "PR@1": float(np.mean(finite & (ranks <= 1))),
        "PR@3": float(np.mean(finite & (ranks <= 3))),
        "PR@5": float(np.mean(finite & (ranks <= 5))),
        "MAP@3": float(np.mean(np.where(finite & (ranks <= 3), reciprocal, 0.0))),
        "MAP@5": float(np.mean(np.where(finite & (ranks <= 5), reciprocal, 0.0))),
        "MRR": float(np.mean(reciprocal)),
    }
