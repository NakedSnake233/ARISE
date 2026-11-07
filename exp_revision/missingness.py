from __future__ import annotations

import numpy as np

from .common import stable_seed


SCENARIOS = [("S0", ""), ("S1", "metric"), ("S1", "log"), ("S1", "trace")]
SCENARIOS += [("S2", f"{p:.2f}") for p in (0.10, 0.25, 0.50, 0.75)]
SCENARIOS += [("S3", x) for x in ("trace", "all")]
SCENARIOS += [("S3c", x) for x in ("trace", "all")]


def make_mask(incident: str | dict, a0: np.ndarray, scenario: str, param: str, model_seed: int, root: int | None = None) -> tuple[np.ndarray, float]:
    """Return K and realized deletion rate using only the declared scenario inputs."""
    a0 = np.asarray(a0, dtype=np.uint8)
    if a0.ndim != 2 or a0.shape[1] != 3:
        raise ValueError("A0 must have shape Sx3")
    k = np.ones_like(a0, dtype=np.uint8)
    incident_id = incident["incident_id"] if isinstance(incident, dict) else incident
    dataset = incident.get("dataset", "") if isinstance(incident, dict) else ""
    rng = np.random.default_rng(stable_seed(dataset, incident_id, scenario, param, model_seed))
    if scenario == "S0":
        pass
    elif scenario == "S1":
        k[:, {"metric": 0, "log": 1, "trace": 2}[param]] = 0
    elif scenario == "S2":
        p = float(param)
        if not 0 <= p <= 1:
            raise ValueError(param)
        k[(a0 == 1) & (rng.random(a0.shape) < p)] = 0
        # The recovery rule is applied per service, only when it had telemetry.
        for service in range(a0.shape[0]):
            available = np.flatnonzero(a0[service] == 1)
            if len(available) and not np.any(k[service, available]):
                k[service, int(rng.choice(available))] = 1
    elif scenario in {"S3", "S3c"}:
        if root is None:
            raise ValueError(f"{scenario} requires root for mask generation")
        target = int(root)
        if scenario == "S3c":
            candidates = np.flatnonzero(np.arange(a0.shape[0]) != target)
            if param == "trace":
                candidates = candidates[a0[candidates, 2] == 1]
            else:
                candidates = candidates[np.any(a0[candidates] == 1, axis=1)]
            if not len(candidates):
                return k, 0.0
            target = int(rng.choice(candidates))
        if param == "trace":
            k[target, 2] = 0
        elif param == "all":
            k[target, :] = 0
        else:
            raise ValueError(param)
    elif scenario == "S4":
        raise NotImplementedError("S4 needs trace_id sampling and complete span-derived feature recomputation; this path is not implemented yet")
    else:
        raise ValueError(f"unknown scenario {scenario}")
    denom = max(int(a0.sum()), 1)
    realized = float(np.sum((a0 == 1) & (k == 0)) / denom)
    return k, realized


def generate_masks(bundle, output, scenarios=SCENARIOS, seeds=(42, 43, 44, 45, 46)):
    import json
    from pathlib import Path
    output = Path(output)
    for scenario, param in scenarios:
        folder = output / bundle.dataset / f"{scenario}_{param or 'none'}"
        folder.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            masks, rates = {}, {}
            for i, row in enumerate(bundle.rows):
                root = int(bundle.labels[i]) if scenario in {"S3", "S3c"} else None
                mask, rate = make_mask({"dataset": bundle.dataset, "incident_id": row["incident_id"]}, bundle.availability[i], scenario, param, seed, root)
                masks[row["incident_id"]] = mask
                rates[row["incident_id"]] = rate
            np.savez_compressed(folder / f"seed{seed}.npz", **{k: v for k, v in masks.items()})
            (folder / f"seed{seed}.rates.json").write_text(json.dumps(rates, indent=2, sort_keys=True) + "\n")
