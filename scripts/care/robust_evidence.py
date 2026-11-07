"""Reliability-aware evidence scoring for CARE reviewer-response experiments.

The feature transforms are identity-free: they consume telemetry contrasts and
availability only.  Fault types and root labels are used solely to choose
training/evaluation rows and to fit the supervised service ranker.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from evidence_reranker import fit, model_for, predict


MODALITIES = ("metric", "log", "trace")


def fault_type_holdout(
    rows: Sequence[dict],
    candidate_indices: Sequence[int],
    fault_type: str,
    *,
    held_scope_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[object]]:
    """Split a type holdout and purge telemetry-overlapping fit incidents.

    ``held_scope_indices`` controls which held-type incidents define the
    overlap groups.  Outer folds pass all dataset rows so a training incident
    that overlaps a held test event is removed.  Inner selection folds leave
    it unset, limiting the grouping operation to their training-only cohort.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    scope = (
        candidate_indices
        if held_scope_indices is None
        else np.asarray(held_scope_indices, dtype=int)
    )
    held_groups = {
        rows[index].get("overlap_group")
        for index in scope
        if rows[index]["fault_type"] == fault_type
        and rows[index].get("overlap_group") is not None
    }
    held = candidate_indices[
        [rows[index]["fault_type"] == fault_type for index in candidate_indices]
    ]
    overlap_purged = candidate_indices[
        [
            rows[index]["fault_type"] != fault_type
            and rows[index].get("overlap_group") in held_groups
            for index in candidate_indices
        ]
    ]
    fit = candidate_indices[
        [
            rows[index]["fault_type"] != fault_type
            and rows[index].get("overlap_group") not in held_groups
            for index in candidate_indices
        ]
    ]
    return fit, held, overlap_purged, sorted(held_groups, key=str)


def feature_groups(names: Sequence[str]) -> dict[str, np.ndarray]:
    """Return masks for observation modalities and graph-derived evidence."""
    names = list(names)
    result = {
        "metric": np.asarray(
            [n.startswith(("metric:", "raw_metric:")) for n in names], dtype=bool
        ),
        "log": np.asarray([n.startswith("log:") for n in names], dtype=bool),
        # RPC summaries are trace-derived and disappear with tracing.
        "trace": np.asarray(
            [
                n.startswith(("trace:", "rpc:", "callers:", "callees:"))
                for n in names
            ],
            dtype=bool,
        ),
        "rpc": np.asarray([n.startswith("rpc:") for n in names], dtype=bool),
        "call_graph": np.asarray(
            [n.startswith(("callers:", "callees:")) for n in names], dtype=bool
        ),
        "graph": np.asarray(
            [n.startswith(("rpc:", "callers:", "callees:")) for n in names],
            dtype=bool,
        ),
    }
    # Some cached descriptors aggregate other services/endpoints.  They must
    # disappear for the whole incident if any contributing observation is
    # missing; otherwise a masked service can leak through the cached summary.
    result["metric_cross_service"] = np.asarray(
        [
            n.startswith(("metric:", "raw_metric:"))
            and any(
                marker in f":{n}:"
                for marker in (":relative_shock:", ":shock_share:")
            )
            for n in names
        ],
        dtype=bool,
    )
    result["trace_cross_service"] = np.asarray(
        [
            n.startswith("trace:")
            and any(
                marker in f":{n}:"
                for marker in (":relative_shock:", ":shock_share:")
            )
            for n in names
        ],
        dtype=bool,
    )
    result["joint_neighbor_shock"] = np.asarray(
        [
            n.startswith(("callers:", "callees:"))
            and not n.endswith(":degree")
            for n in names
        ],
        dtype=bool,
    )
    return result


def all_available(shape: Sequence[int]) -> np.ndarray:
    """Create an [incident, service, modality] availability tensor."""
    return np.ones((int(shape[0]), int(shape[1]), len(MODALITIES)), dtype=np.float32)


def apply_availability(
    x: np.ndarray,
    names: Sequence[str],
    availability: np.ndarray | None = None,
    *,
    append_flags: bool = True,
) -> np.ndarray:
    """Zero unavailable modality features and optionally append availability flags."""
    x = np.asarray(x, dtype=np.float32)
    availability = all_available(x.shape) if availability is None else np.asarray(
        availability, dtype=np.float32
    )
    expected = (*x.shape[:2], len(MODALITIES))
    if availability.shape != expected:
        raise ValueError(f"availability shape {availability.shape} != {expected}")
    if not np.isin(availability, [0.0, 1.0]).all():
        raise ValueError("availability must contain only zero/one values")
    groups = feature_groups(names)
    masked = x.copy()
    for index, modality in enumerate(MODALITIES):
        masked[:, :, groups[modality]] *= availability[:, :, index, None]
    incident_metric = availability[:, :, MODALITIES.index("metric")].min(1)
    incident_trace = availability[:, :, MODALITIES.index("trace")].min(1)
    masked[:, :, groups["metric_cross_service"]] *= incident_metric[:, None, None]
    masked[:, :, groups["trace_cross_service"]] *= incident_trace[:, None, None]
    # RPC values depend on both endpoints, and neighbor shock combines metric
    # and trace values across the incident.  Without raw recomputation, global
    # removal is the only transformation that guarantees no hidden observation.
    masked[:, :, groups["rpc"]] *= incident_trace[:, None, None]
    masked[:, :, groups["call_graph"]] *= incident_trace[:, None, None]
    joint_available = incident_metric * incident_trace
    masked[:, :, groups["joint_neighbor_shock"]] *= joint_available[:, None, None]
    if append_flags:
        masked = np.concatenate([masked, availability], axis=-1)
    return np.nan_to_num(masked).astype(np.float32)


def complete_modality_availability(shape: Sequence[int], missing: str | None) -> np.ndarray:
    availability = all_available(shape)
    if missing is not None:
        availability[:, :, MODALITIES.index(missing)] = 0
    return availability


def local_missing_availability(
    shape: Sequence[int],
    rate: float,
    seed: int,
    modality: str | None = None,
) -> np.ndarray:
    """Mask event-service observations while leaving at least one modality available."""
    if not 0 <= rate <= 1:
        raise ValueError("missing rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    availability = all_available(shape)
    indices = range(len(MODALITIES)) if modality is None else [MODALITIES.index(modality)]
    for index in indices:
        availability[:, :, index] = (rng.random(shape[:2]) >= rate).astype(np.float32)
    if modality is None:
        none = availability.sum(-1) == 0
        if none.any():
            restore = rng.integers(0, len(MODALITIES), size=int(none.sum()))
            rows, services = np.where(none)
            availability[rows, services, restore] = 1
    return availability


def aggregate_type_agnostic(
    x: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Remove KPI identities by aggregating each descriptor across channels."""
    x = np.asarray(x, dtype=np.float32)
    output: list[np.ndarray] = []
    output_names: list[str] = []
    collapsed = {"metric", "raw_metric", "trace"}
    for prefix in sorted(collapsed):
        descriptors: dict[str, list[int]] = collections.defaultdict(list)
        for index, name in enumerate(names):
            parts = name.split(":")
            if len(parts) >= 3 and parts[0] == prefix:
                descriptors[parts[-1]].append(index)
        for descriptor, indices in sorted(descriptors.items()):
            values = x[:, :, indices]
            statistics = {
                "max": values.max(-1),
                "mean": values.mean(-1),
                "q90": np.quantile(values, 0.9, axis=-1),
                "std": values.std(-1),
                "absmax": np.abs(values).max(-1),
            }
            for statistic, values_out in statistics.items():
                output.append(values_out)
                output_names.append(f"{prefix}:ALL:{descriptor}:{statistic}")
    for index, name in enumerate(names):
        if name.split(":", 1)[0] not in collapsed:
            output.append(x[:, :, index])
            output_names.append(name)
    if not output:
        raise ValueError("No evidence features available")
    return np.stack(output, axis=-1).astype(np.float32), output_names


@dataclass
class RobustRanker:
    model: object
    names: list[str]
    append_flags: bool
    type_agnostic: bool
    training_rows: list[int]
    augmentation_rows: int

    def predict(
        self, x: np.ndarray, availability: np.ndarray | None = None
    ) -> np.ndarray:
        transformed, names = (
            aggregate_type_agnostic(x, self.names)
            if self.type_agnostic
            else (np.asarray(x, dtype=np.float32), self.names)
        )
        return predict(
            self.model,
            apply_availability(
                transformed, names, availability, append_flags=self.append_flags
            ),
        )


def _augmentation_availabilities(
    shape: Sequence[int], seed: int, partial_rates: Iterable[float]
) -> list[np.ndarray]:
    result = [complete_modality_availability(shape, None)]
    result.extend(complete_modality_availability(shape, modality) for modality in MODALITIES)
    result.extend(local_missing_availability(shape, rate, seed + i) for i, rate in enumerate(partial_rates))
    return result


def fit_robust_ranker(
    x: np.ndarray,
    names: Sequence[str],
    labels: np.ndarray,
    train_indices: Sequence[int],
    *,
    seed: int = 42,
    augment: bool = True,
    append_flags: bool = True,
    type_agnostic: bool = False,
    partial_rates: Sequence[float] = (0.25, 0.5, 0.75),
    trees: int = 256,
    min_samples_leaf: int = 3,
    depth: int | None = None,
) -> RobustRanker:
    train_indices = np.asarray(train_indices, dtype=int)
    labels = np.asarray(labels)
    if not len(train_indices):
        raise ValueError("No training incidents")
    transformed, transformed_names = (
        aggregate_type_agnostic(x, names)
        if type_agnostic
        else (np.asarray(x, dtype=np.float32), list(names))
    )
    train_x = transformed[train_indices]
    availabilities = (
        _augmentation_availabilities(train_x.shape, seed, partial_rates)
        if augment
        else [complete_modality_availability(train_x.shape, None)]
    )
    augmented = np.concatenate(
        [
            apply_availability(
                train_x, transformed_names, availability, append_flags=append_flags
            )
            for availability in availabilities
        ],
        axis=0,
    )
    augmented_labels = np.tile(labels[train_indices], len(availabilities))
    model = model_for(depth)
    model.set_params(
        random_state=seed,
        n_estimators=trees,
        min_samples_leaf=min_samples_leaf,
    )
    fit(model, augmented, augmented_labels)
    return RobustRanker(
        model=model,
        # The original names are needed to reproduce the generic transform.
        names=list(names),
        append_flags=append_flags,
        type_agnostic=type_agnostic,
        training_rows=train_indices.tolist(),
        augmentation_rows=len(augmented),
    )


def fit_fixed_availability_ranker(
    x: np.ndarray,
    names: Sequence[str],
    labels: np.ndarray,
    train_indices: Sequence[int],
    availability: np.ndarray,
    *,
    seed: int = 42,
    append_flags: bool = True,
    type_agnostic: bool = False,
    trees: int = 256,
    min_samples_leaf: int = 3,
    depth: int | None = None,
) -> RobustRanker:
    """Fit a modality-ablation model under one fixed availability pattern."""
    train_indices = np.asarray(train_indices, dtype=int)
    availability = np.asarray(availability, dtype=np.float32)
    if availability.shape != (*np.asarray(x).shape[:2], len(MODALITIES)):
        raise ValueError("Fixed availability does not align with evidence tensor")
    transformed, transformed_names = (
        aggregate_type_agnostic(x, names)
        if type_agnostic
        else (np.asarray(x, dtype=np.float32), list(names))
    )
    model = model_for(depth)
    model.set_params(
        random_state=seed,
        n_estimators=trees,
        min_samples_leaf=min_samples_leaf,
    )
    fit(
        model,
        apply_availability(
            transformed[train_indices],
            transformed_names,
            availability[train_indices],
            append_flags=append_flags,
        ),
        np.asarray(labels)[train_indices],
    )
    return RobustRanker(
        model=model,
        names=list(names),
        append_flags=append_flags,
        type_agnostic=type_agnostic,
        training_rows=train_indices.tolist(),
        augmentation_rows=len(train_indices),
    )


def normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.maximum(np.asarray(scores, dtype=np.float64), 0)
    return scores / np.maximum(scores.sum(-1, keepdims=True), 1e-12)


def fuse_all(*weighted_scores: tuple[float, np.ndarray]) -> np.ndarray:
    """Convex fusion over all services, without a base Top-k mask."""
    if not weighted_scores:
        raise ValueError("At least one score matrix is required")
    total_weight = sum(float(weight) for weight, _ in weighted_scores)
    if total_weight <= 0 or any(weight < 0 for weight, _ in weighted_scores):
        raise ValueError("Fusion weights must be nonnegative and sum to a positive value")
    shape = np.asarray(weighted_scores[0][1]).shape
    if any(np.asarray(scores).shape != shape for _, scores in weighted_scores):
        raise ValueError("All score matrices must have the same shape")
    combined = sum(float(weight) * normalize(scores) for weight, scores in weighted_scores)
    return normalize(combined)


def mask_feature_group(
    x: np.ndarray,
    names: Sequence[str],
    group: str,
    retention: float,
    seed: int,
) -> np.ndarray:
    """Deterministically retain graph evidence at the event-service level."""
    if not 0 <= retention <= 1:
        raise ValueError("retention must be in [0, 1]")
    result = np.asarray(x, dtype=np.float32).copy()
    feature_mask = feature_groups(names)[group]
    keep = np.random.default_rng(seed).random(result.shape[:2]) < retention
    result[:, :, feature_mask] *= keep[:, :, None]
    return result
