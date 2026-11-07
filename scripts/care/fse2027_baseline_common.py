"""Shared, label-safe utilities for the FSE 2027 matched-baseline study.

The common evaluator is deliberately separate from every method adapter.  Each
adapter writes a complete incident-by-service score matrix; only this module
turns that matrix into paper metrics.  Stable service-order tie breaking is part
of the frozen protocol.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/care/fse2027-baselines"

DATASETS = {
    "trainticket": ROOT / "data/care/trainticket-prepared",
    "aiops2022": ROOT / "data/care/aiops2022-paper128-prepared",
}

METRICS = ("PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


@dataclasses.dataclass
class DatasetBundle:
    name: str
    path: Path
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    features: dict[str, np.ndarray]
    services: list[str]
    labels: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    fit: np.ndarray
    test: np.ndarray
    split_bin: int

    @property
    def service_count(self) -> int:
        return len(self.services)


def _aiops_inner_split(rows: list[dict[str, Any]], train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic, overlap-group-safe validation split from training.

    This split is used only for baseline adapter selection.  The external test
    set remains untouched.  GroupShuffleSplit is deterministic for the fixed
    manifest and seed; the chosen incident ids are serialized in the protocol.
    """

    groups = np.asarray([str(rows[i].get("overlap_group", rows[i]["incident_id"])) for i in train])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2027)
    fit_pos, validation_pos = next(splitter.split(train, groups=groups))
    return np.sort(train[fit_pos]), np.sort(train[validation_pos])


def load_dataset(name: str) -> DatasetBundle:
    path = DATASETS[name]
    rows = json.loads((path / "incidents.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    archive = np.load(path / "features.npz")
    features = {key: np.asarray(archive[key]) for key in archive.files}
    labels = features["labels"].astype(int)
    train = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=int)
    outer_validation = np.asarray(
        [i for i, row in enumerate(rows) if row["split"] == "validation"], dtype=int
    )
    test = np.asarray([i for i, row in enumerate(rows) if row["split"] == "test"], dtype=int)
    if name == "aiops2022":
        fit, validation = _aiops_inner_split(rows, train)
    else:
        fit, validation = train.copy(), outer_validation.copy()
    split_bin = int(round(float(metadata["before"]) / float(metadata["step"])))
    if split_bin <= 0 or split_bin >= features["metrics"].shape[2]:
        raise ValueError(f"Invalid reference/fault boundary for {name}: {split_bin}")
    return DatasetBundle(
        name=name,
        path=path,
        rows=rows,
        metadata=metadata,
        features=features,
        services=list(metadata["services"]),
        labels=labels,
        train=train,
        validation=validation,
        fit=fit,
        test=test,
        split_bin=split_bin,
    )


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores -= scores.min(axis=1, keepdims=True)
    total = scores.sum(axis=1, keepdims=True)
    uniform = np.full_like(scores, 1.0 / scores.shape[1])
    return np.divide(scores, total, out=uniform, where=total > 1e-15)


def row_unit_interval(scores: np.ndarray) -> np.ndarray:
    """Map each incident row to [0, 1], retaining stable all-equal ties."""

    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64))
    low = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - low
    return np.divide(scores - low, span, out=np.zeros_like(scores), where=span > 1e-15)


def ranks_from_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), axis=1, kind="stable")
    matches = order == np.asarray(labels)[:, None]
    if not matches.any(axis=1).all():
        raise ValueError("Every label must occur exactly once in the candidate universe")
    return matches.argmax(axis=1) + 1


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    ranks = ranks_from_scores(scores, labels)
    return {
        **{f"PR@{k}": float(np.mean(ranks <= k)) for k in (1, 3, 5)},
        **{
            f"MAP@{k}": float(np.mean(np.where(ranks <= k, 1.0 / ranks, 0.0)))
            for k in (3, 5)
        },
    }


def paired_bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    repeats: int = 10_000,
    seed: int = 2027,
) -> dict[str, dict[str, float]]:
    """Incident bootstrap confidence intervals for a single ranking method."""

    ranks = ranks_from_scores(scores, labels)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ranks), size=(repeats, len(ranks)))
    sampled = ranks[draw]
    values: dict[str, np.ndarray] = {}
    for k in (1, 3, 5):
        values[f"PR@{k}"] = np.mean(sampled <= k, axis=1)
    for k in (3, 5):
        values[f"MAP@{k}"] = np.mean(np.where(sampled <= k, 1 / sampled, 0), axis=1)
    return {
        key: {
            "low": float(np.quantile(value, 0.025)),
            "high": float(np.quantile(value, 0.975)),
        }
        for key, value in values.items()
    }


def summarize_repeats(score_matrices: Iterable[np.ndarray], labels: np.ndarray) -> dict[str, Any]:
    matrices = [np.asarray(matrix) for matrix in score_matrices]
    runs = [ranking_metrics(matrix, labels) for matrix in matrices]
    summary: dict[str, Any] = {"runs": runs, "run_count": len(runs)}
    for metric in METRICS:
        values = np.asarray([run[metric] for run in runs], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def test_case_records(bundle: DatasetBundle, scores: np.ndarray) -> list[dict[str, Any]]:
    records = []
    for index, row_scores in zip(bundle.test, scores):
        order = np.argsort(-row_scores, kind="stable")
        rank = int(np.flatnonzero(order == bundle.labels[index])[0] + 1)
        records.append(
            {
                "incident_id": bundle.rows[index]["incident_id"],
                "root_service": bundle.rows[index]["root_service"],
                "fault_type": bundle.rows[index]["fault_type"],
                "rank": rank,
                "ranking": [bundle.services[position] for position in order],
                "scores": {
                    bundle.services[position]: float(row_scores[position]) for position in order
                },
            }
        )
    return records


def dataset_protocol(bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "dataset": bundle.name,
        "path": str(bundle.path.relative_to(ROOT)),
        "counts": {
            "outer_train": int(len(bundle.train)),
            "adapter_fit": int(len(bundle.fit)),
            "adapter_validation": int(len(bundle.validation)),
            "test": int(len(bundle.test)),
        },
        "services": bundle.services,
        "split_bin": bundle.split_bin,
        "fit_incident_ids": [bundle.rows[i]["incident_id"] for i in bundle.fit],
        "validation_incident_ids": [
            bundle.rows[i]["incident_id"] for i in bundle.validation
        ],
        "test_incident_ids": [bundle.rows[i]["incident_id"] for i in bundle.test],
        "features_sha256": sha256(bundle.path / "features.npz"),
        "incidents_sha256": sha256(bundle.path / "incidents.json"),
        "metadata_sha256": sha256(bundle.path / "metadata.json"),
        "tie_break": "stable ascending metadata service order",
    }


def modality_views(values: np.ndarray, split_bin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pre, post, and channel-wise post-vs-pre descriptor tensors.

    Descriptor order per channel: post mean, post absolute mean, post absolute
    maximum, post standard deviation, mean shift, and active-bin fraction.
    """

    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    pre, post = values[:, :, :split_bin], values[:, :, split_bin:]
    pre_mean = pre.mean(axis=2)
    post_mean = post.mean(axis=2)
    descriptor = np.stack(
        [
            post_mean,
            np.abs(post).mean(axis=2),
            np.abs(post).max(axis=2),
            post.std(axis=2),
            post_mean - pre_mean,
            (np.abs(post) > 2.0).mean(axis=2),
        ],
        axis=-1,
    )
    descriptor = descriptor.reshape(*descriptor.shape[:2], -1)
    return pre, post, descriptor


def anomaly_score(values: np.ndarray, split_bin: int) -> np.ndarray:
    """Robust per-incident/per-service anomaly magnitude without labels."""

    pre, post, _ = modality_views(values, split_bin)
    pre_level = np.median(pre, axis=2)
    level = np.abs(post - pre_level[:, :, None, :])
    top = np.quantile(level, 0.90, axis=2)
    persistence = np.mean(level > 2.0, axis=2)
    channel = top + persistence
    return np.nan_to_num(np.mean(channel, axis=-1))


def robust_deviation(values: np.ndarray, split_bin: int) -> np.ndarray:
    """Absolute post-boundary deviation in robust within-incident scale units."""

    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    pre, post = values[:, :, :split_bin], values[:, :, split_bin:]
    center = np.median(pre, axis=2, keepdims=True)
    mad = np.median(np.abs(pre - center), axis=2, keepdims=True)
    scale = np.maximum(1.4826 * mad, 0.25)
    return np.abs(post - center) / scale


def fallback_scores(bundle: DatasetBundle) -> np.ndarray:
    """Frozen label-independent ranking used after adapter failures.

    Each available modality contributes equally after per-incident min-max
    scaling.  The all-zero case stays tied and therefore follows metadata order.
    """

    available = []
    for modality in ("metrics", "logs", "traces"):
        if modality in bundle.features and bundle.features[modality].shape[-1] > 0:
            available.append(row_unit_interval(anomaly_score(bundle.features[modality], bundle.split_bin)))
    if not available:
        return np.zeros((len(bundle.rows), bundle.service_count), dtype=np.float64)
    return np.mean(available, axis=0)


def complete_or_fallback(scores: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace only invalid incident rows and return their Boolean mask."""

    scores = np.asarray(scores, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    if scores.shape != fallback.shape:
        raise ValueError(f"score/fallback shape mismatch: {scores.shape} != {fallback.shape}")
    invalid = (~np.isfinite(scores).all(axis=1)) | (np.max(scores, axis=1) < np.min(scores, axis=1))
    output = scores.copy()
    output[invalid] = fallback[invalid]
    return output, invalid


def adjacency_for_incident(bundle: DatasetBundle, index: int) -> np.ndarray:
    adjacency = np.eye(bundle.service_count, dtype=np.float64)
    position = {service: i for i, service in enumerate(bundle.services)}
    for edge in bundle.rows[index].get("calls", []):
        if edge["source"] not in position or edge["target"] not in position:
            continue
        source, target = position[edge["source"]], position[edge["target"]]
        adjacency[source, target] += np.log1p(max(float(edge.get("count", 1)), 0.0))
    return adjacency


def training_adjacency(bundle: DatasetBundle) -> np.ndarray:
    adjacency = np.eye(bundle.service_count, dtype=np.float64)
    for index in bundle.fit:
        adjacency += adjacency_for_incident(bundle, int(index)) - np.eye(bundle.service_count)
    return adjacency


def random_walk(
    adjacency: np.ndarray,
    personalization: np.ndarray,
    damping: float = 0.85,
    reverse: bool = False,
    iterations: int = 100,
) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float64)
    if reverse:
        matrix = matrix.T
    matrix = np.maximum(matrix, 0)
    row_sum = matrix.sum(axis=1, keepdims=True)
    transition = np.divide(
        matrix, row_sum, out=np.full_like(matrix, 1.0 / len(matrix)), where=row_sum > 0
    )
    seed = np.maximum(np.asarray(personalization, dtype=np.float64), 0)
    if seed.sum() <= 1e-15:
        seed = np.ones_like(seed)
    seed /= seed.sum()
    state = seed.copy()
    for _ in range(iterations):
        update = damping * transition.T.dot(state) + (1 - damping) * seed
        if np.max(np.abs(update - state)) < 1e-12:
            state = update
            break
        state = update
    return state / max(state.sum(), 1e-15)


def choose_configuration(
    candidates: Iterable[tuple[dict[str, Any], np.ndarray]],
    bundle: DatasetBundle,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """Select by validation PR@1, MAP@5, PR@3, then lexical config JSON."""

    records = []
    best_key: tuple[Any, ...] | None = None
    best_config: dict[str, Any] | None = None
    best_scores: np.ndarray | None = None
    for config, scores in candidates:
        metric = ranking_metrics(scores[bundle.validation], bundle.labels[bundle.validation])
        record = {"configuration": config, "validation": metric}
        records.append(record)
        key = (
            metric["PR@1"],
            metric["MAP@5"],
            metric["PR@3"],
            json.dumps(config, sort_keys=True),
        )
        if best_key is None or key > best_key:
            best_key, best_config, best_scores = key, config, scores
    if best_config is None or best_scores is None:
        raise ValueError("No candidate configuration")
    return best_config, best_scores, records


def save_method_run(
    bundle: DatasetBundle,
    method: str,
    scores: np.ndarray,
    provenance: dict[str, Any],
    selection: list[dict[str, Any]] | None = None,
    run_name: str = "run-0",
) -> dict[str, Any]:
    scores = normalize_rows(scores)
    directory = OUTPUT / bundle.name / method / run_name
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "scores.npz",
        all_scores=scores,
        test_indices=bundle.test,
        test_scores=scores[bundle.test],
        labels=bundle.labels,
    )
    result = {
        "method": method,
        "run_name": run_name,
        "dataset": bundle.name,
        "metrics": ranking_metrics(scores[bundle.test], bundle.labels[bundle.test]),
        "bootstrap_95": paired_bootstrap(scores[bundle.test], bundle.labels[bundle.test]),
        "provenance": provenance,
        "dataset_protocol": dataset_protocol(bundle),
        "selection": selection or [],
        "scores_sha256": sha256(directory / "scores.npz"),
    }
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "test_cases.json", test_case_records(bundle, scores[bundle.test]))
    return result
