"""Frozen deterministic baseline adapters for the FSE 2027 matched study."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # direct script execution
    from fse2027_baseline_common import (
        OUTPUT, ROOT, DatasetBundle, adjacency_for_incident, anomaly_score, atomic_json,
        choose_configuration, complete_or_fallback, fallback_scores, load_dataset,
        modality_views, normalize_rows, random_walk, robust_deviation, row_unit_interval,
        save_method_run, sha256,
    )
except ImportError:  # package import in tests
    from .fse2027_baseline_common import (
        OUTPUT, ROOT, DatasetBundle, adjacency_for_incident, anomaly_score, atomic_json,
        choose_configuration, complete_or_fallback, fallback_scores, load_dataset,
        modality_views, normalize_rows, random_walk, robust_deviation, row_unit_interval,
        save_method_run, sha256,
    )


PROTOCOL = OUTPUT / "BASELINE_PROTOCOL_FROZEN.json"
PROTOCOL_HASH = OUTPUT / "BASELINE_PROTOCOL_FROZEN.sha256"
EXPECTED_PROTOCOL_SHA256 = "81d0a737e261cddfda665d4a59fb81d567e371e846893dee30f58631e3d676da"


def verify_protocol() -> dict[str, Any]:
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen baseline protocol hash mismatch")
    digest_line = PROTOCOL_HASH.read_text().split()[0]
    if digest_line != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen baseline sidecar mismatch")
    return json.loads(PROTOCOL.read_text())


def configurations(grid: dict[str, list[Any]]):
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def _graph_scores(
    bundle: DatasetBundle,
    direct: np.ndarray,
    *,
    damping: float,
    reverse: bool,
) -> np.ndarray:
    result = np.zeros_like(direct, dtype=np.float64)
    for index in range(len(bundle.rows)):
        result[index] = random_walk(
            adjacency_for_incident(bundle, index), direct[index], damping=damping, reverse=reverse
        )
    return result


def microscope_scores(bundle: DatasetBundle, config: dict[str, Any]) -> np.ndarray:
    deviation = robust_deviation(bundle.features["metrics"], bundle.split_bin)
    threshold = float(config["anomaly_threshold"])
    exceedance = np.maximum(deviation - threshold, 0.0)
    direct = exceedance.mean(axis=(2, 3)) + (deviation > threshold).mean(axis=(2, 3))
    direct = row_unit_interval(direct)
    propagated = _graph_scores(
        bundle, direct, damping=float(config["damping"]), reverse=bool(config["reverse"])
    )
    mix = float(config["direct_mix"])
    return mix * direct + (1.0 - mix) * row_unit_interval(propagated)


def microrca_scores(bundle: DatasetBundle, config: dict[str, Any]) -> np.ndarray:
    metric = row_unit_interval(anomaly_score(bundle.features["metrics"], bundle.split_bin))
    trace = row_unit_interval(anomaly_score(bundle.features["traces"], bundle.split_bin))
    direct = 0.7 * metric + 0.3 * trace
    graph_scores = np.zeros_like(direct)
    for index in range(len(bundle.rows)):
        adjacency = adjacency_for_incident(bundle, index).copy()
        # Attribute observed call edges by endpoint trace symptoms, matching the
        # source algorithm's use of anomalous response-time edges.
        attribute = 1.0 + 0.5 * (trace[index, :, None] + trace[index, None, :])
        adjacency *= attribute
        graph_scores[index] = random_walk(
            adjacency,
            direct[index],
            damping=float(config["damping"]),
            reverse=bool(config["reverse"]),
        )
    mix = float(config["direct_mix"])
    return mix * direct + (1.0 - mix) * row_unit_interval(graph_scores)


def spectrum_formula(
    ef: np.ndarray, ep: np.ndarray, nf: np.ndarray, npass: np.ndarray, name: str
) -> np.ndarray:
    ef, ep, nf, npass = [np.asarray(value, dtype=np.float64) for value in (ef, ep, nf, npass)]
    if name == "ochiai":
        denominator = np.sqrt((ef + nf) * (ef + ep))
        return np.divide(ef, denominator, out=np.zeros_like(ef), where=denominator > 0)
    if name == "tarantula":
        fail_rate = np.divide(ef, ef + nf, out=np.zeros_like(ef), where=(ef + nf) > 0)
        pass_rate = np.divide(ep, ep + npass, out=np.zeros_like(ep), where=(ep + npass) > 0)
        return np.divide(fail_rate, fail_rate + pass_rate, out=np.zeros_like(ef), where=(fail_rate + pass_rate) > 0)
    if name == "jaccard":
        denominator = ef + ep + nf
        return np.divide(ef, denominator, out=np.zeros_like(ef), where=denominator > 0)
    if name == "dstar2":
        denominator = ep + nf
        return np.divide(ef**2, denominator, out=np.zeros_like(ef), where=denominator > 0)
    raise ValueError(name)


def sbld_scores(bundle: DatasetBundle, config: dict[str, Any]) -> np.ndarray:
    logs = np.nan_to_num(bundle.features["logs"], nan=0.0)
    post = np.abs(logs[:, :, bundle.split_bin :])
    active = np.any(post > float(config["active_threshold"]), axis=2)
    fit = bundle.fit
    channel_weights = np.zeros((bundle.service_count, logs.shape[-1]), dtype=np.float64)
    for service in range(bundle.service_count):
        fail_mask = bundle.labels[fit] == service
        pass_mask = ~fail_mask
        fail_active = active[fit[fail_mask], service]
        pass_active = active[fit[pass_mask], service]
        ef = fail_active.sum(axis=0) if len(fail_active) else np.zeros(logs.shape[-1])
        ep = pass_active.sum(axis=0) if len(pass_active) else np.zeros(logs.shape[-1])
        nf = fail_mask.sum() - ef
        npass = pass_mask.sum() - ep
        channel_weights[service] = spectrum_formula(ef, ep, nf, npass, str(config["formula"]))
    contributions = active * channel_weights[None]
    if config["service_aggregation"] == "max":
        return contributions.max(axis=-1)
    return contributions.sum(axis=-1)


def _log_examples(bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    _, _, descriptor = modality_views(bundle.features["logs"], bundle.split_bin)
    rows, services, width = descriptor.shape
    x = descriptor.reshape(rows * services, width)
    y = (np.arange(services)[None, :] == bundle.labels[:, None]).astype(int).reshape(-1)
    return x, y, (rows, services)


def logfaultflagger_scores(bundle: DatasetBundle, config: dict[str, Any]) -> np.ndarray:
    x, y, shape = _log_examples(bundle)
    train_positions = np.concatenate(
        [np.arange(index * bundle.service_count, (index + 1) * bundle.service_count) for index in bundle.fit]
    )
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(config["C"]),
            class_weight="balanced",
            max_iter=int(config["max_iter"]),
            solver="liblinear",
            random_state=2027,
        ),
    )
    estimator.fit(x[train_positions], y[train_positions])
    return estimator.predict_proba(x)[:, 1].reshape(shape)


def microrank_scores(bundle: DatasetBundle, config: dict[str, Any]) -> np.ndarray:
    deviation = robust_deviation(bundle.features["traces"], bundle.split_bin)
    # [incident, service, post time]
    activity = deviation.mean(axis=-1)
    scores = np.zeros((len(bundle.rows), bundle.service_count), dtype=np.float64)
    for index in range(len(bundle.rows)):
        matrix = activity[index]
        request_anomaly = matrix.max(axis=0)
        threshold = np.quantile(request_anomaly, float(config["anomaly_quantile"]))
        abnormal = request_anomaly >= threshold
        normal = ~abnormal
        ef = matrix[:, abnormal].sum(axis=1) if abnormal.any() else np.zeros(bundle.service_count)
        ep = matrix[:, normal].sum(axis=1) if normal.any() else np.zeros(bundle.service_count)
        fail_total = np.full(bundle.service_count, max(ef.sum(), 1e-15))
        pass_total = np.full(bundle.service_count, max(ep.sum(), 1e-15))
        nf = np.maximum(fail_total - ef, 0)
        npass = np.maximum(pass_total - ep, 0)
        spectrum = spectrum_formula(ef, ep, nf, npass, str(config["spectrum"]))
        page = random_walk(
            adjacency_for_incident(bundle, index),
            ef,
            damping=float(config["damping"]),
            reverse=True,
        )
        scores[index] = row_unit_interval(spectrum[None])[0] * (0.5 + 0.5 * row_unit_interval(page[None])[0])
    return scores


def _nezha_event_weight(
    bundle: DatasetBundle, modality: str, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    deviation = robust_deviation(bundle.features[modality], bundle.split_bin)
    active = deviation > threshold
    weights = np.zeros((bundle.service_count, deviation.shape[-1]), dtype=np.float64)
    for service in range(bundle.service_count):
        root = bundle.fit[bundle.labels[bundle.fit] == service]
        other = bundle.fit[bundle.labels[bundle.fit] != service]
        root_rate = active[root, service].mean(axis=(0, 1)) if len(root) else np.zeros(deviation.shape[-1])
        other_rate = active[other, service].mean(axis=(0, 1)) if len(other) else np.zeros(deviation.shape[-1])
        weights[service] = np.maximum(np.log((root_rate + 0.05) / (other_rate + 0.05)), 0.0)
    post_rate = active.mean(axis=2)
    return post_rate, weights


def nezha_scores(bundle: DatasetBundle, config: dict[str, Any]) -> np.ndarray:
    threshold = float(config["event_threshold"])
    modality_scores = []
    for modality in ("metrics", "logs", "traces"):
        active, weight = _nezha_event_weight(bundle, modality, threshold)
        contribution = active * weight[None]
        if config["service_aggregation"] == "max":
            score = contribution.max(axis=-1)
        else:
            score = contribution.sum(axis=-1)
        modality_scores.append(row_unit_interval(score))
    direct = np.mean(modality_scores, axis=0)
    propagated = _graph_scores(bundle, direct, damping=0.85, reverse=True)
    graph_mix = float(config["graph_mix"])
    return (1.0 - graph_mix) * direct + graph_mix * row_unit_interval(propagated)


METHODS: dict[str, Callable[[DatasetBundle, dict[str, Any]], np.ndarray]] = {
    "MicroScope": microscope_scores,
    "MicroRCA": microrca_scores,
    "SBLD": sbld_scores,
    "LogFaultFlagger": logfaultflagger_scores,
    "MicroRank": microrank_scores,
    "Nezha": nezha_scores,
}


def run_method(bundle: DatasetBundle, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    scorer = METHODS[name]
    grid = entry["tuning"]
    candidates = []
    invalid_counts = []
    fallback = fallback_scores(bundle)
    for config in configurations(grid):
        raw = scorer(bundle, config)
        completed, invalid = complete_or_fallback(raw, fallback)
        candidates.append((config, completed))
        invalid_counts.append({"configuration": config, "invalid_incidents": int(invalid.sum())})
    selected, scores, selection = choose_configuration(candidates, bundle)
    provenance = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "implementation_class": entry["implementation_class"],
        "material_adapter": entry["material_adapter"],
        "adaptation": entry["adaptation"],
        "modalities": entry["modalities"],
        "selected_configuration": selected,
        "invalid_counts_by_configuration": invalid_counts,
        "code": str(Path(__file__).resolve().relative_to(ROOT)),
        "code_sha256": sha256(Path(__file__)),
    }
    return save_method_run(bundle, name, scores, provenance, selection=selection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["trainticket", "aiops2022"], default=["trainticket", "aiops2022"])
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    args = parser.parse_args()
    protocol = verify_protocol()
    results = []
    for dataset_name in args.datasets:
        bundle = load_dataset(dataset_name)
        for method_name in args.methods:
            result = run_method(bundle, method_name, protocol["methods"][method_name])
            results.append({"dataset": dataset_name, "method": method_name, "metrics": result["metrics"]})
    atomic_json(OUTPUT / "classical_run_summary.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
