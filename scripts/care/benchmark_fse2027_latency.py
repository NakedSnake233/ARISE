"""Loaded-model, per-incident RCA latency on the frozen FSE cohorts.

This benchmark replays saved checkpoints and frozen adapter configurations. It
never fits or selects a model on the test set. Raw telemetry ingestion, disk I/O,
checkpoint loading, and training are outside the timed region.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fse2027_baseline_common import (
    OUTPUT, ROOT, DatasetBundle, load_dataset, modality_views, normalize_rows, row_unit_interval,
    sha256,
)
from fse2027_classical_baselines import (
    METHODS, _log_examples, _nezha_event_weight, sbld_scores, spectrum_formula,
)
from fse2027_neural_baselines import (
    EadroAdapter, MulanAdapter, TraceVAE, adjacency_tensor,
    standardized_descriptors,
)
from run_review_experiments import BasePredictor
from robust_evidence import RobustRanker, all_available, fuse_all


DEST = ROOT / "artifacts/care/fse2027-latency-20260927"
CARE_DIR = ROOT / "artifacts/care/review-response-v1"
METHODS_ORDER = [
    "MicroScope", "MicroRCA", "SBLD", "LogFaultFlagger", "TraceAnomaly",
    "MicroRank", "Nezha", "Eadro", "MULAN", "CARE",
]
SEEDS = (42, 43, 44, 45, 46)


def one_row(bundle: DatasetBundle, index: int) -> DatasetBundle:
    return dataclasses.replace(
        bundle,
        rows=[bundle.rows[index]],
        features={k: v[index:index + 1] for k, v in bundle.features.items()},
        labels=bundle.labels[index:index + 1],
        train=np.array([], dtype=int), validation=np.array([], dtype=int),
        fit=np.array([], dtype=int), test=np.array([0], dtype=int),
    )


def _sbld_weights(bundle: DatasetBundle, config: dict) -> np.ndarray:
    logs = np.nan_to_num(bundle.features["logs"], nan=0.0)
    active = np.any(np.abs(logs[:, :, bundle.split_bin:]) > float(config["active_threshold"]), axis=2)
    weights = np.zeros((bundle.service_count, logs.shape[-1]), dtype=np.float64)
    for service in range(bundle.service_count):
        fail_mask = bundle.labels[bundle.fit] == service
        pass_mask = ~fail_mask
        fail_active = active[bundle.fit[fail_mask], service]
        pass_active = active[bundle.fit[pass_mask], service]
        ef = fail_active.sum(axis=0) if len(fail_active) else np.zeros(logs.shape[-1])
        ep = pass_active.sum(axis=0) if len(pass_active) else np.zeros(logs.shape[-1])
        nf = fail_mask.sum() - ef
        n_pass = pass_mask.sum() - ep
        weights[service] = spectrum_formula(ef, ep, nf, n_pass, str(config["formula"]))
    return weights


def classical_callable(bundle: DatasetBundle, method: str, config: dict, index: int, cache: dict):
    incident = one_row(bundle, index)
    if method in ("MicroScope", "MicroRCA", "MicroRank"):
        return lambda: METHODS[method](incident, config)[0]
    if method == "SBLD":
        if "weights" not in cache:
            cache["weights"] = _sbld_weights(bundle, config)
        weights = cache["weights"]

        def score():
            logs = np.nan_to_num(incident.features["logs"], nan=0.0)
            active = np.any(np.abs(logs[:, :, incident.split_bin:]) > float(config["active_threshold"]), axis=2)
            contribution = active * weights[None]
            if config["service_aggregation"] == "max":
                return contribution.max(axis=-1)[0]
            return contribution.sum(axis=-1)[0]

        return score
    if method == "LogFaultFlagger":
        if "estimator" not in cache:
            x, y, _ = _log_examples(bundle)
            positions = np.concatenate([
                np.arange(i * bundle.service_count, (i + 1) * bundle.service_count)
                for i in bundle.fit
            ])
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(config["C"]), class_weight="balanced",
                    max_iter=int(config["max_iter"]), solver="liblinear", random_state=2027,
                ),
            )
            estimator.fit(x[positions], y[positions])
            cache["estimator"] = estimator
        estimator = cache["estimator"]

        def score():
            _, _, descriptor = modality_views(incident.features["logs"], incident.split_bin)
            return estimator.predict_proba(descriptor.reshape(incident.service_count, -1))[:, 1]

        return score
    if method == "Nezha":
        if "weights" not in cache:
            cache["weights"] = {
                name: _nezha_event_weight(bundle, name, float(config["event_threshold"]))[1]
                for name in ("metrics", "logs", "traces")
            }
        weights = cache["weights"]

        def score():
            from fse2027_baseline_common import robust_deviation, random_walk, adjacency_for_incident

            contributions = []
            for name in ("metrics", "logs", "traces"):
                active = (robust_deviation(incident.features[name], incident.split_bin)
                          > float(config["event_threshold"])).mean(axis=2)
                weighted = active * weights[name][None]
                value = weighted.max(axis=-1) if config["service_aggregation"] == "max" else weighted.sum(axis=-1)
                contributions.append(row_unit_interval(value)[0])
            direct = np.mean(contributions, axis=0)
            propagated = random_walk(adjacency_for_incident(incident, 0), direct, damping=0.85, reverse=True)
            mix = float(config["graph_mix"])
            return (1 - mix) * direct + mix * row_unit_interval(propagated[None])[0]

        return score
    raise ValueError(method)


def neural_callables(bundle: DatasetBundle, method: str, config: dict, device: torch.device):
    entries = {}
    if method == "TraceAnomaly":
        trace = np.nan_to_num(bundle.features["traces"].astype(np.float32))
        fit = trace[bundle.fit, :, :bundle.split_bin].reshape(-1, trace.shape[-1])
        center, scale = fit.mean(axis=0), fit.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        post = (trace[:, :, bundle.split_bin:] - center) / scale
    else:
        descriptors = standardized_descriptors(bundle)
        widths = {name: array.shape[-1] for name, array in descriptors.items()}
        adjacency = adjacency_tensor(bundle)
    for seed in SEEDS:
        checkpoint = OUTPUT / bundle.name / method / f"seed-{seed}" / "model.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if method == "TraceAnomaly":
            model = TraceVAE(trace.shape[-1], int(config["hidden_dim"]), int(config["latent_dim"]))
        elif method == "Eadro":
            model = EadroAdapter(widths, int(config["hidden_dim"]), int(config["attention_heads"]), float(config["dropout"]))
        else:
            model = MulanAdapter(widths, int(config["hidden_dim"]))
        model.load_state_dict(state)
        model = model.to(device).eval()
        per_incident = []
        for index in bundle.test:
            if method == "TraceAnomaly":
                inp = torch.as_tensor(post[index].reshape(-1, trace.shape[-1]), dtype=torch.float32, device=device)

                def score(model=model, inp=inp, shape=post[index].shape):
                    with torch.no_grad():
                        reconstruction, _, _ = model(inp)
                        error = (reconstruction - inp).square().mean(dim=-1).cpu().numpy().reshape(shape[:2])
                    aggregation = config["aggregation"]
                    if aggregation == "mean":
                        return error.mean(axis=-1)
                    if aggregation == "max":
                        return error.max(axis=-1)
                    return np.quantile(error, 0.90, axis=-1)
            else:
                inputs = {
                    name: torch.as_tensor(value[index:index + 1], dtype=torch.float32, device=device)
                    for name, value in descriptors.items()
                }
                graph = torch.as_tensor(adjacency[index:index + 1], dtype=torch.float32, device=device)

                if method == "Eadro":
                    def score(model=model, inputs=inputs, graph=graph):
                        with torch.no_grad():
                            logits = model(inputs, graph)
                            return torch.softmax(logits, dim=-1).cpu().numpy()[0]
                else:
                    def score(model=model, inputs=inputs, graph=graph, config=config):
                        from fse2027_baseline_common import random_walk

                        with torch.no_grad():
                            logits, _ = model(inputs)
                            direct = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                        propagated = random_walk(graph[0].cpu().numpy(), direct, damping=0.85, reverse=True)
                        mix = float(config["graph_mix"])
                        return (1 - mix) * row_unit_interval(direct[None])[0] + mix * row_unit_interval(propagated[None])[0]
            per_incident.append(score)
        entries[seed] = per_incident
    return entries


def care_callables(bundle: DatasetBundle, device: torch.device):
    evidence_dir = CARE_DIR / bundle.name
    protocol = json.loads((CARE_DIR / "protocol.json").read_text())
    dataset_config = protocol["configuration"]["datasets"][bundle.name]
    x = np.load(ROOT / dataset_config["evidence"] / "full.npz")["x"]
    names = json.loads((ROOT / dataset_config["evidence"] / "full-names.json").read_text())
    base_weight = float(protocol["configuration"]["evidence"]["base_weight"])
    evidence_weight = float(protocol["configuration"]["evidence"]["evidence_weight"])
    availability = all_available(x.shape)
    features = {name: bundle.features[name] for name in ("metrics", "logs", "traces")}
    entries = {}
    for seed in SEEDS:
        model_dir = CARE_DIR / "base" / bundle.name / f"seed-{seed}"
        predictor = BasePredictor(model_dir, device.type, batch_size=1)
        run_dir = evidence_dir / f"seed-{seed}"
        with (run_dir / "robust_evidence_model.pkl").open("rb") as handle:
            robust_model = pickle.load(handle)
        with (run_dir / "generic_evidence_model.pkl").open("rb") as handle:
            generic_model = pickle.load(handle)
        robust = RobustRanker(robust_model, names, True, False, [], 0)
        generic = RobustRanker(generic_model, names, True, True, [], 0)
        result = json.loads((run_dir / "result.json").read_text())
        weight = float(result["main"]["selected_generic_weight"])
        per_incident = []
        for index in bundle.test:
            selected = np.asarray([index], dtype=int)

            def score(predictor=predictor, selected=selected, robust=robust, generic=generic, weight=weight):
                base = predictor.predict(features, availability, selected)[0]
                raw = robust.predict(x[selected], availability[selected])
                generic_scores = generic.predict(x[selected], availability[selected])
                correction = fuse_all((1 - weight, raw), (weight, generic_scores))[0]
                return fuse_all((base_weight, base[None]), (evidence_weight, correction[None]))[0]

            per_incident.append(score)
        entries[seed] = per_incident
    return entries


def validate_scores(bundle: DatasetBundle, method: str, seed: int | None, values: np.ndarray):
    run = f"seed-{seed}" if seed is not None else "run-0"
    if method == "CARE":
        archive = CARE_DIR / bundle.name / run / "predictions.npz"
        reference = np.load(archive)["final"]
    else:
        archive = OUTPUT / bundle.name / method / run / "scores.npz"
        reference = np.load(archive)["test_scores"]
    current = normalize_rows(values.copy())
    expected = reference.copy() if method != "CARE" else normalize_rows(reference.copy())
    if current.shape != expected.shape:
        raise AssertionError((method, bundle.name, current.shape, expected.shape))
    max_error = float(np.max(np.abs(current - expected)))
    rank_agreement = float(np.mean(np.argmax(current, axis=1) == np.argmax(expected, axis=1)))
    if max_error > 1e-4 or rank_agreement != 1.0:
        raise AssertionError(f"{bundle.name}/{method}/{run}: score mismatch {max_error}, top1 agreement {rank_agreement}")
    return {"max_normalized_score_error": max_error, "top1_agreement": rank_agreement, "reference_sha256": sha256(archive)}


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_callables(callables, device: torch.device, repeats: int):
    observations = []
    outputs = []
    for fn in callables:
        for _ in range(2):
            fn()
        outputs.append(np.asarray(fn(), dtype=np.float64))
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter_ns()
            fn()
            synchronize(device)
            observations.append((time.perf_counter_ns() - start) / 1e6)
    return np.stack(outputs), observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--datasets", nargs="+", default=["trainticket", "aiops2022"])
    parser.add_argument("--methods", nargs="+", default=METHODS_ORDER)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    DEST.mkdir(parents=True, exist_ok=True)
    raw = []
    for dataset in args.datasets:
        bundle = load_dataset(dataset)
        for method in args.methods:
            if method == "CARE":
                entries = care_callables(bundle, device)
            elif method in ("TraceAnomaly", "Eadro", "MULAN"):
                result = json.loads((OUTPUT / dataset / method / "seed-42" / "result.json").read_text())
                entries = neural_callables(bundle, method, result["provenance"]["selected_configuration"], device)
            else:
                result = json.loads((OUTPUT / dataset / method / "run-0" / "result.json").read_text())
                config = result["provenance"]["selected_configuration"]
                cache = {}
                entries = {None: [classical_callable(bundle, method, config, int(i), cache) for i in bundle.test]}
            for seed, per_incident in entries.items():
                values, times = time_callables(per_incident, device, args.repeats)
                validation = validate_scores(bundle, method, seed, values)
                record = {
                    "dataset": dataset, "method": method, "seed": seed,
                    "incident_count": len(bundle.test), "repeats": args.repeats,
                    "latency_ms": times, "mean_ms": float(np.mean(times)),
                    "median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95)),
                    "validation": validation,
                }
                raw.append(record)
                print(f"{dataset:12s} {method:16s} {str(seed):>4s} {record['mean_ms']:9.3f} ms; parity {validation['max_normalized_score_error']:.2g}", flush=True)
                (DEST / "results.json").write_text(json.dumps(raw, indent=2) + "\n")
    provenance = {
        "scope": "loaded-model single-incident scoring from prepared numeric tensors; excludes data acquisition, parsing, model loading, training, and storage I/O",
        "repeats_per_incident_per_seed": args.repeats,
        "warmup_calls_per_incident": 2,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cpu": platform.processor(), "platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "numpy": np.__version__, "script_sha256": sha256(Path(__file__)),
        "baseline_protocol_sha256": sha256(OUTPUT / "BASELINE_PROTOCOL_FROZEN.json"),
        "care_protocol_sha256": sha256(CARE_DIR / "protocol.json"),
    }
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
