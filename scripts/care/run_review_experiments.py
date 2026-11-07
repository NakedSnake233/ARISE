"""Run the reviewer-requested CARE robustness and cost experiment suite.

The suite is resumable by phase and writes only under its configured output
directory.  Main and history-type-holdout base models are trained by the isolated
``train_robust_base.py`` entry point; all evidence variants are fitted from
frozen train rows and evaluated against dataset ground truth.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from evidence_reranker import fuse, paired_statistics
from build_sft import build_fpg, evidence as build_observations, retrieve
from correct import correct
from robust_evidence import (
    MODALITIES,
    all_available,
    complete_modality_availability,
    feature_groups,
    fit_fixed_availability_ranker,
    fit_robust_ranker,
    fault_type_holdout,
    fuse_all,
    local_missing_availability,
    mask_feature_group,
)
from train_base import CAREBase, ranking_metrics


ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    "robust_evidence.py",
    "train_robust_base.py",
    "run_review_experiments.py",
    "evidence_reranker.py",
    "full_evidence.py",
    "train_base.py",
    "build_sft.py",
    "correct.py",
    "profile_llm_costs.py",
    "contrast_prompt.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_digest(path: Path, pattern: str = "*") -> str:
    hasher = hashlib.sha256()
    for item in sorted(p for p in path.glob(pattern) if p.is_file()):
        hasher.update(item.name.encode())
        hasher.update(bytes.fromhex(digest(item)))
    return hasher.hexdigest()


def files_digest(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths, key=str):
        hasher.update(str(path).encode())
        hasher.update(bytes.fromhex(digest(path)))
    return hasher.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def slug(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def mean_std(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "std": float(np.std([record[key] for record in records], ddof=1)) if len(records) > 1 else 0.0,
        }
        for key in records[0]
    }


def measurement(scores: np.ndarray, labels: np.ndarray, rows: list[dict]) -> dict:
    by_fault = {}
    for fault_type in sorted({row["fault_type"] for row in rows}):
        indices = np.asarray([i for i, row in enumerate(rows) if row["fault_type"] == fault_type])
        by_fault[fault_type] = {
            "count": len(indices),
            "metrics": ranking_metrics(scores[indices], labels[indices]),
        }
    return {"metrics": ranking_metrics(scores, labels), "by_fault_type": by_fault}


def base_directory(output: Path, dataset: str, seed: int, strict: str | None = None) -> Path:
    if strict is None:
        return output / "base" / dataset / f"seed-{seed}"
    return output / "strict-base" / dataset / slug(strict) / f"seed-{seed}"


def ablation_base_directory(output: Path, dataset: str, seed: int, ablation: str) -> Path:
    return output / "ablation-base" / dataset / ablation / f"seed-{seed}"


def train_base_command(
    cfg: dict,
    dataset: str,
    seed: int,
    output: Path,
    fault_type: str | None = None,
    ablation: str = "full",
    data_path: Path | None = None,
) -> list[str]:
    dataset_cfg = cfg["datasets"][dataset]
    base_cfg = cfg["base"]
    command = [
        sys.executable,
        str(ROOT / "scripts/care/train_robust_base.py"),
        "--data",
        str(data_path) if data_path is not None else dataset_cfg["data"],
        "--output",
        str(output),
        "--epochs",
        str(base_cfg["epochs"]),
        "--seed",
        str(seed),
        "--batch-size",
        str(base_cfg["batch_size"]),
        "--lr",
        str(base_cfg["lr"]),
        "--contrast-weight",
        str(base_cfg["contrast_weight"]),
        "--threads",
        str(base_cfg["threads"]),
        "--modality-dropout",
        str(0.0 if ablation == "no_modality_dropout" else base_cfg["modality_dropout"]),
        "--dropout-loss-weight",
        str(0.0 if ablation == "no_modality_dropout" else base_cfg["dropout_loss_weight"]),
        "--ablation",
        "full" if ablation == "no_modality_dropout" else ablation,
        "--device",
        dataset_cfg["device"],
        "--metric-patch-size",
        str(dataset_cfg["metric_patch_size"]),
        "--checkpoint-selection",
        dataset_cfg["checkpoint_selection"],
    ]
    if fault_type is not None:
        command.extend(["--exclude-fault-type", fault_type])
    return command


def _incident_baseline_standardize(values: np.ndarray, before: int) -> np.ndarray:
    """Standardize each service from its own pre-incident window only."""
    values = np.asarray(values, dtype=np.float32)
    baseline = values[:, :, :before]
    center = np.median(baseline, axis=2, keepdims=True)
    mad = np.median(np.abs(baseline - center), axis=2, keepdims=True) * 1.4826
    scale = np.maximum(mad, 0.5)
    return np.clip((values - center) / scale, -20, 20).astype(np.float32)


def prepare_history_coldstart_data(
    cfg: dict, dataset: str, output: Path
) -> tuple[Path, Path, Path]:
    """Create features for a fault-signature/FPG history-exclusion test.

    This function re-centers each incident from its own pre-window.  The source
    encoder's shared calibration is retained and is not reconstructed per
    fold; a pre-window can overlap a neighboring incident's fault interval.
    Logs are conservatively unavailable because their Drain vocabulary saw
    full training windows.  The resulting claim is supervised/history-type
    generalization under shared calibration, not end-to-end row exclusion.
    """
    dataset_cfg = cfg["datasets"][dataset]
    source = ROOT / dataset_cfg["data"]
    destination = output / "strict-data" / dataset
    evidence_file = destination / "evidence.npz"
    names_file = destination / "evidence-names.json"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["source_features_sha256"] != digest(source / "features.npz")
            or manifest["source_incidents_sha256"] != digest(source / "incidents.json")
            or manifest["features_sha256"] != digest(destination / "features.npz")
            or manifest["evidence_sha256"] != digest(evidence_file)
        ):
            raise ValueError(f"Stale or changed strict preprocessing cache: {destination}")
        return destination, evidence_file, names_file
    destination.mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / "incidents.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    loaded = np.load(source / "features.npz")
    before = int(metadata["before"] // metadata["step"])
    features = {
        "metrics": _incident_baseline_standardize(loaded["metrics"], before),
        "traces": _incident_baseline_standardize(loaded["traces"], before),
        # A fold-specific Drain reparse would change dimensionality by held
        # type.  Treating logs as unavailable is conservative and leakage-free.
        "logs": np.zeros_like(loaded["logs"], dtype=np.float32),
    }
    labels = loaded["labels"]
    np.savez_compressed(destination / "features.npz", **features, labels=labels)
    (destination / "incidents.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    metadata = {
        **metadata,
        "strict_history_preprocessing": (
            "Shared pre-window numeric source calibration is not refit per "
            "outer fold and may include telemetry overlapping a neighboring "
            "incident; it is followed by per-incident pre-window median/MAD. "
            "Logs are unavailable."
        ),
        "limitations": [
            *metadata.get("limitations", []),
            "Strict history-exclusion uses conservative no-log inputs because fold-specific raw Drain reparsing was not retained.",
        ],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    raw = None
    if dataset == "trainticket":
        from evidence_reranker import raw_tt_metrics

        raw = raw_tt_metrics(
            rows,
            metadata["services"],
            metadata["metric_names"],
            metadata["step"],
            features["metrics"].shape[2],
        )
    from evidence_reranker import extract

    evidence, names = extract(features, metadata, rows, raw)
    rpc = np.load(ROOT / dataset_cfg["rpc"] / "features.npz")["x"]
    rpc_names = json.loads(
        (ROOT / dataset_cfg["rpc"] / "feature_names.json").read_text()
    )
    evidence = np.concatenate([evidence, rpc], axis=-1)
    names += rpc_names
    np.savez_compressed(evidence_file, x=evidence)
    names_file.write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n")
    atomic_json(
        manifest_path,
        {
            "dataset": dataset,
            "source_features_sha256": digest(source / "features.npz"),
            "source_incidents_sha256": digest(source / "incidents.json"),
            "features_sha256": digest(destination / "features.npz"),
            "evidence_sha256": digest(evidence_file),
            "fault_labels_used_for_fold_specific_preprocessing": False,
            "shared_calibration_may_include_overlapping_incident_telemetry": True,
            "held_type_incidents_in_shared_calibration": True,
            "claim_scope": (
                "Supervised/history-type generalization with overlap-group "
                "purging under shared pre-window numeric calibration; not "
                "end-to-end row exclusion."
            ),
            "logs_available": False,
        },
    )
    return destination, evidence_file, names_file


def run_base_jobs(cfg: dict, output: Path, strict: bool) -> None:
    seeds = cfg["strict_unseen_seeds"] if strict else cfg["seeds"]
    gpu_lock = (ROOT / "artifacts/care/finetuning-gpu.lock").open("w")
    mapping = {}
    for dataset, dataset_cfg in cfg["datasets"].items():
        rows = json.loads((ROOT / dataset_cfg["data"] / "incidents.json").read_text())
        strict_data = (
            prepare_history_coldstart_data(cfg, dataset, output)[0]
            if strict
            else None
        )
        fault_types = sorted({row["fault_type"] for row in rows if row["split"] == "test"}) if strict else [None]
        for fault_type in fault_types:
            if fault_type is not None:
                mapping[slug(fault_type)] = fault_type
            for seed in seeds:
                destination = base_directory(output, dataset, seed, fault_type)
                evaluation = destination / "evaluation.json"
                if evaluation.exists():
                    saved = json.loads(evaluation.read_text())
                    if saved["excluded_fault_type"] != fault_type:
                        raise ValueError(f"Mismatched completed base run: {destination}")
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                command = train_base_command(
                    cfg,
                    dataset,
                    seed,
                    destination,
                    fault_type,
                    data_path=strict_data,
                )
                atomic_json(destination / "launch.json", {"command": command, "started_at": time.time()})
                if dataset_cfg["device"] == "cuda":
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                try:
                    with (destination / "train.log").open("a") as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                finally:
                    if dataset_cfg["device"] == "cuda":
                        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                if not evaluation.exists():
                    raise RuntimeError(f"Base training did not write {evaluation}")
        if not strict:
            for ablation in cfg["base"].get("ablations", []):
                for seed in seeds:
                    destination = ablation_base_directory(
                        output, dataset, seed, ablation
                    )
                    evaluation = destination / "evaluation.json"
                    if evaluation.exists():
                        continue
                    destination.mkdir(parents=True, exist_ok=True)
                    command = train_base_command(
                        cfg, dataset, seed, destination, ablation=ablation
                    )
                    atomic_json(
                        destination / "launch.json",
                        {"command": command, "started_at": time.time()},
                    )
                    if dataset_cfg["device"] == "cuda":
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                    try:
                        with (destination / "train.log").open("a") as log:
                            subprocess.run(
                                command,
                                cwd=ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                    finally:
                        if dataset_cfg["device"] == "cuda":
                            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                    if not evaluation.exists():
                        raise RuntimeError(
                            f"Base ablation did not write {evaluation}"
                        )
    if mapping:
        atomic_json(output / "strict-base/fault-type-map.json", mapping)


def load_base_probabilities(directory: Path) -> dict[str, np.ndarray]:
    values = np.load(directory / "predictions.npz")
    return {key: values[key] for key in values.files if key != "attention"}


class BasePredictor:
    """Load one base checkpoint once and replay deterministic perturbations."""

    # Evidence availability is metric/log/trace, while CAREBase consumes
    # metrics/traces/logs.  Keep this mapping explicit to prevent silent swaps.
    AVAILABILITY_INDEX = {"metrics": 0, "traces": 2, "logs": 1}

    def __init__(self, directory: Path, device_name: str, batch_size: int = 16):
        import torch

        self.torch = torch
        self.device = torch.device(device_name)
        checkpoint = torch.load(
            directory / "model.pt", map_location=self.device, weights_only=False
        )
        self.model = CAREBase(**checkpoint["model_args"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.adjacency_numpy = np.asarray(checkpoint["adjacency"], dtype=bool)
        self.adjacency = torch.as_tensor(self.adjacency_numpy, device=self.device)
        self.batch_size = batch_size

    def predict(
        self,
        feature_arrays: dict[str, np.ndarray],
        availability: np.ndarray,
        indices: np.ndarray,
        adjacency: np.ndarray | None = None,
    ) -> np.ndarray:
        probabilities = []
        adjacency_tensor = (
            self.adjacency
            if adjacency is None
            else self.torch.as_tensor(adjacency, device=self.device)
        )
        with self.torch.no_grad():
            chunks = np.array_split(
                indices, max(1, int(np.ceil(len(indices) / self.batch_size)))
            )
            for chunk in chunks:
                inputs = []
                for name in ("metrics", "traces", "logs"):
                    values = self.torch.as_tensor(
                        feature_arrays[name][chunk], dtype=self.torch.float32,
                        device=self.device,
                    )
                    position = self.AVAILABILITY_INDEX[name]
                    mask = self.torch.as_tensor(
                        availability[chunk, :, position], device=self.device
                    )
                    inputs.append(values * mask[:, :, None, None])
                logits, _, _ = self.model(*inputs, adjacency_tensor)
                probabilities.append(logits.softmax(-1).cpu().numpy())
        return np.concatenate(probabilities)

    def close(self) -> None:
        self.model.to("cpu")
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


def retain_adjacency(adjacency: np.ndarray, retention: float, seed: int) -> np.ndarray:
    """Drop observed undirected training edges while preserving self-loops."""
    adjacency = np.asarray(adjacency, dtype=bool)
    result = np.eye(len(adjacency), dtype=bool)
    rng = np.random.default_rng(seed)
    rows, columns = np.where(np.triu(adjacency, 1))
    keep = rng.random(len(rows)) < retention
    result[rows[keep], columns[keep]] = True
    result[columns[keep], rows[keep]] = True
    return result


def measure_pipeline_latency(
    predictor: BasePredictor,
    feature_arrays: dict[str, np.ndarray],
    x: np.ndarray,
    availability: np.ndarray,
    indices: np.ndarray,
    robust,
    generic,
    generic_weight: float,
    repeats: int = 3,
) -> dict:
    """Measure loaded-model, single-incident latency; exclude I/O/extraction."""
    base_times, correction_times, total_times = [], [], []
    for index in indices:
        selected = np.asarray([index], dtype=int)
        for _ in range(repeats):
            predictor.synchronize()
            start = time.perf_counter()
            predictor.predict(feature_arrays, availability, selected)
            predictor.synchronize()
            middle = time.perf_counter()
            raw = robust.predict(x[selected], availability[selected])
            generic_scores = generic.predict(x[selected], availability[selected])
            fuse_all(
                (1 - generic_weight, raw),
                (generic_weight, generic_scores),
            )
            end = time.perf_counter()
            base_times.append((middle - start) * 1000)
            correction_times.append((end - middle) * 1000)
            total_times.append((end - start) * 1000)

    def summarize(values: list[float]) -> dict:
        return {
            "mean_ms": float(np.mean(values)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "measurements": len(values),
        }

    return {
        "base": summarize(base_times),
        "evidence_and_fusion": summarize(correction_times),
        "total": summarize(total_times),
        "torch_cpu_threads": int(predictor.torch.get_num_threads()),
        "torch_interop_threads": int(predictor.torch.get_num_interop_threads()),
        "scope": "loaded models; numeric features precomputed; single incident; excludes I/O and telemetry extraction",
    }


def evidence_predict(ranker, x: np.ndarray, availability: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return ranker.predict(x[indices], availability[indices])


def fixed_availability(shape: tuple[int, ...], keep: tuple[str, ...]) -> np.ndarray:
    result = np.zeros((shape[0], shape[1], len(MODALITIES)), dtype=np.float32)
    for modality in keep:
        result[:, :, MODALITIES.index(modality)] = 1
    return result


def choose_unseen_generic_weight(
    x: np.ndarray,
    names: list[str],
    labels: np.ndarray,
    rows: list[dict],
    train_indices: np.ndarray,
    seed: int,
    ranker_cfg: dict,
) -> tuple[float, list[dict]]:
    raw_predictions, generic_predictions, held_labels = [], [], []
    records = []
    for fault_type in sorted({rows[i]["fault_type"] for i in train_indices}):
        fit_indices, held_indices, overlap_purged, held_groups = fault_type_holdout(
            rows, train_indices, fault_type
        )
        raw = fit_robust_ranker(x, names, labels, fit_indices, seed=seed, **ranker_cfg)
        generic = fit_robust_ranker(x, names, labels, fit_indices, seed=seed, type_agnostic=True, **ranker_cfg)
        availability = all_available((len(held_indices), x.shape[1]))
        raw_predictions.append(raw.predict(x[held_indices], availability))
        generic_predictions.append(generic.predict(x[held_indices], availability))
        held_labels.append(labels[held_indices])
        records.append({
            "fault_type": fault_type,
            "fit_count": len(fit_indices),
            "held_count": len(held_indices),
            "fit_incident_ids": [rows[i]["incident_id"] for i in fit_indices],
            "held_incident_ids": [rows[i]["incident_id"] for i in held_indices],
            "overlap_purged_incident_ids": [
                rows[i]["incident_id"] for i in overlap_purged
            ],
            "held_overlap_groups": held_groups,
        })
    raw_predictions = np.concatenate(raw_predictions)
    generic_predictions = np.concatenate(generic_predictions)
    held_labels = np.concatenate(held_labels)
    grid = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = fuse_all((1 - weight, raw_predictions), (weight, generic_predictions))
        grid.append({"generic_weight": weight, "metrics": ranking_metrics(scores, held_labels)})
    selected = max(grid, key=lambda item: (item["metrics"]["PR@1"], item["metrics"]["MAP@5"], -item["generic_weight"]))
    return float(selected["generic_weight"]), [{"selection_grid": grid}, *records]


def llm_scores(dataset: str, dataset_cfg: dict, rows: list[dict], services: list[str], test_indices: np.ndarray) -> tuple[np.ndarray | None, dict]:
    if dataset == "aiops2022":
        path = ROOT / dataset_cfg["legacy_llm"] / "comparison.npz"
        values = np.load(path)
        if not np.array_equal(values["indices"], test_indices):
            raise ValueError("AIOps LLM indices do not match frozen test split")
        inference_path = ROOT / dataset_cfg["legacy_llm"] / "lora-inference.json"
        inference = json.loads(inference_path.read_text())
        latency = [item["seconds"] for item in inference]
        input_tokens = [item["input_tokens"] for item in inference]
        return values["lora_42"], {
            "source": str(path),
            "kind": "constrained single-token LoRA",
            "inference_records": str(inference_path),
            "mean_seconds": float(np.mean(latency)),
            "p95_seconds": float(np.quantile(latency, 0.95)),
            "mean_input_tokens": float(np.mean(input_tokens)),
            "output_tokens_per_case": 1,
        }
    directory = ROOT / dataset_cfg["legacy_llm"]
    scores = []
    seconds = []
    input_characters = []
    output_characters = []
    valid = 0
    for index in test_indices:
        path = directory / f"case-{index:05d}.json"
        if not path.exists():
            return None, {"source": str(directory), "missing_case": int(index)}
        record = json.loads(path.read_text())
        seconds.append(record["seconds"])
        input_characters.append(len(json.dumps(record["input"], ensure_ascii=False)))
        output_characters.append(len(record["response"]))
        vector = np.zeros(len(services), dtype=np.float64)
        try:
            proposal = json.loads(record["response"])
            for service, score in proposal.get("causal_scores", {}).items():
                if service in services and isinstance(score, (float, int)) and score >= 0:
                    vector[services.index(service)] = score
            if vector.sum() > 0:
                valid += 1
            else:
                vector = np.asarray([record["input"]["telemetry"].get(s, {}).get("base_score", 0.0) for s in services])
        except (ValueError, TypeError, KeyError):
            vector = np.asarray([record["input"]["telemetry"].get(s, {}).get("base_score", 0.0) for s in services])
        scores.append(vector)
    return np.asarray(scores), {
        "source": str(directory),
        "kind": "archived free-form constrained proposal",
        "valid_score_vectors": valid,
        "mean_seconds": float(np.mean(seconds)),
        "p95_seconds": float(np.quantile(seconds, 0.95)),
        # The archived free-form run predates token counters.  Character counts
        # are recorded rather than presenting an unverifiable token estimate.
        "mean_input_characters": float(np.mean(input_characters)),
        "mean_output_characters": float(np.mean(output_characters)),
    }


def simplex_grid(base: np.ndarray, correction: np.ndarray, reasoner: np.ndarray, labels: np.ndarray, step: float) -> list[dict]:
    units = int(round(1 / step))
    result = []
    for base_units in range(units + 1):
        for correction_units in range(units + 1 - base_units):
            reasoner_units = units - base_units - correction_units
            weights = (base_units / units, correction_units / units, reasoner_units / units)
            scores = fuse_all((weights[0], base), (weights[1], correction), (weights[2], reasoner))
            result.append({
                "w_b": weights[0],
                "w_c": weights[1],
                "w_r": weights[2],
                "metrics": ranking_metrics(scores, labels),
            })
    return result


def archived_proposals(
    dataset: str,
    dataset_cfg: dict,
    services: list[str],
    test_indices: np.ndarray,
    reasoner: np.ndarray,
) -> list[dict]:
    """Recover label-free proposals produced before test labels were opened."""
    if dataset == "trainticket":
        proposals = []
        directory = ROOT / dataset_cfg["legacy_llm"]
        for index in test_indices:
            record = json.loads((directory / f"case-{index:05d}.json").read_text())
            try:
                proposal = json.loads(record["response"])
            except (ValueError, TypeError):
                proposal = {"root_cause": None, "causal_scores": {}}
            proposals.append(proposal)
        return proposals
    return [
        {
            "root_cause": services[int(scores.argmax())],
            "causal_scores": {
                service: float(score) for service, score in zip(services, scores)
            },
            "propagation_edges": [],
        }
        for scores in reasoner
    ]


def auditability_evidence(
    dataset: str,
    dataset_cfg: dict,
    rows: list[dict],
    test_indices: np.ndarray,
) -> dict:
    """Check provenance/faithfulness properties that do not require path gold."""
    training_ids = {
        row["incident_id"] for row in rows if row["split"] == "train"
    }
    fpg_path = ROOT / dataset_cfg["llm_sft"] / "fpg.json"
    fpg = json.loads(fpg_path.read_text())
    recorded_training_ids = set(
        fpg.get("training_incident_ids", fpg.get("training_ids", []))
    )
    support_ids = {
        support["incident_id"]
        for edge in fpg["edges"]
        for supports in edge["relations"].values()
        for support in supports
    }
    result = {
        "fpg_training_id_set_exact": recorded_training_ids == training_ids,
        "all_fpg_support_is_training_only": support_ids <= training_ids,
        "fpg_support_incidents": len(support_ids),
        "test_incidents_in_fpg_support": len(
            support_ids
            & {rows[index]["incident_id"] for index in test_indices}
        ),
        "path_ground_truth_available": False,
        "edge_ground_truth_available": False,
        "scope": (
            "These checks establish structural constraint and provenance only; "
            "they do not establish causal path correctness or confidence calibration."
        ),
    }
    if dataset != "trainticket":
        result.update({
            "generated_paths": 0,
            "generated_explanations": 0,
            "reason": "The compact AIOps LLM emits one constrained service token.",
        })
        return result

    proposed_edges = valid_edges = proposed_gate_pass = 0
    faithful_explanations = explanations = 0
    audited_edges = gate_pass_edges = rejected_edges = 0
    candidate_valid = parsed = 0
    for index in test_indices:
        record = json.loads(
            (ROOT / dataset_cfg["legacy_llm"] / f"case-{index:05d}.json").read_text()
        )
        try:
            proposal = json.loads(record["response"])
        except (TypeError, ValueError):
            continue
        parsed += 1
        observed = record["input"]
        candidates = set(observed["candidates"])
        candidate_valid += int(
            (proposal.get("root_cause") is None or proposal.get("root_cause") in candidates)
            and set(proposal.get("causal_scores", {})) <= candidates
        )
        facts = {
            fact["fact_id"]: fact for fact in observed["retrieved_facts"]
        }
        for edge in proposal.get("propagation_edges", []):
            proposed_edges += 1
            fact = facts.get(edge.get("fact_id"))
            valid = (
                fact is not None
                and fact["relation"] == "PROPAGATES_TO"
                and fact["source"] == edge.get("source")
                and fact["target"] == edge.get("target")
                and fact["support_count"] > 0
                and isinstance(edge.get("confidence"), (int, float))
                and 0 <= edge["confidence"] <= 1
            )
            valid_edges += int(valid)
            if valid:
                source = observed["telemetry"][edge["source"]]
                target = observed["telemetry"][edge["target"]]
                gate = (
                    source["anomalous"]
                    and target["anomalous"]
                    and source["onset"] is not None
                    and target["onset"] is not None
                    and source["onset"]
                    <= target["onset"] + observed["time_tolerance_seconds"]
                )
                proposed_gate_pass += int(gate)
        explanation = proposal.get("explanation", {})
        if "root_evidence" in explanation:
            explanations += 1
            root = proposal.get("root_cause")
            faithful_explanations += int(
                root in observed["telemetry"]
                and explanation["root_evidence"] == observed["telemetry"][root]
            )
        audit = record.get("audit", {})
        audited_edges += len(audit.get("edges", []))
        gate_pass_edges += sum(
            bool(edge.get("gate")) for edge in audit.get("edges", [])
        )
        rejected_edges += len(audit.get("rejected_proposals", []))
    result.update({
        "responses": len(test_indices),
        "parsed_responses": parsed,
        "json_valid_rate": parsed / len(test_indices) if len(test_indices) else None,
        "candidate_constraint_rate": candidate_valid / parsed if parsed else None,
        "proposed_edges": proposed_edges,
        "exact_citation_valid_edges": valid_edges,
        "exact_citation_valid_rate": (
            valid_edges / proposed_edges if proposed_edges else None
        ),
        "validly_cited_proposal_gate_pass": proposed_gate_pass,
        "validly_cited_proposal_gate_pass_rate": (
            proposed_gate_pass / valid_edges if valid_edges else None
        ),
        "validly_cited_proposal_gate_rejected": (
            valid_edges - proposed_gate_pass
        ),
        "validly_cited_proposal_gate_rejection_rate": (
            (valid_edges - proposed_gate_pass) / valid_edges
            if valid_edges else None
        ),
        "generated_explanations": explanations,
        "verbatim_input_evidence_faithful": faithful_explanations,
        "input_evidence_faithfulness_rate": (
            faithful_explanations / explanations if explanations else None
        ),
        "audited_candidate_edges": audited_edges,
        "gate_pass_edges": gate_pass_edges,
        "gate_pass_rate": (
            gate_pass_edges / audited_edges if audited_edges else None
        ),
        "invalid_citation_or_confidence_proposals": rejected_edges,
        "invalid_citation_or_confidence_rate": (
            rejected_edges / proposed_edges if proposed_edges else None
        ),
    })
    return result


def rule_parameter_sensitivity(
    dataset: str,
    dataset_cfg: dict,
    cfg: dict,
    rows: list[dict],
    metadata: dict,
    raw_features,
    test_indices: np.ndarray,
    base_all: np.ndarray,
    attention_all: np.ndarray,
    reasoner: np.ndarray,
) -> dict:
    """Evaluate the legacy equation-level rules without test-time fault labels."""
    services = metadata["services"]
    labels = raw_features["labels"][test_indices]
    proposals = archived_proposals(
        dataset, dataset_cfg, services, test_indices, reasoner
    )

    def observations(threshold: float):
        observed = build_observations(
            raw_features, metadata, rows, base_all, threshold=threshold
        )
        graph = build_fpg(rows, observed, metadata["step"])
        return observed, graph

    def evaluate(
        observed: list[dict],
        graph: list[dict],
        *,
        weights: tuple[float, float, float] = (0.6, 0.25, 0.15),
        hops: int = 3,
    ) -> dict:
        scores, changed, abstained = [], 0, 0
        for position, (index, proposal) in enumerate(zip(test_indices, proposals)):
            base = dict(zip(services, map(float, base_all[index])))
            candidates = list(proposal.get("causal_scores", {}))
            facts = retrieve(
                graph,
                candidates,
                exclude=rows[index]["incident_id"],
                hops=hops,
            )
            a = attention_all[index]
            attention = {
                (source, target): float(a[source_index, target_index])
                for source_index, source in enumerate(services)
                for target_index, target in enumerate(services)
                if source_index != target_index and a[source_index, target_index] > 0
            }
            try:
                audit = correct(
                    base,
                    observed[index],
                    facts,
                    proposal,
                    attention,
                    weights=weights,
                    hops=hops,
                    delta=metadata["step"],
                )
                final = audit["scores"]
                changed += int(audit["changed"])
                abstained += int(audit["reason"] == "LLM abstained")
            except (KeyError, TypeError, ValueError):
                final = base
                abstained += 1
            scores.append([final[service] for service in services])
        scores_array = np.asarray(scores)
        return {
            "metrics": ranking_metrics(scores_array, labels),
            "changed_top1": changed,
            "abstained_or_invalid": abstained,
        }

    default_observed, default_graph = observations(2.0)
    weights = []
    units = int(round(1 / cfg["sensitivity"]["simplex_step"]))
    for base_units in range(units + 1):
        for consistency_units in range(units + 1 - base_units):
            rule_units = units - base_units - consistency_units
            triple = (
                base_units / units,
                consistency_units / units,
                rule_units / units,
            )
            weights.append({
                "w_b": triple[0],
                "w_c": triple[1],
                "w_r": triple[2],
                **evaluate(default_observed, default_graph, weights=triple),
            })
    thresholds = []
    for threshold in cfg["sensitivity"]["health_thresholds"]:
        observed, graph = observations(float(threshold))
        thresholds.append({
            "anomaly_z_threshold": threshold,
            **evaluate(observed, graph),
        })
    hops = [
        {"hops": hops, **evaluate(default_observed, default_graph, hops=hops)}
        for hops in cfg["sensitivity"]["hops"]
    ]
    return {
        "protocol": (
            "Downstream replay of frozen archived LLM proposals. Thresholds and "
            "hops alter deterministic evidence/rules but do not regenerate prompts or LLM output."
        ),
        "weights": weights,
        "health_thresholds": thresholds,
        "retrieval_hops": hops,
        "health_definition": (
            "h_u = exp(-max(0, max_post_fault_abs_z - tau)); a service is "
            "anomalous iff any post-fault metric/trace max-abs-z >= tau"
        ),
        "fault_type_used_at_inference": False,
    }


def shift_time(values: np.ndarray, bins: int) -> np.ndarray:
    result = np.zeros_like(values)
    if bins == 0:
        return values.copy()
    if bins > 0:
        result[:, :, bins:, :] = values[:, :, :-bins, :]
    else:
        result[:, :, :bins, :] = values[:, :, -bins:, :]
    return result


def coarsen_time(values: np.ndarray, factor: int) -> np.ndarray:
    result = np.empty_like(values)
    for start in range(0, values.shape[2], factor):
        end = min(values.shape[2], start + factor)
        result[:, :, start:end, :] = values[:, :, start:end, :].mean(2, keepdims=True)
    return result


def perturbation_evidence(dataset: str, dataset_cfg: dict, output: Path, full_names: list[str]) -> dict[str, Path]:
    cache = output / dataset / "perturbations"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        return {name: Path(path) for name, path in json.loads(manifest_path.read_text()).items()}
    from evidence_reranker import extract, raw_tt_metrics

    data = ROOT / dataset_cfg["data"]
    rows = json.loads((data / "incidents.json").read_text())
    metadata = json.loads((data / "metadata.json").read_text())
    loaded = np.load(data / "features.npz")
    original = {name: loaded[name] for name in ("metrics", "traces", "logs")}
    raw = None
    if dataset == "trainticket":
        raw = raw_tt_metrics(rows, metadata["services"], metadata["metric_names"], metadata["step"], original["metrics"].shape[2])
    rpc = np.load(ROOT / dataset_cfg["rpc"] / "features.npz")["x"]
    rpc_names = json.loads((ROOT / dataset_cfg["rpc"] / "feature_names.json").read_text())
    variants = {}
    for modality in ("metrics", "traces"):
        for bins in (-2, -1, 1, 2):
            features = {name: values.copy() for name, values in original.items()}
            features[modality] = shift_time(features[modality], bins)
            raw_variant = shift_time(raw, bins) if raw is not None and modality == "metrics" else raw
            evidence, names = extract(features, metadata, rows, raw_variant)
            evidence = np.concatenate([evidence, rpc], axis=-1)
            names += rpc_names
            if modality == "traces":
                # Cached RPC residuals and call degrees cannot be shifted
                # exactly without replaying raw spans.  Remove them instead of
                # leaking the unshifted trace view into the drift condition.
                dependent = np.asarray(
                    [
                        name.startswith(("rpc:", "callers:", "callees:"))
                        for name in names
                    ],
                    dtype=bool,
                )
                evidence[:, :, dependent] = 0
            if names != full_names:
                raise ValueError("Perturbed evidence feature order differs from frozen cache")
            key = f"{modality}_shift_{bins:+d}"
            path = cache / f"{key}.npz"
            np.savez_compressed(path, x=evidence)
            variants[key] = str(path)
    for factor in (2, 4):
        features = {name: values.copy() for name, values in original.items()}
        features["metrics"] = coarsen_time(features["metrics"], factor)
        raw_variant = coarsen_time(raw, factor) if raw is not None else None
        evidence, names = extract(features, metadata, rows, raw_variant)
        evidence = np.concatenate([evidence, rpc], axis=-1)
        names += rpc_names
        if names != full_names:
            raise ValueError("Coarsened evidence feature order differs from frozen cache")
        key = f"metrics_coarsen_{factor}x"
        path = cache / f"{key}.npz"
        np.savez_compressed(path, x=evidence)
        variants[key] = str(path)
    atomic_json(manifest_path, variants)
    return {name: Path(path) for name, path in variants.items()}


def evaluate_dataset(cfg: dict, dataset: str, output: Path) -> None:
    dataset_cfg = cfg["datasets"][dataset]
    destination = output / dataset
    destination.mkdir(parents=True, exist_ok=True)
    data = ROOT / dataset_cfg["data"]
    rows = json.loads((data / "incidents.json").read_text())
    metadata = json.loads((data / "metadata.json").read_text())
    features_file = np.load(data / "features.npz")
    labels = features_file["labels"]
    feature_arrays = {name: features_file[name] for name in ("metrics", "traces", "logs")}
    train_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=int)
    test_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "test"], dtype=int)
    test_labels = labels[test_indices]
    test_rows = [rows[i] for i in test_indices]
    evidence_dir = ROOT / dataset_cfg["evidence"]
    x = np.load(evidence_dir / "full.npz")["x"]
    names = json.loads((evidence_dir / "full-names.json").read_text())
    ranker_cfg = {
        "partial_rates": cfg["evidence"]["partial_training_rates"],
        "trees": cfg["evidence"]["trees"],
        "min_samples_leaf": cfg["evidence"]["min_samples_leaf"],
    }
    base_weight = cfg["evidence"]["base_weight"]
    evidence_weight = cfg["evidence"]["evidence_weight"]
    perturbations = perturbation_evidence(dataset, dataset_cfg, output, names)
    atomic_json(
        destination / "auditability.json",
        auditability_evidence(dataset, dataset_cfg, rows, test_indices),
    )
    per_seed = []
    all_missing = []
    all_local_missing = []
    all_graph = []
    all_unseen = []
    all_sensitivity = []
    all_modalities = []

    for seed in cfg["seeds"]:
        run_dir = destination / f"seed-{seed}"
        result_path = run_dir / "result.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            per_seed.append(saved["main"])
            all_missing.append(saved["complete_missing"])
            all_local_missing.append(saved["local_missing"])
            all_graph.append(saved["graph_retention"])
            all_unseen.append(saved["unseen_fault"])
            all_sensitivity.append(saved["sensitivity"])
            all_modalities.append(saved["modality_models"])
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        base_dir = base_directory(output, dataset, seed)
        base_predictions = load_base_probabilities(base_dir)
        attention_all = np.load(base_dir / "predictions.npz")["attention"]
        base_predictor = BasePredictor(
            base_dir, dataset_cfg["device"], cfg["base"]["batch_size"]
        )
        base = base_predictions["full"][test_indices]
        evidence_suite_started = time.perf_counter()
        started = evidence_suite_started
        robust = fit_robust_ranker(x, names, labels, train_indices, seed=seed, **ranker_cfg)
        robust_fit_seconds = time.perf_counter() - started
        availability = all_available(x.shape)
        evidence = robust.predict(x[test_indices], availability[test_indices])
        standard = fit_robust_ranker(x, names, labels, train_indices, seed=seed, augment=False, append_flags=False, **ranker_cfg)
        standard_evidence = standard.predict(x[test_indices], availability[test_indices])
        no_flags = fit_robust_ranker(x, names, labels, train_indices, seed=seed, append_flags=False, **ranker_cfg)
        no_flags_evidence = no_flags.predict(x[test_indices], availability[test_indices])
        generic = fit_robust_ranker(x, names, labels, train_indices, seed=seed, type_agnostic=True, **ranker_cfg)
        generic_evidence = generic.predict(x[test_indices], availability[test_indices])
        standard_generic = fit_robust_ranker(
            x, names, labels, train_indices, seed=seed, augment=False,
            append_flags=False, type_agnostic=True, **ranker_cfg
        )
        no_flags_generic = fit_robust_ranker(
            x, names, labels, train_indices, seed=seed, append_flags=False,
            type_agnostic=True, **ranker_cfg
        )
        generic_weight, generic_selection = choose_unseen_generic_weight(
            x, names, labels, rows, train_indices, seed, ranker_cfg
        )
        evidence_suite_fit_seconds = time.perf_counter() - evidence_suite_started
        correction = fuse_all(
            (1 - generic_weight, evidence),
            (generic_weight, generic_evidence),
        )
        final = fuse_all((base_weight, base), (evidence_weight, correction))
        ablation_predictions = {
            name: load_base_probabilities(
                ablation_base_directory(output, dataset, seed, name)
            )
            for name in cfg["base"].get("ablations", [])
        }

        groups = feature_groups(names)
        standard_correction = fuse_all(
            (1 - generic_weight, standard_evidence),
            (
                generic_weight,
                standard_generic.predict(
                    x[test_indices], availability[test_indices]
                ),
            ),
        )
        no_flags_correction = fuse_all(
            (1 - generic_weight, no_flags_evidence),
            (
                generic_weight,
                no_flags_generic.predict(
                    x[test_indices], availability[test_indices]
                ),
            ),
        )
        module_ablations = {
            "full": measurement(final, test_labels, test_rows),
            "no_type_agnostic_expert": measurement(fuse_all((base_weight, base), (evidence_weight, evidence)), test_labels, test_rows),
            "no_missing_augmentation": measurement(fuse_all((base_weight, base), (evidence_weight, standard_correction)), test_labels, test_rows),
            "no_availability_flags": measurement(fuse_all((base_weight, base), (evidence_weight, no_flags_correction)), test_labels, test_rows),
            "top5_restriction": measurement(fuse(base, correction, evidence_weight, min(5, len(metadata["services"]))), test_labels, test_rows),
            "unaugmented_with_generic_top5": measurement(
                fuse(
                    base,
                    standard_correction,
                    evidence_weight,
                    min(5, len(metadata["services"])),
                ),
                test_labels,
                test_rows,
            ),
            "original_unaugmented_top5": measurement(
                fuse(
                    ablation_predictions["no_modality_dropout"]["full"][test_indices],
                    standard_evidence,
                    evidence_weight,
                    min(5, len(metadata["services"])),
                ),
                test_labels,
                test_rows,
            ),
            "type_agnostic_only": measurement(fuse_all((base_weight, base), (evidence_weight, generic_evidence)), test_labels, test_rows),
        }
        for key in ("rpc", "call_graph", "graph"):
            ablated_x = x.copy()
            ablated_x[:, :, groups[key]] = 0
            ranker = fit_robust_ranker(ablated_x, names, labels, train_indices, seed=seed, **ranker_cfg)
            scores = ranker.predict(ablated_x[test_indices], availability[test_indices])
            generic_ranker = fit_robust_ranker(
                ablated_x, names, labels, train_indices, seed=seed,
                type_agnostic=True, **ranker_cfg
            )
            generic_scores = generic_ranker.predict(
                ablated_x[test_indices], availability[test_indices]
            )
            combined = fuse_all(
                (1 - generic_weight, scores),
                (generic_weight, generic_scores),
            )
            module_ablations[f"no_{key}"] = measurement(fuse_all((base_weight, base), (evidence_weight, combined)), test_labels, test_rows)
        for name, predictions in ablation_predictions.items():
            if name in {"no_metrics", "no_traces", "no_logs"}:
                continue
            scores = predictions["full"][test_indices]
            module_ablations[f"base_{name}"] = measurement(
                fuse_all((base_weight, scores), (evidence_weight, correction)),
                test_labels,
                test_rows,
            )

        complete_missing = {}
        for missing in MODALITIES:
            observed = complete_modality_availability(x.shape, missing)
            robust_scores = robust.predict(x[test_indices], observed[test_indices])
            generic_scores = generic.predict(x[test_indices], observed[test_indices])
            standard_scores = standard.predict(x[test_indices], observed[test_indices])
            standard_generic_scores = standard_generic.predict(
                x[test_indices], observed[test_indices]
            )
            robust_correction = fuse_all(
                (1 - generic_weight, robust_scores),
                (generic_weight, generic_scores),
            )
            standard_correction_missing = fuse_all(
                (1 - generic_weight, standard_scores),
                (generic_weight, standard_generic_scores),
            )
            base_key = {
                "metric": "no_metrics", "log": "no_logs", "trace": "no_traces"
            }[missing]
            masked_base = base_predictions[base_key][test_indices]
            complete_missing[missing] = {
                "standard_evidence": ranking_metrics(standard_correction_missing, test_labels),
                "robust_evidence": ranking_metrics(robust_scores, test_labels),
                "generic_evidence": ranking_metrics(generic_scores, test_labels),
                "robust_correction": ranking_metrics(robust_correction, test_labels),
                "robust_method": ranking_metrics(fuse_all((base_weight, masked_base), (evidence_weight, robust_correction)), test_labels),
            }
            if "no_modality_dropout" in ablation_predictions:
                dropout_free_base = ablation_predictions["no_modality_dropout"][
                    base_key
                ][test_indices]
                complete_missing[missing]["no_modality_dropout_method"] = (
                    ranking_metrics(
                        fuse_all(
                            (base_weight, dropout_free_base),
                            (evidence_weight, robust_correction),
                        ),
                        test_labels,
                    )
                )

        local_missing = []
        for modality in (*MODALITIES, None):
            for rate in cfg["missingness"]["rates"]:
                for mask_seed in cfg["missingness"]["mask_seeds"]:
                    observed = local_missing_availability(x.shape, rate, mask_seed, modality)
                    observed_test = observed[test_indices]
                    robust_scores = robust.predict(x[test_indices], observed[test_indices])
                    generic_scores = generic.predict(
                        x[test_indices], observed[test_indices]
                    )
                    standard_scores = standard.predict(x[test_indices], observed[test_indices])
                    standard_generic_scores = standard_generic.predict(
                        x[test_indices], observed[test_indices]
                    )
                    robust_correction = fuse_all(
                        (1 - generic_weight, robust_scores),
                        (generic_weight, generic_scores),
                    )
                    standard_correction_missing = fuse_all(
                        (1 - generic_weight, standard_scores),
                        (generic_weight, standard_generic_scores),
                    )
                    masked_base = base_predictor.predict(
                        feature_arrays, observed, test_indices
                    )
                    local_missing.append({
                        "modality": modality or "mixed",
                        "missing_rate": rate,
                        "realized_missing_rate": float(1 - observed_test.mean()),
                        "realized_missing_rate_by_modality": {
                            name: float(1 - observed_test[:, :, position].mean())
                            for position, name in enumerate(MODALITIES)
                        },
                        "mask_seed": mask_seed,
                        "standard_evidence": ranking_metrics(standard_correction_missing, test_labels),
                        "robust_evidence": ranking_metrics(robust_scores, test_labels),
                        "generic_evidence": ranking_metrics(generic_scores, test_labels),
                        "robust_correction": ranking_metrics(robust_correction, test_labels),
                        "robust_method": ranking_metrics(fuse_all((base_weight, masked_base), (evidence_weight, robust_correction)), test_labels),
                    })

        graph_retention = []
        for retention in cfg["sensitivity"]["graph_retention"]:
            for mask_seed in cfg["missingness"]["mask_seeds"]:
                masked_x = mask_feature_group(x, names, "graph", retention, mask_seed)
                scores = robust.predict(masked_x[test_indices], availability[test_indices])
                generic_scores = generic.predict(
                    masked_x[test_indices], availability[test_indices]
                )
                combined = fuse_all(
                    (1 - generic_weight, scores),
                    (generic_weight, generic_scores),
                )
                retained = retain_adjacency(
                    base_predictor.adjacency_numpy, retention, mask_seed
                )
                graph_base = base_predictor.predict(
                    feature_arrays, availability, test_indices, adjacency=retained
                )
                graph_retention.append({
                    "retention": retention,
                    "mask_seed": mask_seed,
                    "retained_training_edges": int(
                        (np.triu(retained, 1)).sum()
                    ),
                    "metrics": ranking_metrics(fuse_all((base_weight, graph_base), (evidence_weight, combined)), test_labels),
                })

        modality_models = {
            "evidence_only": {},
            "end_to_end_leave_one_out": {},
        }
        fixed_modality_scores = {}
        modality_sets = [
            ("metric",), ("log",), ("trace",),
            ("metric", "log"), ("metric", "trace"), ("log", "trace"),
            tuple(MODALITIES),
        ]
        for keep in modality_sets:
            observed = fixed_availability(x.shape, keep)
            ranker = fit_fixed_availability_ranker(x, names, labels, train_indices, observed, seed=seed, trees=ranker_cfg["trees"], min_samples_leaf=ranker_cfg["min_samples_leaf"])
            generic_ranker = fit_fixed_availability_ranker(
                x, names, labels, train_indices, observed, seed=seed,
                type_agnostic=True, trees=ranker_cfg["trees"],
                min_samples_leaf=ranker_cfg["min_samples_leaf"]
            )
            raw_scores = ranker.predict(x[test_indices], observed[test_indices])
            generic_scores = generic_ranker.predict(
                x[test_indices], observed[test_indices]
            )
            scores = fuse_all(
                (1 - generic_weight, raw_scores),
                (generic_weight, generic_scores),
            )
            fixed_modality_scores[keep] = scores
            modality_models["evidence_only"]["+".join(keep)] = ranking_metrics(
                scores, test_labels
            )
        base_ablation_for_missing = {
            "metric": "no_metrics", "trace": "no_traces", "log": "no_logs"
        }
        for missing, ablation in base_ablation_for_missing.items():
            keep = tuple(modality for modality in MODALITIES if modality != missing)
            scores = fixed_modality_scores[keep]
            ablated_base = ablation_predictions[ablation]["full"][test_indices]
            modality_models["end_to_end_leave_one_out"][missing] = ranking_metrics(
                fuse_all(
                    (base_weight, ablated_base),
                    (evidence_weight, scores),
                ),
                test_labels,
            )

        unseen_fault = {"selection_scope": "nested within each outer held fault type", "fault_types": {}}
        unseen_scores = np.zeros_like(evidence)
        for fault_type in sorted({row["fault_type"] for row in test_rows}):
            fit_indices, _, overlap_purged, held_groups = fault_type_holdout(
                rows,
                train_indices,
                fault_type,
                held_scope_indices=np.arange(len(rows)),
            )
            unseen_generic_weight, unseen_selection = choose_unseen_generic_weight(
                x, names, labels, rows, fit_indices, seed, ranker_cfg
            )
            held_positions = np.asarray([i for i, row in enumerate(test_rows) if row["fault_type"] == fault_type], dtype=int)
            held_indices = test_indices[held_positions]
            raw_unseen = fit_robust_ranker(x, names, labels, fit_indices, seed=seed, **ranker_cfg)
            generic_unseen = fit_robust_ranker(x, names, labels, fit_indices, seed=seed, type_agnostic=True, **ranker_cfg)
            raw_scores = raw_unseen.predict(x[held_indices], availability[held_indices])
            generic_scores = generic_unseen.predict(x[held_indices], availability[held_indices])
            fallback_scores = fuse_all(
                (1 - unseen_generic_weight, raw_scores),
                (unseen_generic_weight, generic_scores),
            )
            unseen_scores[held_positions] = fallback_scores
            unseen_fault["fault_types"][fault_type] = {
                "train_count": len(fit_indices),
                "test_count": len(held_indices),
                "overlap_purged_count": len(overlap_purged),
                "overlap_purged_incident_ids": [
                    rows[i]["incident_id"] for i in overlap_purged
                ],
                "held_overlap_groups": held_groups,
                "selected_generic_weight": unseen_generic_weight,
                "selection": unseen_selection,
                "standard": ranking_metrics(raw_scores, labels[held_indices]),
                "fallback": ranking_metrics(fallback_scores, labels[held_indices]),
            }
        unseen_fault["aggregate"] = ranking_metrics(unseen_scores, test_labels)

        sensitivity = []
        for top_k in cfg["sensitivity"]["top_k"]:
            actual_k = len(metadata["services"]) if top_k == "all" else int(top_k)
            for weight in cfg["sensitivity"]["evidence_weights"]:
                scores = fuse(base, correction, weight, actual_k)
                sensitivity.append({
                    "parameter": "candidate_and_fusion",
                    "top_k": top_k,
                    "evidence_weight": weight,
                    "metrics": ranking_metrics(scores, test_labels),
                })
        for trees in cfg["sensitivity"]["tree_counts"]:
            tuned = fit_robust_ranker(
                x,
                names,
                labels,
                train_indices,
                seed=seed,
                partial_rates=ranker_cfg["partial_rates"],
                trees=trees,
                min_samples_leaf=ranker_cfg["min_samples_leaf"],
            )
            tuned_generic = fit_robust_ranker(
                x, names, labels, train_indices, seed=seed,
                type_agnostic=True, partial_rates=ranker_cfg["partial_rates"],
                trees=trees, min_samples_leaf=ranker_cfg["min_samples_leaf"],
            )
            scores = fuse_all(
                (1 - generic_weight, tuned.predict(x[test_indices], availability[test_indices])),
                (generic_weight, tuned_generic.predict(x[test_indices], availability[test_indices])),
            )
            sensitivity.append({
                "parameter": "tree_count",
                "tree_count": trees,
                "metrics": ranking_metrics(
                    fuse_all((base_weight, base), (evidence_weight, scores)),
                    test_labels,
                ),
            })
        for leaf in cfg["sensitivity"]["min_samples_leaf"]:
            tuned = fit_robust_ranker(
                x,
                names,
                labels,
                train_indices,
                seed=seed,
                partial_rates=ranker_cfg["partial_rates"],
                trees=ranker_cfg["trees"],
                min_samples_leaf=leaf,
            )
            tuned_generic = fit_robust_ranker(
                x, names, labels, train_indices, seed=seed,
                type_agnostic=True, partial_rates=ranker_cfg["partial_rates"],
                trees=ranker_cfg["trees"], min_samples_leaf=leaf,
            )
            scores = fuse_all(
                (1 - generic_weight, tuned.predict(x[test_indices], availability[test_indices])),
                (generic_weight, tuned_generic.predict(x[test_indices], availability[test_indices])),
            )
            sensitivity.append({
                "parameter": "min_samples_leaf",
                "min_samples_leaf": leaf,
                "metrics": ranking_metrics(
                    fuse_all((base_weight, base), (evidence_weight, scores)),
                    test_labels,
                ),
            })

        perturbation_results = {}
        perturbation_predictions = {}
        for name, path in perturbations.items():
            perturbed_x = np.load(path)["x"]
            raw_scores = robust.predict(
                perturbed_x[test_indices], availability[test_indices]
            )
            generic_scores = generic.predict(
                perturbed_x[test_indices], availability[test_indices]
            )
            scores = fuse_all(
                (1 - generic_weight, raw_scores),
                (generic_weight, generic_scores),
            )
            # Reconstruct only the raw modality inputs needed by the base model.
            modified = {key: value.copy() for key, value in feature_arrays.items()}
            if name.startswith("metrics_shift_"):
                modified["metrics"] = shift_time(modified["metrics"], int(name.rsplit("_", 1)[1]))
            elif name.startswith("traces_shift_"):
                modified["traces"] = shift_time(modified["traces"], int(name.rsplit("_", 1)[1]))
            elif name.startswith("metrics_coarsen_"):
                modified["metrics"] = coarsen_time(modified["metrics"], int(name.split("_")[-1][:-1]))
            perturbed_base = base_predictor.predict(modified, availability, test_indices)
            perturbation_results[name] = ranking_metrics(
                fuse_all((base_weight, perturbed_base), (evidence_weight, scores)),
                test_labels,
            )
            perturbation_predictions[name] = fuse_all(
                (base_weight, perturbed_base), (evidence_weight, scores)
            )

        llm, llm_meta = llm_scores(dataset, dataset_cfg, rows, metadata["services"], test_indices)
        reasoner = None
        simplex = None
        rule_sensitivity = None
        llm_comparison = {"metadata": llm_meta}
        if llm is not None:
            reasoner = llm
            deployment_selection_path = (
                ROOT
                / cfg["datasets"]["trainticket"]["compact_llm"]
                / "selection.json"
            )
            deployment_selection = json.loads(
                deployment_selection_path.read_text()
            )
            deployment_llm_weight = float(
                deployment_selection["selected_weight"]
            )
            deployed = fuse_all(
                (1 - deployment_llm_weight, final),
                (deployment_llm_weight, reasoner),
            )
            llm_comparison.update({
                "llm_only": ranking_metrics(reasoner, test_labels),
                "robust_no_llm": ranking_metrics(final, test_labels),
                "robust_plus_llm_25": ranking_metrics(fuse_all((0.75, final), (0.25, reasoner)), test_labels),
                "fusion_weight": 0.25,
                "fusion_role": "prespecified diagnostic; not selected as the main method",
                "deployment_selection_source": str(deployment_selection_path),
                "deployment_selected_weight": deployment_llm_weight,
                "deployment_metrics": ranking_metrics(deployed, test_labels),
                "deployment_llm_calls": (
                    len(test_indices) if deployment_llm_weight > 0 else 0
                ),
                "deployment_policy": (
                    "One TrainTicket validation policy is fixed across both datasets; "
                    "weight zero disables LLM calls."
                ),
            })
            simplex = simplex_grid(base, correction, reasoner, test_labels, cfg["sensitivity"]["simplex_step"])
            rule_sensitivity = rule_parameter_sensitivity(
                dataset,
                dataset_cfg,
                cfg,
                rows,
                metadata,
                {**feature_arrays, "labels": labels},
                test_indices,
                base_predictions["full"],
                attention_all,
                reasoner,
            )

        latency = measure_pipeline_latency(
            base_predictor,
            feature_arrays,
            x,
            availability,
            test_indices,
            robust,
            generic,
            generic_weight,
        )

        np.savez_compressed(
            run_dir / "predictions.npz",
            indices=test_indices,
            labels=test_labels,
            base=base,
            standard_evidence=standard_evidence,
            robust_evidence=evidence,
            generic_evidence=generic_evidence,
            robust_correction=correction,
            selected_generic_weight=np.asarray(generic_weight),
            final=final,
            unseen_fallback=unseen_scores,
            **({"llm": reasoner} if reasoner is not None else {}),
        )
        np.savez_compressed(
            run_dir / "perturbation_predictions.npz",
            indices=test_indices,
            labels=test_labels,
            **perturbation_predictions,
        )
        with (run_dir / "robust_evidence_model.pkl").open("wb") as handle:
            pickle.dump(robust.model, handle)
        with (run_dir / "generic_evidence_model.pkl").open("wb") as handle:
            pickle.dump(generic.model, handle)
        main = {
            "seed": seed,
            "base": measurement(base, test_labels, test_rows),
            "standard_evidence": measurement(standard_evidence, test_labels, test_rows),
            "robust_evidence": measurement(evidence, test_labels, test_rows),
            "generic_evidence": measurement(generic_evidence, test_labels, test_rows),
            "robust_correction": measurement(correction, test_labels, test_rows),
            "robust_method": measurement(final, test_labels, test_rows),
            "selected_generic_weight": generic_weight,
            "generic_weight_selection": generic_selection,
            "paired_vs_base": paired_statistics(base, final, test_labels),
            "candidate_recall": {str(k): float(np.mean([label in np.argsort(-score, kind="stable")[:k] for score, label in zip(base, test_labels)])) for k in (1, 3, 5, len(metadata["services"]))},
            "fit_seconds": robust_fit_seconds,
            "evidence_suite_fit_seconds": evidence_suite_fit_seconds,
            "evidence_suite_fit_scope": (
                "robust, standard, no-flag, generic variants and training-only "
                "generic-weight selection used by the main configuration"
            ),
            "latency": latency,
        }
        result = {
            "dataset": dataset,
            "seed": seed,
            "main": main,
            "module_ablations": module_ablations,
            "complete_missing": complete_missing,
            "local_missing": local_missing,
            "graph_retention": graph_retention,
            "modality_models": modality_models,
            "unseen_fault": unseen_fault,
            "sensitivity": sensitivity,
            "perturbations": perturbation_results,
            "perturbation_protocol": {
                "metric_shift": "shift prepared metric bins and raw TrainTicket metric bins",
                "trace_shift": "shift prepared trace-summary bins; conservatively remove cached RPC/call-degree evidence because raw-span replay is unavailable",
                "metric_coarsening": "replace each adjacent 2/4-bin block by its block mean",
                "graph_retention": "drop cached graph/RPC evidence and historical base adjacency edges at the requested retention; not a coherent service-topology rewrite",
            },
            "llm": llm_comparison,
            "simplex": simplex,
            "legacy_rule_sensitivity": rule_sensitivity,
        }
        atomic_json(result_path, result)
        base_predictor.close()
        per_seed.append(main)
        all_missing.append(complete_missing)
        all_local_missing.append(local_missing)
        all_graph.append(graph_retention)
        all_unseen.append(unseen_fault)
        all_sensitivity.append(sensitivity)
        all_modalities.append(modality_models)

    summary = {
        "dataset": dataset,
        "seeds": cfg["seeds"],
        "main": {
            method: mean_std([seed_result[method]["metrics"] for seed_result in per_seed])
            for method in (
                "base", "standard_evidence", "robust_evidence",
                "generic_evidence", "robust_correction", "robust_method",
            )
        },
        "paired_vs_base": [seed_result["paired_vs_base"] for seed_result in per_seed],
        "candidate_recall": {
            key: {"mean": float(np.mean([seed_result["candidate_recall"][key] for seed_result in per_seed])), "std": float(np.std([seed_result["candidate_recall"][key] for seed_result in per_seed], ddof=1))}
            for key in per_seed[0]["candidate_recall"]
        },
        "per_seed_files": [str(destination / f"seed-{seed}" / "result.json") for seed in cfg["seeds"]],
    }
    atomic_json(destination / "summary.json", summary)


def evaluate_strict(cfg: dict, output: Path) -> None:
    for dataset, dataset_cfg in cfg["datasets"].items():
        data, evidence_file, names_file = prepare_history_coldstart_data(
            cfg, dataset, output
        )
        rows = json.loads((data / "incidents.json").read_text())
        labels = np.load(data / "features.npz")["labels"]
        train_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=int)
        test_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "test"], dtype=int)
        x = np.load(evidence_file)["x"]
        names = json.loads(names_file.read_text())
        ranker_cfg = {
            "partial_rates": cfg["evidence"]["partial_training_rates"],
            "trees": cfg["evidence"]["trees"],
            "min_samples_leaf": cfg["evidence"]["min_samples_leaf"],
        }
        records = []
        prediction_arrays = {}
        test_position = {index: position for position, index in enumerate(test_indices)}
        for seed in cfg["strict_unseen_seeds"]:
            seed_arrays = {
                name: np.zeros((len(test_indices), x.shape[1]), dtype=np.float64)
                for name in ("base", "raw_evidence", "fallback_evidence", "without_generic", "robust_method")
            }
            for fault_type in sorted({rows[i]["fault_type"] for i in test_indices}):
                target = np.asarray([i for i in test_indices if rows[i]["fault_type"] == fault_type], dtype=int)
                fit_indices, _, overlap_purged, held_groups = fault_type_holdout(
                    rows,
                    train_indices,
                    fault_type,
                    held_scope_indices=np.arange(len(rows)),
                )
                generic_weight, selection = choose_unseen_generic_weight(
                    x, names, labels, rows, fit_indices, seed, ranker_cfg
                )
                base_dir = base_directory(output, dataset, seed, fault_type)
                base = load_base_probabilities(base_dir)["full"][target]
                raw = fit_robust_ranker(x, names, labels, fit_indices, seed=seed, **ranker_cfg)
                generic = fit_robust_ranker(x, names, labels, fit_indices, seed=seed, type_agnostic=True, **ranker_cfg)
                available = all_available((len(target), x.shape[1]))
                raw_scores = raw.predict(x[target], available)
                generic_scores = generic.predict(x[target], available)
                fallback = fuse_all((1 - generic_weight, raw_scores), (generic_weight, generic_scores))
                without_generic = fuse_all(
                    (cfg["evidence"]["base_weight"], base),
                    (cfg["evidence"]["evidence_weight"], raw_scores),
                )
                final = fuse_all(
                    (cfg["evidence"]["base_weight"], base),
                    (cfg["evidence"]["evidence_weight"], fallback),
                )
                positions = np.asarray([test_position[index] for index in target])
                for name, values in (
                    ("base", base),
                    ("raw_evidence", raw_scores),
                    ("fallback_evidence", fallback),
                    ("without_generic", without_generic),
                    ("robust_method", final),
                ):
                    seed_arrays[name][positions] = values
                records.append({
                    "dataset": dataset,
                    "seed": seed,
                    "fault_type": fault_type,
                    "fault_slug": slug(fault_type),
                    "train_count": len(fit_indices),
                    "test_count": len(target),
                    "overlap_purged_count": len(overlap_purged),
                    "overlap_purged_incident_ids": [
                        rows[i]["incident_id"] for i in overlap_purged
                    ],
                    "held_overlap_groups": held_groups,
                    "selected_generic_weight": generic_weight,
                    "selection": selection,
                    "base": ranking_metrics(base, labels[target]),
                    "raw_evidence": ranking_metrics(raw_scores, labels[target]),
                    "fallback_evidence": ranking_metrics(fallback, labels[target]),
                    "without_generic": ranking_metrics(
                        without_generic, labels[target]
                    ),
                    "robust_method": ranking_metrics(final, labels[target]),
                    "preprocessing": (
                        "shared pre-window numeric calibration (not refit and may "
                        "include telemetry overlapping another incident), then "
                        "per-incident pre-window normalization; logs unavailable"
                    ),
                    "claim_scope": (
                        "held-type labeled examples, supervised fits, and topology "
                        "history excluded with overlap-group purging; shared numeric "
                        "calibration retained"
                    ),
                })
            for name, values in seed_arrays.items():
                prediction_arrays[f"{name}_seed_{seed}"] = values
        atomic_json(output / dataset / "strict-unseen.json", {"records": records})
        np.savez_compressed(
            output / dataset / "strict-unseen-predictions.npz",
            indices=test_indices,
            labels=labels[test_indices],
            **prediction_arrays,
        )


def establish_protocol(cfg_path: Path, cfg: dict, output: Path) -> str:
    inputs = {}
    for dataset, dataset_cfg in cfg["datasets"].items():
        data = ROOT / dataset_cfg["data"]
        evidence = ROOT / dataset_cfg["evidence"]
        rpc = ROOT / dataset_cfg["rpc"]
        llm = ROOT / dataset_cfg["legacy_llm"]
        llm_sft = ROOT / dataset_cfg["llm_sft"]
        inputs[dataset] = {
            **{
                name: digest(data / name)
                for name in ("features.npz", "incidents.json", "metadata.json")
            },
            "evidence_full.npz": digest(evidence / "full.npz"),
            "evidence_full-names.json": digest(evidence / "full-names.json"),
            "rpc_features.npz": digest(rpc / "features.npz"),
            "rpc_feature_names.json": digest(rpc / "feature_names.json"),
            "llm_json_files": directory_digest(llm, "*.json"),
            "llm_comparison.npz": (
                digest(llm / "comparison.npz")
                if (llm / "comparison.npz").exists()
                else None
            ),
            "llm_sft_fpg.json": digest(llm_sft / "fpg.json"),
            "llm_sft_test_inputs.json": digest(llm_sft / "test_inputs.json"),
        }
        if dataset == "trainticket":
            rows = json.loads((data / "incidents.json").read_text())
            raw_metrics = sorted(
                {Path(row["source"]).parent / "metrics.json.gz" for row in rows},
                key=str,
            )
            inputs[dataset]["raw_metrics_files"] = len(raw_metrics)
            inputs[dataset]["raw_metrics_sha256"] = files_digest(raw_metrics)
            inputs[dataset]["compact_llm_selection.json"] = digest(
                ROOT / dataset_cfg["compact_llm"] / "selection.json"
            )
            inputs[dataset]["compact_llm_json_files"] = directory_digest(
                ROOT / dataset_cfg["compact_llm"], "*.json"
            )
            inputs[dataset]["llm_train_results.json"] = digest(
                ROOT / "artifacts/care/qwen3-care-lora-trainticket/train_results.json"
            )
        else:
            inputs[dataset]["llm_train_results.json"] = digest(
                llm / "adapter/train_results.json"
            )
    protocol = {
        "configuration": cfg,
        "configuration_sha256": digest(cfg_path),
        "sources": {name: digest(ROOT / "scripts/care" / name) for name in SOURCES},
        "inputs": inputs,
        "review_sha256": digest(ROOT / "review/review_care.txt"),
        "created_at": time.time(),
    }
    path = output / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text())
        comparable_existing = {key: value for key, value in existing.items() if key != "created_at"}
        comparable_new = {key: value for key, value in protocol.items() if key != "created_at"}
        if comparable_existing != comparable_new:
            raise ValueError("Experiment protocol changed; use a new output directory")
    else:
        atomic_json(path, protocol)
    return digest(path)


def verify_completion(cfg: dict, output: Path) -> None:
    missing = []
    for dataset in cfg["datasets"]:
        for relative in (
            "summary.json", "auditability.json", "strict-unseen.json",
            "strict-unseen-predictions.npz",
        ):
            if not (output / dataset / relative).exists():
                missing.append(str(output / dataset / relative))
        for seed in cfg["seeds"]:
            for relative in (
                "result.json", "predictions.npz", "perturbation_predictions.npz",
            ):
                if not (output / dataset / f"seed-{seed}" / relative).exists():
                    missing.append(
                        str(output / dataset / f"seed-{seed}" / relative)
                    )
    if not (output / "llm_costs.json").exists():
        missing.append(str(output / "llm_costs.json"))
    if missing:
        raise RuntimeError("Incomplete experiment outputs: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/care/review-response-v1.json"))
    parser.add_argument(
        "--phase",
        choices=[
            "train-main", "evaluate", "train-strict", "evaluate-strict",
            "costs", "verify", "all",
        ],
        default="all",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())
    import torch

    torch.set_num_threads(int(cfg["base"]["threads"]))
    try:
        torch.set_num_interop_threads(int(cfg["base"]["threads"]))
    except RuntimeError:
        # Safe when a resumed interpreter has already initialized interop.
        pass
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = establish_protocol(cfg_path, cfg, output)
    state = {"pid": os.getpid(), "phase": args.phase, "state": "running", "started_at": time.time(), "protocol_sha256": protocol}
    atomic_json(output / "status.json", state)
    try:
        if args.phase in ("train-main", "all"):
            state["stage"] = "train-main"; atomic_json(output / "status.json", state)
            run_base_jobs(cfg, output, strict=False)
        if args.phase in ("evaluate", "all"):
            state["stage"] = "evaluate"; atomic_json(output / "status.json", state)
            for dataset in cfg["datasets"]:
                evaluate_dataset(cfg, dataset, output)
        if args.phase in ("train-strict", "all"):
            state["stage"] = "train-strict"; atomic_json(output / "status.json", state)
            run_base_jobs(cfg, output, strict=True)
        if args.phase in ("evaluate-strict", "all"):
            state["stage"] = "evaluate-strict"; atomic_json(output / "status.json", state)
            evaluate_strict(cfg, output)
        if args.phase in ("costs", "all"):
            state["stage"] = "costs"; atomic_json(output / "status.json", state)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/care/profile_llm_costs.py"),
                    "--output",
                    str(output / "llm_costs.json"),
                ],
                cwd=ROOT,
                check=True,
            )
        if args.phase in ("verify", "all"):
            state["stage"] = "verify"; atomic_json(output / "status.json", state)
            verify_completion(cfg, output)
        state.update(state="complete", finished_at=time.time())
        atomic_json(output / "status.json", state)
    except BaseException as error:
        state.update(state="failed", error=repr(error), failed_at=time.time())
        atomic_json(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
