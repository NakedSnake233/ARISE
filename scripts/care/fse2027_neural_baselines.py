"""Frozen stochastic TraceAnomaly, Eadro, and MULAN adapters."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from fse2027_baseline_common import (
        OUTPUT, ROOT, DatasetBundle, adjacency_for_incident, atomic_json,
        choose_configuration, complete_or_fallback, fallback_scores, load_dataset,
        modality_views, random_walk, row_unit_interval, save_method_run, sha256,
        summarize_repeats,
    )
except ImportError:
    from .fse2027_baseline_common import (
        OUTPUT, ROOT, DatasetBundle, adjacency_for_incident, atomic_json,
        choose_configuration, complete_or_fallback, fallback_scores, load_dataset,
        modality_views, random_walk, row_unit_interval, save_method_run, sha256,
        summarize_repeats,
    )


PROTOCOL = OUTPUT / "BASELINE_PROTOCOL_FROZEN.json"
EXPECTED_PROTOCOL_SHA256 = "81d0a737e261cddfda665d4a59fb81d567e371e846893dee30f58631e3d676da"
RESULT_SEEDS = [42, 43, 44, 45, 46]


def verify_protocol() -> dict[str, Any]:
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen baseline protocol hash mismatch")
    return json.loads(PROTOCOL.read_text())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def configurations(grid: dict[str, list[Any]]):
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def standardized_descriptors(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for modality in ("metrics", "logs", "traces"):
        _, _, descriptor = modality_views(bundle.features[modality], bundle.split_bin)
        fit = descriptor[bundle.fit].reshape(-1, descriptor.shape[-1])
        center = np.mean(fit, axis=0)
        scale = np.std(fit, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        output[modality] = np.nan_to_num((descriptor - center) / scale).astype(np.float32)
    return output


def adjacency_tensor(bundle: DatasetBundle) -> np.ndarray:
    matrices = []
    for index in range(len(bundle.rows)):
        matrix = adjacency_for_incident(bundle, index)
        matrix = np.maximum(matrix, 0)
        matrices.append(matrix.astype(np.float32))
    return np.stack(matrices)


class TraceVAE(nn.Module):
    def __init__(self, width: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, width))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        if self.training:
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            latent = mean
        return self.decoder(latent), mean, logvar


def traceanomaly_fit_score(
    bundle: DatasetBundle, config: dict[str, Any], seed: int
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    set_seed(seed)
    trace = np.nan_to_num(bundle.features["traces"].astype(np.float32))
    train = trace[bundle.fit, :, : bundle.split_bin].reshape(-1, trace.shape[-1])
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    train = (train - center) / scale
    model = TraceVAE(trace.shape[-1], int(config["hidden_dim"]), int(config["latent_dim"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    tensor = torch.as_tensor(train, dtype=torch.float32)
    generator = np.random.default_rng(seed)
    batch_size = min(256, len(tensor))
    for _ in range(int(config["epochs"])):
        model.train()
        order = generator.permutation(len(tensor))
        for start in range(0, len(order), batch_size):
            x = tensor[order[start : start + batch_size]]
            reconstruction, mean, logvar = model(x)
            reconstruction_loss = F.mse_loss(reconstruction, x)
            kl = -0.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
            loss = reconstruction_loss + 0.01 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    post = (trace[:, :, bundle.split_bin :] - center) / scale
    flat = torch.as_tensor(post.reshape(-1, trace.shape[-1]), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        reconstruction, _, _ = model(flat)
        error = (reconstruction - flat).square().mean(dim=-1).numpy()
    error = error.reshape(post.shape[:3])
    aggregation = config["aggregation"]
    if aggregation == "mean":
        scores = error.mean(axis=-1)
    elif aggregation == "max":
        scores = error.max(axis=-1)
    else:
        scores = np.quantile(error, 0.90, axis=-1)
    return scores, copy.deepcopy(model.state_dict())


class DenseGraphAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden dimension must be divisible by heads")
        self.heads = heads
        self.head_width = hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, services, hidden = x.shape
        reshape = lambda value: value.view(batch, services, self.heads, self.head_width).transpose(1, 2)
        query, key, value = reshape(self.query(x)), reshape(self.key(x)), reshape(self.value(x))
        logits = torch.einsum("bhtd,bhsd->bhts", query, key) / np.sqrt(self.head_width)
        # adjacency is [source, target]; a target attends to observed sources.
        incoming = adjacency.transpose(1, 2)
        mask = incoming > 0
        logits = logits + torch.log(incoming.clamp_min(1e-8))[:, None]
        logits = logits.masked_fill(~mask[:, None], -1e9)
        attention = self.dropout(torch.softmax(logits, dim=-1))
        aggregate = torch.einsum("bhts,bhsd->bhtd", attention, value)
        aggregate = aggregate.transpose(1, 2).contiguous().view(batch, services, hidden)
        return self.output(aggregate)


class EadroAdapter(nn.Module):
    def __init__(self, widths: dict[str, int], hidden: int, heads: int, dropout: float):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout))
            for name, width in widths.items()
        })
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.graph = DenseGraphAttention(hidden, heads, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.pool = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, inputs: dict[str, torch.Tensor], adjacency: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat([self.encoders[name](inputs[name]) for name in ("metrics", "logs", "traces")], dim=-1))
        nodes = self.norm(fused + self.graph(fused, adjacency))
        weights = torch.softmax(self.pool(nodes).squeeze(-1), dim=-1)
        context = torch.sum(weights[..., None] * nodes, dim=1, keepdim=True).expand_as(nodes)
        return self.classifier(torch.cat([nodes, context], dim=-1)).squeeze(-1)


class MulanAdapter(nn.Module):
    def __init__(self, widths: dict[str, int], hidden: int, dropout: float = 0.2):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            for name, width in widths.items()
        })
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = {name: F.normalize(self.encoders[name](inputs[name]), dim=-1) for name in ("metrics", "logs", "traces")}
        concatenated = torch.cat([encoded[name] for name in ("metrics", "logs", "traces")], dim=-1)
        weights = torch.softmax(self.gate(concatenated), dim=-1)
        stack = torch.stack([encoded[name] for name in ("metrics", "logs", "traces")], dim=-2)
        fused = torch.sum(weights[..., None] * stack, dim=-2)
        return self.classifier(fused).squeeze(-1), encoded


def contrastive_loss(anchor: torch.Tensor, other: torch.Tensor, temperature: float) -> torch.Tensor:
    anchor = anchor.reshape(-1, anchor.shape[-1])
    other = other.reshape(-1, other.shape[-1])
    logits = anchor @ other.T / temperature
    targets = torch.arange(len(anchor), device=anchor.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def _tensor_inputs(descriptors: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(value[indices], dtype=torch.float32) for name, value in descriptors.items()}


def train_supervised(
    bundle: DatasetBundle,
    method: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    set_seed(seed)
    descriptors = standardized_descriptors(bundle)
    widths = {name: value.shape[-1] for name, value in descriptors.items()}
    adjacency = adjacency_tensor(bundle)
    hidden = int(config["hidden_dim"])
    if method == "Eadro":
        model: nn.Module = EadroAdapter(widths, hidden, int(config["attention_heads"]), float(config["dropout"]))
    elif method == "MULAN":
        model = MulanAdapter(widths, hidden)
    else:
        raise ValueError(method)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    counts = np.bincount(bundle.labels[bundle.fit], minlength=bundle.service_count)
    class_weight = counts.sum() / (bundle.service_count * np.maximum(counts, 1))
    class_weight = torch.as_tensor(class_weight, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    batch_size = int(config["batch_size"])
    for _ in range(int(config["max_epochs"])):
        model.train()
        order = rng.permutation(bundle.fit)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            inputs = _tensor_inputs(descriptors, indices)
            labels = torch.as_tensor(bundle.labels[indices], dtype=torch.long)
            if method == "Eadro":
                logits = model(inputs, torch.as_tensor(adjacency[indices], dtype=torch.float32))
                loss = F.cross_entropy(logits, labels, weight=class_weight)
            else:
                logits, encoded = model(inputs)
                classification = F.cross_entropy(logits, labels, weight=class_weight)
                alignment = 0.5 * (
                    contrastive_loss(encoded["metrics"], encoded["logs"], float(config["temperature"]))
                    + contrastive_loss(encoded["metrics"], encoded["traces"], float(config["temperature"]))
                )
                loss = classification + 0.1 * alignment
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_inputs = _tensor_inputs(descriptors, bundle.validation)
            validation_labels = torch.as_tensor(bundle.labels[bundle.validation], dtype=torch.long)
            if method == "Eadro":
                validation_logits = model(validation_inputs, torch.as_tensor(adjacency[bundle.validation], dtype=torch.float32))
            else:
                validation_logits, _ = model(validation_inputs)
            validation_loss = float(F.cross_entropy(validation_logits, validation_labels).item())
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    all_indices = np.arange(len(bundle.rows))
    with torch.no_grad():
        all_inputs = _tensor_inputs(descriptors, all_indices)
        if method == "Eadro":
            logits = model(all_inputs, torch.as_tensor(adjacency, dtype=torch.float32))
        else:
            logits, _ = model(all_inputs)
        direct = torch.softmax(logits, dim=-1).numpy()
    if method == "MULAN":
        propagated = np.zeros_like(direct)
        for index in range(len(bundle.rows)):
            propagated[index] = random_walk(adjacency[index], direct[index], damping=0.85, reverse=True)
        mix = float(config["graph_mix"])
        direct = (1 - mix) * row_unit_interval(direct) + mix * row_unit_interval(propagated)
    return direct, best_state


def score_once(
    bundle: DatasetBundle, method: str, config: dict[str, Any], seed: int
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    if method == "TraceAnomaly":
        return traceanomaly_fit_score(bundle, config, seed)
    return train_supervised(bundle, method, config, seed)


def save_checkpoint(bundle: DatasetBundle, method: str, seed: int, state: dict[str, torch.Tensor]) -> tuple[str, str]:
    path = OUTPUT / bundle.name / method / f"seed-{seed}" / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return str(path.relative_to(ROOT)), sha256(path)


def run_method(bundle: DatasetBundle, method: str, entry: dict[str, Any]) -> dict[str, Any]:
    fallback = fallback_scores(bundle)
    candidates = []
    selection_failures = []
    for config in configurations(entry["tuning"]):
        try:
            raw, _ = score_once(bundle, method, config, 2027)
            completed, invalid = complete_or_fallback(raw, fallback)
            selection_failures.append({"configuration": config, "invalid_incidents": int(invalid.sum()), "exception": None})
        except Exception as error:  # frozen failure policy
            completed = fallback.copy()
            selection_failures.append({"configuration": config, "invalid_incidents": len(bundle.rows), "exception": type(error).__name__})
        candidates.append((config, completed))
    selected, _, selection = choose_configuration(candidates, bundle)
    run_results = []
    test_matrices = []
    for seed in RESULT_SEEDS:
        failure = None
        checkpoint_path = None
        checkpoint_hash = None
        try:
            raw, state = score_once(bundle, method, selected, seed)
            scores, invalid = complete_or_fallback(raw, fallback)
            checkpoint_path, checkpoint_hash = save_checkpoint(bundle, method, seed, state)
        except Exception as error:
            scores = fallback.copy()
            invalid = np.ones(len(bundle.rows), dtype=bool)
            failure = f"{type(error).__name__}: {error}"
        provenance = {
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "implementation_class": entry["implementation_class"],
            "material_adapter": entry["material_adapter"],
            "adaptation": entry["adaptation"],
            "modalities": entry["modalities"],
            "selected_configuration": selected,
            "selection_seed": 2027,
            "result_seed": seed,
            "invalid_incidents": int(invalid.sum()),
            "failure": failure,
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": checkpoint_hash,
            "code": str(Path(__file__).resolve().relative_to(ROOT)),
            "code_sha256": sha256(Path(__file__)),
        }
        result = save_method_run(
            bundle, method, scores, provenance,
            selection=selection, run_name=f"seed-{seed}",
        )
        run_results.append(result)
        test_matrices.append(scores[bundle.test])
    summary = {
        "dataset": bundle.name,
        "method": method,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "selected_configuration": selected,
        "selection_failures": selection_failures,
        "complete_seed_count": sum(result["provenance"]["failure"] is None for result in run_results),
        "required_seed_count": 5,
        "metrics": summarize_repeats(test_matrices, bundle.labels[bundle.test]),
        "runs": [
            {"run_name": result["run_name"], "metrics": result["metrics"], "scores_sha256": result["scores_sha256"]}
            for result in run_results
        ],
    }
    atomic_json(OUTPUT / bundle.name / method / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["trainticket", "aiops2022"], default=["trainticket", "aiops2022"])
    parser.add_argument("--methods", nargs="+", choices=["TraceAnomaly", "Eadro", "MULAN"], default=["TraceAnomaly", "Eadro", "MULAN"])
    args = parser.parse_args()
    protocol = verify_protocol()
    summaries = []
    for dataset_name in args.datasets:
        bundle = load_dataset(dataset_name)
        for method in args.methods:
            summaries.append(run_method(bundle, method, protocol["methods"][method]))
    atomic_json(OUTPUT / "neural_run_summary.json", summaries)
    compact = [
        {"dataset": item["dataset"], "method": item["method"], "complete_seed_count": item["complete_seed_count"], "metrics": {name: value for name, value in item["metrics"].items() if name != "runs"}}
        for item in summaries
    ]
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
