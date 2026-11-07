"""Train CAREBase with modality dropout and optional history-type holdout.

This script is isolated from the frozen reproduction trainer so reviewer-response
experiments cannot invalidate or overwrite the earlier protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_base import CAREBase, ranking_metrics
from robust_evidence import fault_type_holdout


MODALITIES = ("metrics", "traces", "logs")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def modality_mask(
    batch: int,
    rate: float,
    rng: np.random.Generator,
    services: int = 1,
) -> np.ndarray:
    available = rng.random((batch, services, len(MODALITIES))) >= rate
    none = ~available.any(-1)
    if none.any():
        incidents, nodes = np.where(none)
        restore = rng.integers(0, len(MODALITIES), size=int(none.sum()))
        available[incidents, nodes, restore] = True
    return available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--contrast-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-selection", choices=["validation", "final"], default="validation")
    parser.add_argument("--metric-patch-size", type=int, default=0)
    parser.add_argument("--metric-patch-stride", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--modality-dropout", type=float, default=0.5)
    parser.add_argument("--dropout-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--ablation",
        choices=[
            "full", "no_cl", "standard_cl", "no_gat",
            "no_metrics", "no_traces", "no_logs",
        ],
        default="full",
    )
    parser.add_argument("--exclude-fault-type")
    args = parser.parse_args()
    if not 0 <= args.modality_dropout < 1:
        raise ValueError("modality-dropout must be in [0, 1)")
    if not 0 <= args.dropout_loss_weight <= 1:
        raise ValueError("dropout-loss-weight must be in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rows = json.loads((args.data / "incidents.json").read_text())
    metadata = json.loads((args.data / "metadata.json").read_text())
    features = np.load(args.data / "features.npz")
    device = torch.device(args.device)
    tensors = [torch.from_numpy(features[name]).float().to(device) for name in MODALITIES]
    labels = torch.from_numpy(features["labels"]).long().to(device)
    services = metadata["services"]

    split_indices = {
        split: np.asarray(
            [i for i, row in enumerate(rows) if row["split"] == split], dtype=int
        )
        for split in ("train", "validation")
    }
    overlap_purged: dict[str, np.ndarray] = {
        split: np.asarray([], dtype=int) for split in split_indices
    }
    held_overlap_groups: list[object] = []
    if args.exclude_fault_type is None:
        indices = split_indices
    else:
        indices = {}
        for split, candidates in split_indices.items():
            fit, _, purged, groups = fault_type_holdout(
                rows,
                candidates,
                args.exclude_fault_type,
                held_scope_indices=np.arange(len(rows)),
            )
            indices[split] = fit
            overlap_purged[split] = purged
            held_overlap_groups = groups
    all_test = np.asarray([i for i, row in enumerate(rows) if row["split"] == "test"], dtype=int)
    target_test = (
        np.asarray(
            [
                i
                for i in all_test
                if rows[i]["fault_type"] == args.exclude_fault_type
            ],
            dtype=int,
        )
        if args.exclude_fault_type is not None
        else all_test
    )
    if not len(indices["train"]) or not len(target_test):
        raise ValueError("Training and target-test sets must be nonempty")
    if args.checkpoint_selection == "validation" and not len(indices["validation"]):
        raise ValueError("Validation selection requires non-held-out validation incidents")

    adjacency = np.eye(len(services), dtype=bool)
    for index in indices["train"]:
        for edge in rows[index]["calls"]:
            source, target = services.index(edge["source"]), services.index(edge["target"])
            adjacency[source, target] = adjacency[target, source] = True
    adjacency_tensor = torch.tensor(adjacency, device=device)
    model_args = {
        "metric_dim": tensors[0].shape[-1],
        "trace_dim": tensors[1].shape[-1],
        "log_dim": tensors[2].shape[-1],
        "metric_patch_size": args.metric_patch_size,
        "metric_patch_stride": args.metric_patch_stride,
        "ablation": args.ablation,
    }
    model = CAREBase(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def predict(which: np.ndarray, missing: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        probabilities, attentions = [], []
        with torch.no_grad():
            for chunk in np.array_split(which, max(1, int(np.ceil(len(which) / args.batch_size)))):
                inputs = [tensor[chunk] for tensor in tensors]
                if missing is not None:
                    position = MODALITIES.index(missing)
                    inputs[position] = torch.zeros_like(inputs[position])
                logits, _, attention = model(*inputs, adjacency_tensor)
                probabilities.append(logits.softmax(-1).cpu().numpy())
                attentions.append(attention.cpu().numpy())
        return np.concatenate(probabilities), np.concatenate(attentions)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best = (-1.0, -1.0)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(indices["train"])
        losses = []
        for chunk in np.array_split(order, max(1, int(np.ceil(len(order) / args.batch_size)))):
            inputs = [tensor[chunk] for tensor in tensors]
            logits, contrast, _ = model(*inputs, adjacency_tensor)
            full_loss = F.cross_entropy(logits, labels[chunk])
            if args.dropout_loss_weight:
                available = modality_mask(
                    len(chunk), args.modality_dropout, rng, inputs[0].shape[1]
                )
                masked_inputs = []
                for position, values in enumerate(inputs):
                    shape = (len(chunk), values.shape[1]) + (1,) * (values.ndim - 2)
                    mask = torch.as_tensor(
                        available[:, :, position], device=device
                    ).view(shape)
                    masked_inputs.append(values * mask)
                masked_logits, _, _ = model(*masked_inputs, adjacency_tensor)
                masked_loss = F.cross_entropy(masked_logits, labels[chunk])
            else:
                masked_loss = full_loss.new_zeros(())
            loss = (
                (1 - args.dropout_loss_weight) * full_loss
                + args.dropout_loss_weight * masked_loss
                + args.contrast_weight * contrast
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        if args.checkpoint_selection == "validation":
            probabilities, _ = predict(indices["validation"])
            scores = ranking_metrics(probabilities, features["labels"][indices["validation"]])
            record["validation"] = scores
            key = (scores["PR@1"], scores["MAP@5"])
            save = key > best
            if save:
                best = key
        else:
            save = epoch + 1 == args.epochs
        history.append(record)
        print(json.dumps(record), flush=True)
        if save:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_args": model_args,
                    "adjacency": adjacency,
                    "epoch": epoch + 1,
                    "training_ids": [rows[i]["incident_id"] for i in indices["train"]],
                    "excluded_fault_type": args.exclude_fault_type,
                    "overlap_purged_ids": [
                        rows[i]["incident_id"] for i in overlap_purged["train"]
                    ],
                    "held_overlap_groups": held_overlap_groups,
                },
                args.output / "model.pt",
            )
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(args.output / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    all_indices = np.arange(len(rows))
    prediction_sets: dict[str, np.ndarray] = {}
    attention = None
    for missing in (None, *MODALITIES):
        key = "full" if missing is None else f"no_{missing}"
        prediction_sets[key], current_attention = predict(all_indices, missing)
        if attention is None:
            attention = current_attention
    np.savez_compressed(args.output / "predictions.npz", attention=attention, **prediction_sets)

    target_labels = features["labels"][target_test]
    metrics = {
        key: ranking_metrics(values[target_test], target_labels)
        for key, values in prediction_sets.items()
    }
    result = {
        "selected_epoch": checkpoint["epoch"],
        "excluded_fault_type": args.exclude_fault_type,
        "counts": {
            "train": len(indices["train"]),
            "validation": len(indices["validation"]),
            "all_test": len(all_test),
            "target_test": len(target_test),
            "overlap_purged_train": len(overlap_purged["train"]),
            "overlap_purged_validation": len(overlap_purged["validation"]),
        },
        "overlap_purged_incident_ids": {
            split: [rows[i]["incident_id"] for i in values]
            for split, values in overlap_purged.items()
        },
        "held_overlap_groups": held_overlap_groups,
        "target_test_indices": target_test.tolist(),
        "metrics": metrics,
        "training_seconds": training_seconds,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "model_args": model_args,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "incidents_sha256": hashlib.sha256((args.data / "incidents.json").read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256((args.data / "features.npz").read_bytes()).hexdigest(),
        "protocol": (
            "Fault type and any different-type incidents sharing its overlap "
            "groups are removed from train, validation selection, and training "
            "topology; target test contains only the held-out type."
        ),
    }
    atomic_json(args.output / "history.json", history)
    atomic_json(args.output / "evaluation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
