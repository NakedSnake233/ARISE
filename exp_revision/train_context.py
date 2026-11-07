"""Train the original context branch on one leakage-isolated outer fold.

The resulting scores belong to the context-only baseline; they do not include
the ARISE evidence experts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, config_snapshot, json_dump, sha256
from .fold_data import load_fold
from .metrics import ranking_metrics

sys.path.insert(0, str(ROOT / "scripts/care"))
from train_base import CAREBase as ContextEncoder  # noqa: E402


def _mask(batch: int, services: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    available = rng.random((batch, services, 3)) >= rate
    empty = ~available.any(-1)
    if empty.any():
        cases, nodes = np.where(empty)
        restore = rng.integers(0, 3, size=int(empty.sum()))
        available[cases, nodes, restore] = True
    return available


def _inner_indices(dataset: str, fold: int, train_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "aiops":
        return np.arange(len(train_ids)), np.asarray([], dtype=int)
    frozen = json.loads(Path("splits/tt_inner_validation_kfold5.json").read_text())[f"fold_{fold}"]
    mapping = {incident_id: index for index, incident_id in enumerate(train_ids)}
    if set(frozen["fit"]) | set(frozen["validation"]) != set(train_ids):
        raise ValueError("inner validation does not partition the outer training fold")
    if set(frozen["fit"]) & set(frozen["validation"]):
        raise ValueError("inner validation overlaps training fit")
    return (np.asarray([mapping[x] for x in frozen["fit"]], dtype=int),
            np.asarray([mapping[x] for x in frozen["validation"]], dtype=int))


def _ranks(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(-probabilities, axis=1, kind="stable")
    return np.asarray([int(np.flatnonzero(order[i] == label)[0] + 1) for i, label in enumerate(labels)])


def _fit_adjacency(fold_data, fit: np.ndarray) -> np.ndarray:
    """Fit topology on model-fit events, excluding checkpoint validation."""
    if "calls" not in fold_data.metadata:
        if len(fit) != len(fold_data.train_ids):
            raise ValueError("inner holdout requires per-incident training calls")
        return fold_data.adjacency
    service_index = {name: i for i, name in enumerate(fold_data.services)}
    adjacency = np.eye(len(service_index), dtype=bool)
    for i in fit:
        for call in fold_data.metadata["calls"][fold_data.train_ids[int(i)]]:
            u, v = service_index[call["source"]], service_index[call["target"]]
            adjacency[u, v] = adjacency[v, u] = True
    return adjacency


def train(dataset: str, fold: int, seed: int, *, smoke: bool = False,
          device_name: str = "cuda", output: Path | None = None) -> Path:
    started = time.perf_counter()
    start_time = dt.datetime.now(dt.timezone.utc).isoformat()
    cfg = json.loads((ROOT / "configs/care/review-response-v1.json").read_text())
    dataset_key = {"tt": "trainticket", "aiops": "aiops2022"}[dataset]
    base_cfg = cfg["base"]
    data_cfg = cfg["datasets"][dataset_key]
    fold_data = load_fold(dataset, fold, smoke=smoke)
    fit, validation = _inner_indices(dataset, fold, fold_data.train_ids)
    checkpoint_rule = data_cfg["checkpoint_selection"]
    if dataset == "tt" and checkpoint_rule != "validation":
        raise ValueError("TrainTicket must select a validation checkpoint")
    if dataset == "aiops" and checkpoint_rule != "final":
        raise ValueError("AIOps must use the final epoch")
    if checkpoint_rule == "validation" and not len(validation):
        raise ValueError("validation split is empty")
    if output is None:
        suffix = "T1_context_smoke" if smoke else "T1_context"
        output = ROOT / "results_v2" / suffix / dataset / f"fold{fold}_seed{seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(base_cfg["threads"]))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    tensors = [torch.as_tensor(getattr(fold_data, name), dtype=torch.float32, device=device)
               for name in ("metrics", "traces", "logs")]
    labels = torch.as_tensor(fold_data.train_labels, dtype=torch.long, device=device)
    fit_adjacency = _fit_adjacency(fold_data, fit)
    adjacency = torch.as_tensor(fit_adjacency, device=device)
    model_args = {
        "metric_dim": tensors[0].shape[-1], "trace_dim": tensors[1].shape[-1],
        "log_dim": tensors[2].shape[-1], "hidden": 64, "heads": 8,
        "temperature": 0.07, "diversity": 0.5,
        "metric_patch_size": data_cfg["metric_patch_size"], "ablation": "full",
    }
    model = ContextEncoder(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(base_cfg["lr"]))
    batch_size = int(base_cfg["batch_size"])
    epochs = int(base_cfg["epochs"])

    def predict(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        probabilities, attentions = [], []
        with torch.no_grad():
            for chunk in np.array_split(indices, max(1, int(np.ceil(len(indices) / batch_size)))):
                positions = torch.as_tensor(chunk, dtype=torch.long, device=device)
                logits, _, attention = model(*(array[positions] for array in tensors), adjacency)
                probabilities.append(logits.softmax(-1).cpu().numpy())
                attentions.append(attention.cpu().numpy())
        return np.concatenate(probabilities), np.concatenate(attentions)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best = (-1.0, -1.0)
    selected_epoch = None
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(fit)
        losses = []
        for chunk in np.array_split(order, max(1, int(np.ceil(len(order) / batch_size)))):
            positions = torch.as_tensor(chunk, dtype=torch.long, device=device)
            inputs = [array[positions] for array in tensors]
            logits, contrast, _ = model(*inputs, adjacency)
            full_loss = F.cross_entropy(logits, labels[positions])
            mask = _mask(len(chunk), len(fold_data.services), float(base_cfg["modality_dropout"]), rng)
            masked_inputs = []
            for modality, values in enumerate(inputs):
                shape = (len(chunk), len(fold_data.services)) + (1,) * (values.ndim - 2)
                visible = torch.as_tensor(mask[:, :, modality], device=device).view(shape)
                masked_inputs.append(values * visible)
            masked_logits, _, _ = model(*masked_inputs, adjacency)
            masked_loss = F.cross_entropy(masked_logits, labels[positions])
            weight = float(base_cfg["dropout_loss_weight"])
            loss = (1 - weight) * full_loss + weight * masked_loss + float(base_cfg["contrast_weight"]) * contrast
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        if checkpoint_rule == "validation":
            probabilities, _ = predict(validation)
            metrics = ranking_metrics(_ranks(probabilities, fold_data.train_labels[validation]))
            record["validation"] = metrics
            key = (metrics["PR@1"], metrics["MAP@5"])
            save = key > best
            if save:
                best = key
        else:
            save = epoch + 1 == epochs
        if save:
            selected_epoch = epoch + 1
            torch.save({"state_dict": model.state_dict(), "model_args": model_args,
                        "adjacency": fit_adjacency, "epoch": selected_epoch,
                        "fit_ids": [fold_data.train_ids[i] for i in fit],
                        "validation_ids": [fold_data.train_ids[i] for i in validation]}, output / "model.pt")
        history.append(record)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(json.dumps(record), flush=True)
    checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    all_indices = np.arange(len(fold_data.train_ids) + len(fold_data.test_ids))
    probabilities, attention = predict(all_indices)
    np.savez_compressed(output / "scores.npz", probabilities=probabilities, attention=attention,
                        incident_ids=np.asarray(fold_data.train_ids + fold_data.test_ids))
    json_dump(output / "history.json", history)
    run_meta = config_snapshot(
        f"python -m exp_revision.train_context --dataset {dataset} --fold {fold} --seed {seed}{' --smoke' if smoke else ''} --device {device_name}",
        {"method": "context-only", "dataset": dataset, "fold": fold, "seed": seed,
         "smoke": smoke, "base": base_cfg, "dataset_config": data_cfg,
         "checkpoint_rule": checkpoint_rule, "inner_validation_seed": 2027 if dataset == "tt" else None},
    )
    run_meta.update(start_time=start_time, end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                    selected_epoch=selected_epoch, fit_count=len(fit), validation_count=len(validation),
                    test_count=len(fold_data.test_ids), elapsed_seconds=time.perf_counter() - started,
                    gpu_model=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                    status="partial_T1_context_only")
    source_paths = [
        ROOT / "exp_revision/train_context.py", ROOT / "exp_revision/fold_data.py",
        ROOT / "scripts/care/train_base.py", ROOT / "configs/care/review-response-v1.json",
        ROOT / f"splits/{dataset}_kfold5.json",
        ROOT / f"splits/{dataset}_training_labels/fold_{fold}.json",
        ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}{'_smoke' if smoke else ''}/features.npz",
        ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}{'_smoke' if smoke else ''}/metadata.json",
    ]
    if dataset == "tt":
        source_paths.append(ROOT / "splits/tt_inner_validation_kfold5.json")
    run_meta["input_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    run_meta["scores_sha256"] = sha256(output / "scores.npz")
    run_meta["model_sha256"] = sha256(output / "model.pt")
    json_dump(output / "run_meta.json", run_meta)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(train(args.dataset, args.fold, args.seed, smoke=args.smoke, device_name=args.device, output=args.output))


if __name__ == "__main__":
    main()
