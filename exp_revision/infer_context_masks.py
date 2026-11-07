"""Infer context-only missingness scenarios using one frozen T1 checkpoint.

No held-out root or fault-type labels are opened. Each test modality is zeroed
according to the shared mask; the training-only graph and normalization stay
fixed. This is a partial T2 baseline, not full ARISE robustness.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, config_snapshot, json_dump, sha256
from .fold_data import load_fold
from .missingness import SCENARIOS

sys.path.insert(0, str(ROOT / "scripts/care"))
from train_base import CAREBase as ContextEncoder  # noqa: E402


def infer(dataset: str, fold: int, seed: int, *, smoke: bool = False,
          model_root: Path = ROOT / "results_v2/T1_context_full",
          output: Path | None = None, device_name: str = "cpu") -> Path:
    started = time.perf_counter()
    start_time = dt.datetime.now(dt.timezone.utc).isoformat()
    data = load_fold(dataset, fold, smoke=False)
    job_id = f"{dataset}_fold{fold}_seed{seed}"
    state = json.loads((model_root / dataset / "queue_state.json").read_text())
    if state["jobs"][job_id]["status"] != "completed":
        raise ValueError("context model is incomplete")
    model_folder = ROOT / state["jobs"][job_id]["output"]
    train_meta = json.loads((model_folder / "run_meta.json").read_text())
    if (train_meta["status"] != "partial_T1_context_only"
            or train_meta["config"]["dataset"] != dataset
            or train_meta["config"]["fold"] != fold
            or train_meta["config"]["seed"] != seed
            or train_meta["model_sha256"] != sha256(model_folder / "model.pt")
            or train_meta["scores_sha256"] != sha256(model_folder / "scores.npz")):
        raise ValueError("context checkpoint provenance differs from job")
    for relative, digest in train_meta["input_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"context training input changed: {relative}")
    if output is None:
        stage = "T2_context_smoke" if smoke else "T2_context_full"
        output = ROOT / "results_v2" / stage / dataset / job_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite context missingness output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    checkpoint = torch.load(model_folder / "model.pt", map_location=device,
                            weights_only=False)
    model = ContextEncoder(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    adjacency = torch.as_tensor(checkpoint["adjacency"], device=device)
    test_count = 5 if smoke else len(data.test_ids)
    ids = data.test_ids[:test_count]
    test_start = len(data.train_ids)
    arrays = [getattr(data, name)[test_start:test_start + test_count]
              for name in ("metrics", "traces", "logs")]
    raw_availability = data.availability[test_start:test_start + test_count]
    scores = []
    mask_hashes = {}
    scenario_names = []
    for scenario, param in SCENARIOS:
        mask_path = ROOT / "masks" / dataset / f"{scenario}_{param or 'none'}" / f"seed{seed}.npz"
        with np.load(mask_path) as archive:
            if not set(ids) <= set(archive.files):
                raise ValueError("shared mask is missing held-out incident")
            masks = np.stack([archive[incident_id] for incident_id in ids]).astype(np.uint8)
        if (masks.shape != raw_availability.shape
                or not np.isin(masks, (0, 1)).all()):
            raise ValueError("invalid context missingness mask")
        # Fold tensors are already fitted on outer training data. Removing a
        # whole test modality is exactly a zero slice in this context model.
        tensors = [torch.as_tensor(value * masks[:, :, modality, None, None],
                                   dtype=torch.float32, device=device)
                   for value, modality in zip(arrays, (0, 2, 1))]
        chunks = []
        with torch.no_grad():
            for indices in np.array_split(np.arange(test_count),
                                          max(1, int(np.ceil(test_count / 16)))):
                logits, _, _ = model(*(array[indices] for array in tensors), adjacency)
                chunks.append(logits.softmax(-1).cpu().numpy())
        probability = np.concatenate(chunks)
        if (probability.shape != (test_count, len(data.services))
                or not np.isfinite(probability).all()
                or not np.allclose(probability.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("invalid context masked scores")
        scores.append(probability)
        scenario_names.append(f"{scenario}_{param or 'none'}")
        mask_hashes[str(mask_path.relative_to(ROOT))] = sha256(mask_path)
    with np.load(model_folder / "scores.npz") as archive:
        baseline = archive["probabilities"][test_start:test_start + test_count]
        if archive["incident_ids"].tolist() != data.train_ids + data.test_ids:
            raise ValueError("T1 context score incident order differs from fold")
    s0_diff = float(np.max(np.abs(scores[0] - baseline)))
    if s0_diff > 1e-5:
        raise ValueError(f"S0 masked inference does not reproduce T1 context scores: {s0_diff}")
    np.savez_compressed(output / "scores.npz", incident_ids=np.asarray(ids),
                        scenario_keys=np.asarray(scenario_names),
                        probabilities=np.stack(scores))
    meta = config_snapshot(
        f"python -m exp_revision.infer_context_masks --dataset {dataset} --fold {fold} --seed {seed} --device {device_name}",
        {"task": "T2", "method": "context-only", "dataset": dataset,
         "fold": fold, "seed": seed, "smoke": smoke,
         "device": device_name, "scenarios": scenario_names,
         "normalization": "frozen outer-training-fold", "adjacency": "frozen checkpoint"})
    meta.update(status="partial_T2_context_only", start_time=start_time,
                end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                elapsed_seconds=time.perf_counter() - started,
                baseline_s0_max_abs_diff=s0_diff,
                train_model_sha256=sha256(model_folder / "model.pt"),
                train_scores_sha256=sha256(model_folder / "scores.npz"),
                mask_sha256=mask_hashes,
                source_sha256={name: sha256(ROOT / name) for name in (
                    "exp_revision/infer_context_masks.py", "exp_revision/missingness.py",
                    "exp_revision/fold_data.py", "scripts/care/train_base.py")},
                scores_sha256=sha256(output / "scores.npz"))
    json_dump(output / "run_meta.json", meta)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(infer(args.dataset, args.fold, args.seed, smoke=args.smoke,
                device_name=args.device, output=args.output))


if __name__ == "__main__":
    main()
