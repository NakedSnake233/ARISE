"""Original context training with one explicitly requested component removed.

The frozen T1 trainer stays unchanged. This entry uses the same seed, loss,
batch ordering and checkpoint rule, and is reusable on a supplied outer cohort.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, config_snapshot, json_dump, sha256
from .fold_data import load_fold
from .infer_evidence_masks import context_scores
from .metrics import ranking_metrics
from .missingness import SCENARIOS
from .train_context import ContextEncoder, _fit_adjacency, _inner_indices, _mask, _ranks
from .train_evidence import fuse_all
from .train_variant_evidence import T6_SCENARIOS

VARIANTS = {
    "no_cl": "ARISE w/o shared-complementary loss",
    "standard_cl": "ARISE with standard contrastive loss",
    "no_gat": "ARISE w/o graph contextualization",
    "no_modality_dropout": "ARISE w/o modality dropout",
}


def fit_context(data, fit, validation, seed, variant, device_name, output):
    cfg = json.loads((ROOT / "configs/care/review-response-v1.json").read_text())
    base_cfg = cfg["base"].copy(); data_cfg = cfg["datasets"]["trainticket" if data.dataset == "tt" else "aiops2022"]
    if variant == "no_modality_dropout":
        base_cfg.update(modality_dropout=0., dropout_loss_weight=0.)
    checkpoint_rule = data_cfg["checkpoint_selection"]
    if checkpoint_rule == "validation" and not len(validation): raise ValueError("empty context validation")
    torch.set_num_threads(int(base_cfg["threads"]))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed); device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    tensors = [torch.as_tensor(getattr(data, name), dtype=torch.float32, device=device) for name in ("metrics", "traces", "logs")]
    labels = torch.as_tensor(data.train_labels, dtype=torch.long, device=device)
    fit_graph = _fit_adjacency(data, fit); adjacency = torch.as_tensor(fit_graph, device=device)
    args = dict(metric_dim=tensors[0].shape[-1], trace_dim=tensors[1].shape[-1], log_dim=tensors[2].shape[-1],
                hidden=64, heads=8, temperature=.07, diversity=.5, metric_patch_size=data_cfg["metric_patch_size"],
                ablation="full" if variant == "no_modality_dropout" else variant)
    model = ContextEncoder(**args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_cfg["lr"])
    batch_size = int(base_cfg["batch_size"])

    def infer(indices, masks=None):
        model.eval(); probabilities = []
        with torch.no_grad():
            for chunk in np.array_split(np.arange(len(indices)), max(1, int(np.ceil(len(indices) / batch_size)))):
                inputs = [a[indices[chunk]] for a in tensors]
                if masks is not None:
                    inputs = [a * torch.as_tensor(masks[chunk, :, m, None, None], device=device)
                              for a, m in zip(inputs, (0, 2, 1))]
                logits, _, _ = model(*inputs, adjacency)
                probabilities.append(logits.softmax(-1).cpu().numpy())
        return np.concatenate(probabilities)

    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    best, selected_epoch, history = (-1., -1.), None, []
    for epoch in range(int(base_cfg["epochs"])):
        model.train(); order = rng.permutation(fit); losses = []
        for positions in np.array_split(order, max(1, int(np.ceil(len(order) / batch_size)))):
            inputs = [a[positions] for a in tensors]
            logits, contrast, _ = model(*inputs, adjacency)
            full_loss = F.cross_entropy(logits, labels[positions])
            weight = base_cfg["dropout_loss_weight"]
            if weight:
                masks = _mask(len(positions), len(data.services), base_cfg["modality_dropout"], rng)
                masked = [a * torch.as_tensor(masks[:, :, m, None, None], device=device) for m, a in enumerate(inputs)]
                masked_logits, _, _ = model(*masked, adjacency)
                masked_loss = F.cross_entropy(masked_logits, labels[positions])
            else: masked_loss = full_loss.new_zeros(())
            loss = (1 - weight) * full_loss + weight * masked_loss + base_cfg["contrast_weight"] * contrast
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            losses.append(float(loss.item()))
        record = dict(epoch=epoch + 1, loss=float(np.mean(losses)))
        if checkpoint_rule == "validation":
            metrics = ranking_metrics(_ranks(infer(validation), data.train_labels[validation])); record["validation"] = metrics
            key = (metrics["PR@1"], metrics["MAP@5"]); save = key > best
            if save: best = key
        else: save = epoch + 1 == base_cfg["epochs"]
        if save:
            selected_epoch = epoch + 1
            torch.save(dict(state_dict=model.state_dict(), model_args=args, adjacency=fit_graph, epoch=selected_epoch,
                fit_ids=[data.train_ids[i] for i in fit], validation_ids=[data.train_ids[i] for i in validation]), output / "model.pt")
        history.append(record)
        if epoch == 0 or (epoch + 1) % 10 == 0: print(json.dumps(record), flush=True)
    checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); json_dump(output / "history.json", history)
    training = dict(selected_epoch=selected_epoch, fit_count=len(fit), validation_count=len(validation),
        model_args=args, base=base_cfg, checkpoint_rule=checkpoint_rule,
        gpu_model=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None)
    return infer, training


def train(dataset, fold, seed, variant, output, smoke=False):
    started = time.perf_counter(); start = dt.datetime.now(dt.timezone.utc).isoformat()
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable context variant output exists")
    output.mkdir(parents=True, exist_ok=True)
    data = load_fold(dataset, fold); fit, validation = _inner_indices(dataset, fold, data.train_ids)
    # CPU sanity keeps the complete fifty-epoch smoke within ten minutes.
    # Formal TT variants always use the same local A100 backend as T1.
    device = "cuda" if dataset == "tt" and not smoke else "cpu"
    infer, training = fit_context(data, fit, validation, seed, variant, device, output)
    indices = data.test_indices[:5] if smoke else data.test_indices; ids = [data.test_ids[i] for i in range(len(indices))]
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    row = state["jobs"][f"full_evidence_masks_{dataset}_fold{fold}_seed{seed}"]
    if row["status"] != "completed": raise ValueError("full evidence score prerequisite incomplete")
    evidence_folder = ROOT / row["output"]
    with np.load(evidence_folder / "scores.npz") as archive:
        if archive["incident_ids"].tolist()[:len(ids)] != ids: raise ValueError("evidence identity mismatch")
        evidence = archive["evidence"][:, :len(ids)].copy()
    all_scenarios = [f"{s}_{p or 'none'}" for s, p in SCENARIOS]
    scores, context, hashes = [], [], {}
    for scenario in T6_SCENARIOS:
        path = ROOT / f"masks/{dataset}/{scenario}/seed{seed}.npz"
        with np.load(path) as archive: masks = np.stack([archive[key] for key in ids])
        values = infer(indices, masks); context.append(values)
        scores.append(fuse_all((.25, values), (.75, evidence[all_scenarios.index(scenario)])))
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    np.savez_compressed(output / "scores.npz", scores=np.stack(scores), context=np.stack(context),
                        incident_ids=ids, scenario_keys=T6_SCENARIOS)
    config = dict(task="T6", method=VARIANTS[variant], variant=variant, dataset=dataset, fold=fold, seed=seed,
                  smoke=smoke, device=device, scenarios=T6_SCENARIOS, alpha=.75, split="kfold5", preregistered=False)
    meta = config_snapshot("python -m exp_revision.train_context_variant", config)
    sources = [ROOT / name for name in ("exp_revision/train_context_variant.py", "exp_revision/train_context.py",
        "scripts/care/train_base.py", "scripts/care/train_robust_base.py", "exp_revision/fold_data.py", "exp_revision/train_evidence.py",
        "exp_revision/train_variant_evidence.py", "configs/care/review-response-v1.json",
        f"splits/{dataset}_training_labels/fold_{fold}.json", f"cache/fold_preprocessing/{dataset}/fold{fold}/features.npz",
        f"cache/fold_preprocessing/{dataset}/fold{fold}/metadata.json")]
    if dataset == "tt": sources.append(ROOT / "splits/tt_inner_validation_kfold5.json")
    sources += [evidence_folder / name for name in ("scores.npz", "run_meta.json")]
    meta.update(status="completed_context_variant", start_time=start, end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started, training=training, mask_sha256=hashes,
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources}, model_sha256=sha256(output / "model.pt"),
        scores_sha256=sha256(output / "scores.npz"))
    json_dump(output / "run_meta.json", meta)
    return output


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", choices=list(VARIANTS), required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true"); a = p.parse_args()
    print(train(a.dataset, a.fold, a.seed, a.variant, a.output, a.smoke))


if __name__ == "__main__": main()
