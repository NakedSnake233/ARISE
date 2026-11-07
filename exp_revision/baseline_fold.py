"""Frozen real baseline adapters on group-isolated folds and shared masks."""

from __future__ import annotations

import argparse
import copy
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
from torch.nn import functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import ROOT, config_snapshot, json_dump, sha256
from .fold_data import load_fold
from .missingness import SCENARIOS

sys.path.insert(0, str(ROOT / "scripts/care"))
import fse2027_classical_baselines as classical
from fse2027_baseline_common import (DatasetBundle, modality_views, random_walk,
                                    row_unit_interval, training_adjacency)
from fse2027_neural_baselines import EadroAdapter, MulanAdapter, TraceVAE, contrastive_loss

METHODS = ("Eadro", "MULAN", "Eadro+MD", "MULAN+MD", "Nezha", "MicroRCA",
           "MicroScope", "MicroRank", "SBLD", "LogFaultFlagger", "TraceAnomaly")
CONFIG = ROOT / "exp_revision/baseline_settings.json"
MODALITIES = {"metrics": 0, "logs": 1, "traces": 2}


def freeze_settings() -> dict:
    result = {}
    for dataset, legacy in (("tt", "trainticket"), ("aiops", "aiops2022")):
        result[dataset] = {}
        for method in METHODS:
            base = method.removesuffix("+MD")
            repeat = "seed-42" if base in ("Eadro", "MULAN", "TraceAnomaly") else "run-0"
            source = ROOT / f"artifacts/care/fse2027-baselines/{legacy}/{base}/{repeat}/result.json"
            entry = json.loads(source.read_text())["provenance"]["selected_configuration"]
            result[dataset][method] = {"configuration": entry,
                                      "source": str(source.relative_to(ROOT)),
                                      "source_sha256": sha256(source)}
    if CONFIG.exists():
        if json.loads(CONFIG.read_text()) != result:
            raise ValueError("frozen actual baseline settings changed")
    else:
        json_dump(CONFIG, result)
    return result


def inner_indices(data) -> tuple[np.ndarray, np.ndarray]:
    if data.dataset == "tt":
        frozen = json.loads((ROOT / "splits/tt_inner_validation_kfold5.json").read_text())
        part = frozen[f"fold_{data.fold}"]
        mapping = {key: i for i, key in enumerate(data.train_ids)}
        fit, validation = [np.asarray([mapping[k] for k in part[name]])
                           for name in ("fit", "validation")]
    else:
        fit, validation = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=2027)
                               .split(data.train_indices, groups=data.train_groups))
    groups = np.asarray(data.train_groups)
    if (set(groups[fit]) & set(groups[validation]) or not len(validation)
            or set(fit) | set(validation) != set(data.train_indices)):
        raise ValueError("baseline inner validation is not group-isolated")
    return fit, validation


def make_bundle(data, fit: np.ndarray, validation: np.ndarray) -> DatasetBundle:
    ids = data.train_ids + data.test_ids
    # Only train labels are available; unused test placeholders are never used
    # for fitting, checkpoint selection, class weights or feature extraction.
    labels = np.concatenate([data.train_labels, np.full(len(data.test_ids), -1)])
    rows = [{"incident_id": key, "calls": data.metadata["calls"][key]} for key in ids]
    return DatasetBundle(data.dataset, ROOT / f"cache/fold_preprocessing/{data.dataset}/fold{data.fold}",
                         rows, data.metadata, {name: getattr(data, name) for name in MODALITIES},
                         data.services, labels, data.train_indices, validation, fit,
                         data.test_indices, int(data.metadata["before"] // data.metadata["step"]))


def dropout_mask(a0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    keep = rng.random(a0.shape) >= .5
    empty = (a0.astype(bool) & keep).sum(-1) == 0
    for case, service in zip(*np.where(empty & a0.any(-1))):
        keep[case, service, rng.choice(np.flatnonzero(a0[case, service]))] = True
    return keep


def descriptors(features: dict, split: int, scaler: dict | None = None,
                fit: np.ndarray | None = None) -> tuple[dict, dict]:
    raw = {name: modality_views(value, split)[2] for name, value in features.items()}
    if scaler is None:
        if fit is None:
            raise ValueError("descriptor scaler requires training fit indices")
        scaler = {}
        for name, value in raw.items():
            train = value[fit].reshape(-1, value.shape[-1])
            scale = train.std(0)
            scaler[name] = (train.mean(0), np.where(scale > 1e-6, scale, 1.0))
    return {name: np.nan_to_num((value - scaler[name][0]) / scaler[name][1]).astype(np.float32)
            for name, value in raw.items()}, scaler


def fit_neural(data, bundle, method: str, config: dict, seed: int, device: str):
    base = method.removesuffix("+MD")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    prepared, scaler = descriptors(bundle.features, bundle.split_bin, fit=bundle.fit)
    widths = {name: value.shape[-1] for name, value in prepared.items()}
    if base == "Eadro":
        model = EadroAdapter(widths, config["hidden_dim"], config["attention_heads"], config["dropout"])
    else:
        model = MulanAdapter(widths, config["hidden_dim"])
    model.to(device)
    adjacency = training_adjacency(bundle).astype(np.float32)
    graph = torch.as_tensor(adjacency, device=device)
    counts = np.bincount(data.train_labels[bundle.fit], minlength=len(data.services))
    weights = torch.as_tensor(counts.sum() / (len(data.services) * np.maximum(counts, 1)),
                              dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    rng, mask_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    best, best_loss, stale, history = None, float("inf"), 0, []

    def logits_and_loss(values, labels):
        inputs = {name: torch.as_tensor(x, device=device) for name, x in values.items()}
        if base == "Eadro":
            logits = model(inputs, graph[None].expand(len(labels), -1, -1))
            return logits, F.cross_entropy(logits, labels, weight=weights)
        logits, encoded = model(inputs)
        alignment = .5 * (contrastive_loss(encoded["metrics"], encoded["logs"], config["temperature"])
                          + contrastive_loss(encoded["metrics"], encoded["traces"], config["temperature"]))
        return logits, F.cross_entropy(logits, labels, weight=weights) + .1 * alignment

    for epoch in range(config["max_epochs"]):
        model.train()
        order = rng.permutation(bundle.fit)
        losses = []
        for offset in range(0, len(order), config["batch_size"]):
            indices = order[offset:offset + config["batch_size"]]
            labels = torch.as_tensor(data.train_labels[indices], device=device)
            _, loss = logits_and_loss({name: value[indices] for name, value in prepared.items()}, labels)
            if method.endswith("+MD"):
                mask = dropout_mask(data.availability[indices], mask_rng)
                masked = {name: value[indices] * mask[:, :, MODALITIES[name], None, None]
                          for name, value in bundle.features.items()}
                masked, _ = descriptors(masked, bundle.split_bin, scaler)
                _, masked_loss = logits_and_loss(masked, labels)
                loss = .5 * loss + .5 * masked_loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite baseline training loss")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            inputs = {name: torch.as_tensor(value[bundle.validation], device=device)
                      for name, value in prepared.items()}
            if base == "Eadro":
                logits = model(inputs, graph[None].expand(len(bundle.validation), -1, -1))
            else:
                logits, _ = model(inputs)
            validation_loss = float(F.cross_entropy(
                logits, torch.as_tensor(data.train_labels[bundle.validation], device=device)))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, best, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    if best is None:
        raise ValueError("no finite baseline checkpoint")
    model.load_state_dict(best); model.eval()
    return model, scaler, adjacency, history


def fit_classical(bundle, method: str, config: dict):
    if method == "Nezha":
        return {name: classical._nezha_event_weight(bundle, name, config["event_threshold"])[1]
                for name in MODALITIES}
    if method == "SBLD":
        active = np.any(np.abs(bundle.features["logs"][:, :, bundle.split_bin:]) > config["active_threshold"], axis=2)
        weights = np.zeros(active.shape[1:], dtype=float)
        for service in range(bundle.service_count):
            hit = bundle.labels[bundle.fit] == service
            ef, ep = active[bundle.fit[hit], service].sum(0), active[bundle.fit[~hit], service].sum(0)
            weights[service] = classical.spectrum_formula(ef, ep, hit.sum()-ef, (~hit).sum()-ep, config["formula"])
        return weights
    if method == "LogFaultFlagger":
        x = modality_views(bundle.features["logs"][bundle.fit], bundle.split_bin)[2]
        y = (np.arange(bundle.service_count)[None] == bundle.labels[bundle.fit, None]).ravel()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=config["C"], class_weight="balanced",
                          max_iter=config["max_iter"], solver="liblinear", random_state=2027))
        model.fit(x.reshape(-1, x.shape[-1]), y)
        return model
    return None


def fit_trace(bundle, config, seed):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    trace = bundle.features["traces"].astype(np.float32)
    train = trace[bundle.fit, :, :bundle.split_bin].reshape(-1, trace.shape[-1])
    center, scale = train.mean(0), train.std(0)
    scale = np.where(scale > 1e-6, scale, 1.)
    tensor = torch.as_tensor((train-center)/scale)
    model = TraceVAE(trace.shape[-1], config["hidden_dim"], config["latent_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(seed); batch_size = min(256, len(tensor))
    for _ in range(config["epochs"]):
        model.train(); order = rng.permutation(len(tensor))
        for start in range(0,len(order),batch_size):
            x=tensor[order[start:start+batch_size]]
            reconstruction, mean, logvar=model(x)
            loss=F.mse_loss(reconstruction,x) - .005*torch.mean(1+logvar-mean.square()-logvar.exp())
            if not torch.isfinite(loss): raise ValueError("nonfinite VAE loss")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    return model, center, scale


def classical_scores(bundle, method: str, config: dict, fitted, adjacency: np.ndarray) -> np.ndarray:
    if method == "Nezha":
        scores = []
        for name in MODALITIES:
            active = (classical.robust_deviation(bundle.features[name], bundle.split_bin) > config["event_threshold"]).mean(2)
            values = active * fitted[name][None]
            values = values.max(-1) if config["service_aggregation"] == "max" else values.sum(-1)
            scores.append(row_unit_interval(values))
        direct = np.mean(scores, axis=0)
        propagated = np.stack([random_walk(adjacency, row, damping=.85, reverse=True) for row in direct])
        return (1-config["graph_mix"])*direct + config["graph_mix"]*row_unit_interval(propagated)
    if method == "SBLD":
        active = np.any(np.abs(bundle.features["logs"][:, :, bundle.split_bin:]) > config["active_threshold"], axis=2)
        values = active * fitted[None]
        return values.max(-1) if config["service_aggregation"] == "max" else values.sum(-1)
    if method == "LogFaultFlagger":
        x = modality_views(bundle.features["logs"], bundle.split_bin)[2]
        return fitted.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(x.shape[:2])
    # Existing deterministic scorers are reused verbatim. Their call graph is
    # the directed weighted graph fitted on train events, never test calls.
    original = classical.adjacency_for_incident
    classical.adjacency_for_incident = lambda _bundle, _index: adjacency.copy()
    try:
        return classical.METHODS[method](bundle, config)
    finally:
        classical.adjacency_for_incident = original


def run(dataset: str, fold: int, seed: int, method: str, output: Path, *, smoke=False, device="cpu") -> dict:
    started = time.perf_counter()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite baseline attempt")
    output.mkdir(parents=True, exist_ok=True)
    data = load_fold(dataset, fold)
    settings = json.loads(CONFIG.read_text())[dataset][method]
    if sha256(ROOT / settings["source"]) != settings["source_sha256"]:
        raise ValueError("actual frozen baseline configuration source changed")
    config = settings["configuration"]
    fit, validation = inner_indices(data) if method.removesuffix("+MD") in ("Eadro", "MULAN") else (data.train_indices, np.array([], dtype=int))
    bundle = make_bundle(data, fit, validation)
    graph = training_adjacency(bundle)
    meta = config_snapshot(f"python -m exp_revision.baseline_fold --method {method}",
                           {"dataset": dataset, "fold": fold, "seed": seed, "method": method,
                            "smoke": smoke, "device": device, "hyperparameters": config,
                            "graph": "fixed directed weighted model-fit graph",
                            "dropout": {"p": .5, "view_weights": [.5, .5], "restore": "originally available modalities"} if method.endswith("+MD") else None})
    meta.update(start_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                fit_ids=[data.train_ids[i] for i in fit], validation_ids=[data.train_ids[i] for i in validation])
    count = min(5, len(data.test_ids)) if smoke else len(data.test_ids)
    base = method.removesuffix("+MD")
    scaler, history, fitted = None, [], None
    if base in ("Eadro", "MULAN"):
        model, scaler, graph, history = fit_neural(data, bundle, method, config, seed, device)
        torch.save({"state_dict": model.state_dict(), "scaler": scaler, "adjacency": graph,
                    "configuration": config}, output / "model.pt")
    elif base != "TraceAnomaly":
        fitted = fit_classical(bundle, method, config)
        import joblib
        joblib.dump({"model": fitted, "adjacency": graph}, output / "model.joblib", compress=3)
    else:
        model, center, scale = fit_trace(bundle, config, seed)
        torch.save({"state_dict":model.state_dict(), "center":center, "scale":scale,
                    "configuration":config},output/"model.pt")
    scores, mask_hashes, errors = [], {}, []
    for scenario, param in SCENARIOS:
        path = ROOT / f"masks/{dataset}/{scenario}_{param or 'none'}/seed{seed}.npz"
        with np.load(path) as archive:
            mask = np.stack([archive[key] for key in data.test_ids[:count]])
        features = {name: value.copy() for name, value in bundle.features.items()}
        for name, modality in MODALITIES.items():
            features[name][len(data.train_ids):len(data.train_ids)+count] *= mask[:, :, modality, None, None]
        masked_bundle = copy.copy(bundle); masked_bundle.features = features
        try:
            if base in ("Eadro", "MULAN"):
                values = {name: x[len(data.train_ids):len(data.train_ids)+count] for name, x in features.items()}
                values, _ = descriptors(values, bundle.split_bin, scaler)
                inputs = {name: torch.as_tensor(value, device=device) for name, value in values.items()}
                with torch.no_grad():
                    if base == "Eadro":
                        logits = model(inputs, torch.as_tensor(graph, device=device)[None].expand(count, -1, -1))
                    else:
                        logits, _ = model(inputs)
                    score = logits.softmax(-1).cpu().numpy()
                if base == "MULAN":
                    propagated = np.stack([random_walk(graph, row, damping=.85, reverse=True) for row in score])
                    score = (1-config["graph_mix"])*row_unit_interval(score) + config["graph_mix"]*row_unit_interval(propagated)
            elif base == "TraceAnomaly":
                post=(features["traces"][len(data.train_ids):len(data.train_ids)+count,:,bundle.split_bin:]-center)/scale
                flat=torch.as_tensor(post.reshape(-1,post.shape[-1]),dtype=torch.float32)
                with torch.no_grad():
                    reconstruction,_,_=model(flat)
                    error=(reconstruction-flat).square().mean(-1).numpy().reshape(post.shape[:3])
                score=error.mean(-1) if config["aggregation"]=="mean" else error.max(-1) if config["aggregation"]=="max" else np.quantile(error,.90,axis=-1)
            else:
                score = classical_scores(masked_bundle, method, config, fitted, graph)[len(data.train_ids):len(data.train_ids)+count]
            if score.shape != (count, len(data.services)) or not np.isfinite(score).all():
                raise ValueError("invalid baseline score matrix")
        except Exception as error:
            score = np.full((count, len(data.services)), np.nan)
            errors.append({"scenario": scenario, "param": param, "error": repr(error)})
        scores.append(score)
        mask_hashes[str(path.relative_to(ROOT))] = sha256(path)
    np.savez_compressed(output / "scores.npz", scores=np.stack(scores),
                        incident_ids=np.array(data.test_ids[:count]),
                        scenario_keys=np.array([f"{s}_{p or 'none'}" for s,p in SCENARIOS]))
    model_path=output/("model.pt" if (output/"model.pt").exists() else "model.joblib")
    inputs=[CONFIG, ROOT/settings["source"],
            ROOT/f"cache/fold_preprocessing/{dataset}/fold{fold}/features.npz",
            ROOT/f"cache/fold_preprocessing/{dataset}/fold{fold}/metadata.json",
            ROOT/f"splits/{dataset}_training_labels/fold_{fold}.json", ROOT/f"splits/{dataset}_kfold5.json"]
    if dataset=="tt": inputs.append(ROOT/"splits/tt_inner_validation_kfold5.json")
    meta.update(status="failed_scores" if errors else "completed_scores", end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                elapsed_seconds=time.perf_counter()-started, test_count=count,
                scores_sha256=sha256(output / "scores.npz"), mask_sha256=mask_hashes,
                model_sha256=sha256(model_path),
                input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in inputs},
                errors=errors, training_history=history,
                source_sha256={name: sha256(ROOT / name) for name in (
                    "exp_revision/baseline_fold.py", "exp_revision/baseline_settings.json",
                    "exp_revision/fold_data.py", "scripts/care/fse2027_neural_baselines.py",
                    "scripts/care/fse2027_classical_baselines.py", "scripts/care/fse2027_baseline_common.py")})
    json_dump(output / "run_meta.json", meta)
    if errors:
        raise RuntimeError(f"failed baseline scenarios retained in scores: {errors}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-settings", action="store_true")
    parser.add_argument("--dataset", choices=("tt", "aiops"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.freeze_settings:
        freeze_settings(); return
    if not args.dataset or not args.method or not args.output:
        parser.error("dataset, method and output are required")
    result = run(args.dataset, args.fold, args.seed, args.method, args.output,
                 smoke=args.smoke, device=args.device)
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "test_count", "errors")}))


if __name__ == "__main__":
    main()
