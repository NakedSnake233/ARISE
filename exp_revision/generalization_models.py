"""Original context, dual experts and neural baselines on the frozen T5 cohorts."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import torch

from .common import ROOT, config_snapshot, dataset_dir, json_dump, sha256, stable_seed
from .generalization_data import load_cohort, part_for, split_file, training_file
from .masked_evidence import MaskedEvidence, transform
from .train_evidence import _fit_one, _predict_one, fit, model_for, predict, fuse_all
from .train_variant_evidence import VariantTrainer
from .train_context_variant import fit_context
from . import baseline_fold as baseline
from evidence_reranker import extract

EXPERT_METHODS = ["ET-spec", "ET-spec+gate+aug", "RootFreqPrior", "Random"]


def scenarios(task):
    return ["S0_none", "S2_0.50", "S3_trace"] if task == "T5a" else ["S0_none"]


def get_masks(data, task, seed):
    masks, hashes = {}, {}
    for key in scenarios(task):
        path = ROOT / f"masks/{data.dataset}/{key}/seed{seed}.npz"
        with np.load(path) as archive: masks[key] = np.stack([archive[incident] for incident in data.test_ids])
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    return masks, hashes


class CohortEvidence(MaskedEvidence):
    def __init__(self, data, folder, task, seed):
        self.data = data; self.ids = data.train_ids + data.test_ids
        unlabeled = ROOT / f"splits/{data.dataset}_unlabeled_events.json"
        self.rows = {row["incident_id"]: row for row in json.loads(unlabeled.read_text())}
        paths = [folder / name for name in ("features.npz", "metadata.json", "run_meta.json")]
        paths += [unlabeled, split_file(data.dataset, task), training_file(data.dataset, task, data.fold)]
        self.inputs = {str(p.relative_to(ROOT)): sha256(p) for p in paths}; self.raw_metrics = None
        if data.dataset == "tt":
            path = ROOT / "cache/raw_metric_tt.npz"
            names = json.loads((dataset_dir("tt") / "metadata.json").read_text())["metric_names"]
            channels = [names.index(name) for name in data.metadata["metric_names"]]
            with np.load(path) as archive:
                lookup = {key: i for i, key in enumerate(archive["incident_ids"].tolist())}
                self.raw_metrics = archive["metric"][[lookup[key] for key in self.ids]][:, :, :, channels].copy()
            self.inputs[str(path.relative_to(ROOT))] = sha256(path)
        self.spans = {}
        state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
        for key, row in state["jobs"].items():
            if not key.startswith(f"full_rpc_cache_{data.dataset}_") or row["status"] != "completed": continue
            path = ROOT / row["output"] / "run_meta.json"; meta = json.loads(path.read_text())
            if meta["status"] != "completed_span_cache": raise ValueError("raw span cache incomplete")
            self.inputs[str(path.relative_to(ROOT))] = sha256(path)
            for incident, entry in meta["incident_files"].items():
                if incident in self.spans: raise ValueError("duplicate raw span entry")
                self.spans[incident] = entry
        if not set(self.ids).issubset(self.spans): raise ValueError("cohort raw spans missing")
        rows = [{"incident_id": key, "calls": data.metadata["calls"][key]} for key in self.ids]
        x, names = extract({name: getattr(data, name) for name in baseline.MODALITIES}, data.metadata, rows, self.raw_metrics)
        rpc = [self.rpc(key, np.ones(len(data.services), dtype=np.uint8).tobytes()) for key in self.ids]
        self.cached = np.concatenate([x, np.stack([value for value, _ in rpc])], axis=-1)
        self.names = names + rpc[0][1]


class CachedTrainer(VariantTrainer):
    """Preserve T1's cached training augmentation, not T3's changed pipeline."""
    def fit_pair(self, indices, setting):
        self.views(indices)
        evidence = self.evidence; data = evidence.data
        if setting["augment"]:
            return tuple(_fit_one(evidence.cached, evidence.names, data.train_labels, data.availability,
                indices, self.seed, type_agnostic=generic)[0] for generic in (False, True))
        models = []
        for generic in (False, True):
            x = transform(evidence.cached[indices], evidence.names, data.availability[indices], "family_gated", generic=generic)
            model = model_for(None); model.set_params(random_state=self.seed)
            models.append(fit(model, x, data.train_labels[indices]))
        return tuple(models)

    def score_pair(self, models, indices, masks, setting):
        if not np.all(masks == 1): raise ValueError("cached inner beta scoring must use complete views")
        evidence = self.evidence
        return tuple(_predict_one(model, evidence.cached[indices], evidence.names, evidence.data.availability[indices], type_agnostic=generic)
                     for model, generic in zip(models, (False, True)))


def begin(output):
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable T5 model output exists")
    output.mkdir(parents=True, exist_ok=True)
    return dt.datetime.now(dt.timezone.utc).isoformat(), time.perf_counter()


def record(output, data, task, seed, smoke, component, scores, start, clock, inputs, mask_hashes, **extra):
    config = dict(task=task, dataset=data.dataset, fold=data.fold, seed=seed, smoke=smoke,
        component=component, split=part_for(data.dataset, task, data.fold)["split"], scenarios=scenarios(task), alpha=.75)
    config.update(extra.pop("configuration", {}))
    meta = config_snapshot(f"python -m exp_revision.generalization_models --component {component}", config)
    valid = bool(np.isfinite(scores).all())
    meta.update(status=f"completed_generalization_{component}" if valid else "failed_scores", errors=[] if valid else [dict(error="nonfinite scores")], start_time=start,
        end_time=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-clock,
        input_sha256=inputs, mask_sha256=mask_hashes, scores_sha256=sha256(output / "scores.npz"),
        source_sha256={name: sha256(ROOT / name) for name in ("exp_revision/generalization_models.py", "exp_revision/generalization_data.py",
            "exp_revision/train_evidence.py", "exp_revision/train_variant_evidence.py", "exp_revision/train_context_variant.py",
            "exp_revision/masked_evidence.py", "exp_revision/raw_spans_v2.py", "exp_revision/baseline_fold.py")}, **extra)
    json_dump(output / "run_meta.json", meta)
    if not valid: raise ValueError("nonfinite T5 scores retained")
    return output


def train_experts(dataset, task, fold, seed, output, smoke=False):
    start, clock = begin(output); data, folder = load_cohort(dataset, task, fold, smoke)
    evidence = CohortEvidence(data, folder, task, seed); trainer = CachedTrainer(evidence, seed)
    setting = dict(augment=True, pipeline="family_gated", flags=True, drop=None)
    models, selection = trainer.train_variant(setting); model_hashes = {}
    def save_pair(pair, directory, selected):
        directory.mkdir(exist_ok=True)
        for name, model in zip(("specific", "generic"), pair):
            path = directory / f"{name}.joblib"; joblib.dump(model, path, compress=3)
            model_hashes[str(path.relative_to(ROOT))] = sha256(path)
        json_dump(directory / "inner_beta.json", selected)
    save_pair(models, output / "dual", selection)
    noaug_models, noaug_selection = None, None
    if task == "T5a":
        noaug_models, noaug_selection = trainer.train_variant({**setting, "augment": False})
        save_pair(noaug_models, output / "noaug", noaug_selection)
    trainer.save_views(output)
    et = model_for(None); et.set_params(random_state=seed)
    fit(et, evidence.cached[data.train_indices], data.train_labels)
    joblib.dump(et, output / "et.joblib", compress=3); model_hashes[str((output / "et.joblib").relative_to(ROOT))] = sha256(output / "et.joblib")
    prior = np.repeat(np.bincount(data.train_labels, minlength=len(data.services))[None].astype(float), len(data.test_ids), axis=0)
    random = np.zeros_like(prior)
    for i, key in enumerate(data.test_ids):
        order = np.random.default_rng(stable_seed(dataset, key, "Random", seed)).permutation(len(data.services))
        random[i, order] = np.arange(len(data.services), 0, -1)
    masks, mask_hashes = get_masks(data, task, seed)
    batches, spec_values, gen_values, branches, noaug_values = [], [], [], [], []
    for key in scenarios(task):
        x, names, a = evidence.recompute(data.test_indices, masks[key], "family_gated")
        spec, gen = (predict(model, transform(x, names, a, "family_gated", generic=generic)) for model, generic in zip(models, (False, True)))
        beta = selection["selected_beta"]; branch = fuse_all((1-beta, spec), (beta, gen))
        et_scores = predict(et, transform(x, names, a, "recompute_ungated", append_flags=False))
        batches.append(np.stack([et_scores, spec, prior, random])); spec_values.append(spec); gen_values.append(gen); branches.append(branch)
        if noaug_models:
            specific, generic = (predict(model, transform(x, names, a, "family_gated", generic=g)) for model, g in zip(noaug_models, (False, True)))
            beta = noaug_selection["selected_beta"]; noaug_values.append(fuse_all((1-beta, specific), (beta, generic)))
    scores = np.stack(batches, axis=1)
    extras = {"noaug_evidence": np.stack(noaug_values)} if noaug_values else {}
    np.savez_compressed(output / "scores.npz", scores=scores, specific=np.stack(spec_values), generic=np.stack(gen_values), evidence=np.stack(branches),
        method_names=EXPERT_METHODS, incident_ids=data.test_ids, scenario_keys=scenarios(task), **extras)
    return record(output, data, task, seed, smoke, "experts", scores, start, clock, evidence.inputs, mask_hashes,
        configuration=dict(methods=EXPERT_METHODS, training_protocol="unchanged T1 cached seven-view augmentation", noaug=task == "T5a"),
        beta_selection=selection, noaug_beta_selection=noaug_selection, model_sha256=model_hashes,
        training_masks_sha256=sha256(output / "training_masks.npz"), training_mask_cohorts_sha256=sha256(output / "training_mask_cohorts.json"))


def data_inputs(data, folder, task):
    paths = [folder / name for name in ("features.npz", "metadata.json", "run_meta.json")]
    paths += [split_file(data.dataset, task), training_file(data.dataset, task, data.fold), ROOT / "configs/care/review-response-v1.json"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def inner_indices(data, task, name):
    partition = part_for(data.dataset, task, data.fold)[name]
    index = {key: i for i, key in enumerate(data.train_ids)}
    fit, validation = [np.asarray([index[key] for key in partition[field]], dtype=int) for field in ("fit", "validation")]
    groups = np.asarray(data.train_groups)
    if (set(fit) & set(validation) or set(fit) | set(validation) != set(data.train_indices)
            or set(groups[fit]) & set(groups[validation])): raise ValueError("invalid group-isolated inner cohort")
    return fit, validation


def train_context(dataset, task, fold, seed, output, smoke=False):
    start, clock = begin(output); data, folder = load_cohort(dataset, task, fold, smoke)
    device = "cuda" if dataset == "tt" and not smoke else "cpu"
    fit_ids, validation = inner_indices(data, task, "context_inner")
    infer, training = fit_context(data, fit_ids, validation, seed, "full", device, output)
    masks, hashes = get_masks(data, task, seed)
    scores = np.stack([infer(data.test_indices, masks[key]) for key in scenarios(task)])
    np.savez_compressed(output / "scores.npz", scores=scores, incident_ids=data.test_ids, scenario_keys=scenarios(task))
    return record(output, data, task, seed, smoke, "context", scores, start, clock, data_inputs(data, folder, task), hashes,
        configuration=dict(method="context-only", device=device), training=training, model_sha256=sha256(output / "model.pt"))


def neural_scores(model, data, features, scaler, adjacency, method, cfg):
    prepared, _ = baseline.descriptors(features, data.metadata["before"] // data.metadata["step"], scaler)
    with torch.no_grad():
        inputs = {name: torch.as_tensor(value, dtype=torch.float32) for name, value in prepared.items()}
        if method == "Eadro": logits = model(inputs, torch.as_tensor(adjacency)[None].expand(len(data.test_ids), -1, -1))
        else: logits = model(inputs)[0]
        scores = logits.softmax(-1).numpy()
    if method == "MULAN":
        propagated = np.stack([baseline.random_walk(adjacency, row, damping=.85, reverse=True) for row in scores])
        scores = (1-cfg["graph_mix"])*baseline.row_unit_interval(scores) + cfg["graph_mix"]*baseline.row_unit_interval(propagated)
    return scores


def train_baseline(dataset, task, fold, seed, output, method, smoke=False):
    start, clock = begin(output); data, folder = load_cohort(dataset, task, fold, smoke)
    settings = json.loads(baseline.CONFIG.read_text()); cfg = settings[dataset][method]["configuration"]
    fit_ids, validation = inner_indices(data, task, "baseline_inner")
    bundle = baseline.make_bundle(data, fit_ids, validation)
    model, scaler, adjacency, history = baseline.fit_neural(data, bundle, method, cfg, seed, "cpu")
    torch.save(dict(state_dict=model.state_dict(), scaler=scaler, adjacency=adjacency,
        fit_ids=[data.train_ids[i] for i in fit_ids], validation_ids=[data.train_ids[i] for i in validation]), output / "model.pt")
    json_dump(output / "history.json", history)
    masks, hashes = get_masks(data, task, seed); batches = []
    for key in scenarios(task):
        features = {name: getattr(data, name)[data.test_indices] * masks[key][:, :, m, None, None] for name, m in baseline.MODALITIES.items()}
        batches.append(neural_scores(model, data, features, scaler, adjacency, method, cfg))
    scores = np.stack(batches)
    np.savez_compressed(output / "scores.npz", scores=scores, incident_ids=data.test_ids, scenario_keys=scenarios(task))
    inputs = data_inputs(data, folder, task); inputs[str(baseline.CONFIG.relative_to(ROOT))] = sha256(baseline.CONFIG)
    return record(output, data, task, seed, smoke, "baseline", scores, start, clock, inputs, hashes,
        configuration=dict(method=method, hyperparameters=cfg, device="cpu"), model_sha256=sha256(output / "model.pt"),
        fit_ids=[data.train_ids[i] for i in fit_ids], validation_ids=[data.train_ids[i] for i in validation])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    p.add_argument("--task", choices=("T5a", "T5b"), required=True); p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--smoke", action="store_true")
    p.add_argument("--component", choices=("experts", "context", "baseline"), required=True)
    p.add_argument("--method", choices=("Eadro", "MULAN")); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    args = (a.dataset, a.task, a.fold, a.seed, a.output)
    if a.component == "baseline": print(train_baseline(*args, method=a.method, smoke=a.smoke))
    else: print((train_experts if a.component == "experts" else train_context)(*args, smoke=a.smoke))


if __name__ == "__main__": main()
