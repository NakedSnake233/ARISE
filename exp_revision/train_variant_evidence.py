"""Train source-pipeline variants with identical original seven augmentation views."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import time

import joblib
import numpy as np

from .common import ROOT, config_snapshot, json_dump, sha256
from .infer_evidence_masks import context_scores
from .masked_evidence import MaskedEvidence, transform
from .metrics import ranking_metrics
from .missingness import SCENARIOS
from .train_evidence import _ranks, fit, model_for, predict
from robust_evidence import _augmentation_availabilities, fault_type_holdout, feature_groups, fuse_all

T3_VARIANTS = {
    "ARISE[V-a]": dict(pipeline="cached_leaky", augment=True, flags=True, drop=None),
    "ARISE[V-b]": dict(pipeline="recompute_ungated", augment=True, flags=True, drop=None),
    "ARISE[V-c]": dict(pipeline="family_gated", augment=True, flags=True, drop=None),
    "ARISE[V-d]": dict(pipeline="edge_gated", augment=True, flags=True, drop=None),
}
T6_VARIANTS = {
    "ARISE w/o RPC residual evidence": dict(pipeline="family_gated", augment=True, flags=True, drop="rpc"),
    "ARISE w/o incident graph/RPC evidence": dict(pipeline="family_gated", augment=True, flags=True, drop="graph"),
    "ARISE w/o call-neighborhood evidence": dict(pipeline="family_gated", augment=True, flags=True, drop="call_graph"),
    "ARISE w/o availability indicators": dict(pipeline="family_gated", augment=True, flags=False, drop=None),
    "ARISE w/o missing-view augmentation": dict(pipeline="family_gated", augment=False, flags=True, drop=None),
}
T6_SCENARIOS = ["S0_none", "S1_metric", "S1_log", "S1_trace", "S2_0.50", "S3_trace"]


def remove_group(x, names, group):
    if group:
        x = x.copy(); x[:, :, feature_groups(names)[group]] = 0
    return x


class VariantTrainer:
    def __init__(self, evidence, seed):
        self.evidence, self.seed = evidence, seed
        self.view_masks, self.cohorts = {}, {}

    def views(self, indices):
        key = tuple(int(i) for i in indices)
        if key not in self.view_masks:
            if not len(indices) or min(indices) < 0 or max(indices) >= len(self.evidence.data.train_ids):
                raise ValueError("augmentation cohort must contain training events only")
            self.view_masks[key] = _augmentation_availabilities((len(indices), len(self.evidence.data.services)),
                                                               self.seed, (.25, .5, .75))
            self.cohorts[key] = f"cohort{len(self.cohorts)}"
        return self.view_masks[key]

    def fit_pair(self, indices, setting):
        views = self.views(indices)
        specific, generic = [], []
        for view in views if setting["augment"] else views[:1]:
            x, names, a = self.evidence.recompute(indices, view, setting["pipeline"])
            x = remove_group(x, names, setting["drop"])
            specific.append(transform(x, names, a, setting["pipeline"], append_flags=setting["flags"]))
            generic.append(transform(x, names, a, setting["pipeline"], generic=True, append_flags=setting["flags"]))
        labels = np.tile(self.evidence.data.train_labels[indices], len(specific))
        models = []
        for views_x in (specific, generic):
            model = model_for(None)
            model.set_params(random_state=self.seed, n_estimators=256, min_samples_leaf=3,
                             max_features=.7, class_weight="balanced", n_jobs=4)
            models.append(fit(model, np.concatenate(views_x), labels))
        return tuple(models)

    def score_pair(self, models, indices, masks, setting):
        x, names, a = self.evidence.recompute(indices, masks, setting["pipeline"])
        x = remove_group(x, names, setting["drop"])
        return tuple(predict(model, transform(x, names, a, setting["pipeline"], generic=generic,
                                              append_flags=setting["flags"]))
                     for model, generic in zip(models, (False, True)))

    def train_variant(self, setting):
        data = self.evidence.data; indices = data.train_indices
        rows = [dict(fault_type=t, overlap_group=g) for t, g in zip(data.train_fault_types, data.train_groups)]
        scores_s, scores_g, labels, holdouts = [], [], [], []
        for fault in sorted(set(data.train_fault_types)):
            fit_ids, held, purged, groups = fault_type_holdout(rows, indices, fault)
            if not len(fit_ids) or not len(held): raise ValueError("empty training-only fault holdout")
            models = self.fit_pair(fit_ids, setting)
            ones = np.ones_like(data.availability[held])
            spec, gen = self.score_pair(models, held, ones, setting)
            scores_s.append(spec); scores_g.append(gen); labels.append(data.train_labels[held])
            holdouts.append(dict(fault_type=fault, fit_ids=[data.train_ids[i] for i in fit_ids],
                held_ids=[data.train_ids[i] for i in held], purged_ids=[data.train_ids[i] for i in purged], held_groups=groups))
        spec, gen, truth = np.concatenate(scores_s), np.concatenate(scores_g), np.concatenate(labels)
        grid = []
        for beta in (0., .25, .5, .75, 1.):
            metrics = ranking_metrics(_ranks(fuse_all((1 - beta, spec), (beta, gen)), truth))
            grid.append(dict(beta=beta, metrics=metrics))
        beta = max(grid, key=lambda r: (r["metrics"]["PR@1"], r["metrics"]["MAP@5"], -r["beta"]))["beta"]
        return self.fit_pair(indices, setting), dict(selected_beta=beta, grid=grid, holdouts=holdouts)

    def save_views(self, output):
        arrays, cohorts = {}, {}
        for indices, views in self.view_masks.items():
            name = self.cohorts[indices]
            arrays[name] = np.stack(views).astype(np.uint8)
            cohorts[name] = [self.evidence.data.train_ids[i] for i in indices]
        np.savez_compressed(output / "training_masks.npz", **arrays)
        json_dump(output / "training_mask_cohorts.json", cohorts)


def train(dataset, fold, seed, task, output, smoke=False):
    started = time.perf_counter(); start = dt.datetime.now(dt.timezone.utc).isoformat()
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable variant output exists")
    output.mkdir(parents=True, exist_ok=True)
    evidence = MaskedEvidence(dataset, fold); data = evidence.data
    indices = data.test_indices[:5] if smoke else data.test_indices
    ids = [evidence.ids[i] for i in indices]
    context, context_folder = context_scores(dataset, fold, seed, ids)
    settings = T3_VARIANTS if task == "T3" else T6_VARIANTS
    scenarios = [f"{s}_{p or 'none'}" for s, p in SCENARIOS] if task == "T3" else T6_SCENARIOS
    masks, mask_hashes = {}, {}
    for scenario in scenarios:
        path = ROOT / f"masks/{dataset}/{scenario}/seed{seed}.npz"
        with np.load(path) as archive: masks[scenario] = np.stack([archive[key] for key in ids])
        mask_hashes[str(path.relative_to(ROOT))] = sha256(path)
    trainer = VariantTrainer(evidence, seed); scores, selections, model_hashes = [], {}, {}
    errors = []
    all_scenarios = [f"{s}_{p or 'none'}" for s, p in SCENARIOS]
    for variant_index, (method, setting) in enumerate(settings.items()):
        directory = output / f"variant{variant_index}"; directory.mkdir()
        try:
            models, selection = trainer.train_variant(setting)
            selections[method] = selection
            for name, model in zip(("specific", "generic"), models):
                joblib.dump(model, directory / f"{name}.joblib", compress=3)
                model_hashes[str((directory / f"{name}.joblib").relative_to(ROOT))] = sha256(directory / f"{name}.joblib")
            json_dump(directory / "inner_beta.json", selection)
            values = []
            for scenario in scenarios:
                spec, gen = trainer.score_pair(models, indices, masks[scenario], setting)
                branch = fuse_all((1 - selection["selected_beta"], spec), (selection["selected_beta"], gen))
                values.append(fuse_all((.25, context[all_scenarios.index(scenario)]), (.75, branch)))
            scores.append(np.stack(values))
            print(json.dumps(dict(method=method, beta=selection["selected_beta"], elapsed_seconds=time.perf_counter() - started)), flush=True)
        except Exception as error:
            errors.append(dict(method=method, error=repr(error)))
            scores.append(np.full((len(scenarios), len(ids), len(data.services)), np.nan))
            print(json.dumps(errors[-1]), flush=True)
    trainer.save_views(output)
    np.savez_compressed(output / "scores.npz", scores=np.stack(scores), method_names=list(settings),
                        scenario_keys=scenarios, incident_ids=ids)
    config = dict(task=task, dataset=dataset, fold=fold, seed=seed, smoke=smoke, methods=list(settings),
                  variants=settings, scenarios=scenarios, alpha=.75, split="kfold5", preregistered=False if task == "T6" else None)
    meta = config_snapshot("python -m exp_revision.train_variant_evidence", config)
    sources = [ROOT / name for name in ("exp_revision/train_variant_evidence.py", "exp_revision/masked_evidence.py",
               "exp_revision/raw_spans_v2.py", "exp_revision/train_evidence.py", "scripts/care/robust_evidence.py",
               "scripts/care/evidence_reranker.py")]
    meta.update(status="failed_scores" if errors else "completed_variant_evidence", errors=errors, start_time=start,
        end_time=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
        beta_selections=selections, model_sha256=model_hashes, mask_sha256=mask_hashes,
        training_masks_sha256=sha256(output / "training_masks.npz"), training_mask_cohorts_sha256=sha256(output / "training_mask_cohorts.json"),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        input_sha256={**evidence.inputs, **{str((context_folder / n).relative_to(ROOT)): sha256(context_folder / n) for n in ("run_meta.json", "scores.npz")}},
        scores_sha256=sha256(output / "scores.npz"))
    json_dump(output / "run_meta.json", meta)
    if errors: raise RuntimeError("variant failure; scores/errors retained")
    return output


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--task", choices=("T3", "T6"), required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); print(train(a.dataset, a.fold, a.seed, a.task, a.output, a.smoke))


if __name__ == "__main__": main()
