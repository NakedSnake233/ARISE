"""Reuse frozen T1 experts for full raw-recomputed T2 and sanity baselines."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import time

import joblib
import numpy as np

from .common import ROOT, config_snapshot, json_dump, sha256, stable_seed
from .masked_evidence import MaskedEvidence, transform
from .missingness import SCENARIOS
from .train_evidence import fuse_all, predict

METHODS = ["ARISE", "ET-spec", "ET-spec+gate+aug", "RootFreqPrior", "Random"]


def completed_folder(root, dataset, fold, seed):
    state = json.loads((ROOT / root / dataset / "queue_state.json").read_text())
    row = state["jobs"][f"{dataset}_fold{fold}_seed{seed}"]
    if row["status"] != "completed": raise ValueError("prerequisite model is incomplete")
    return ROOT / row["output"]


def context_scores(dataset, fold, seed, ids):
    if dataset == "tt":
        state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
        row = state["jobs"][f"full_context_masks_tt_fold{fold}_seed{seed}"]
        if row["status"] != "completed": raise ValueError("masked context is incomplete")
        folder = ROOT / row["output"]
    else: folder = ROOT / f"results_v2/T2_context_full/aiops/aiops_fold{fold}_seed{seed}"
    meta = json.loads((folder / "run_meta.json").read_text())
    if meta["scores_sha256"] != sha256(folder / "scores.npz"): raise ValueError("context scores changed")
    with np.load(folder / "scores.npz") as archive:
        if archive["incident_ids"].tolist()[:len(ids)] != ids: raise ValueError("context test identity changed")
        if archive["scenario_keys"].tolist() != [f"{s}_{p or 'none'}" for s, p in SCENARIOS]: raise ValueError("context scenarios changed")
        scores = archive["probabilities"][:, :len(ids)].copy()
    return scores, folder


def infer(dataset, fold, seed, output, smoke=False):
    started = time.perf_counter(); start = dt.datetime.now(dt.timezone.utc).isoformat()
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable masked evidence output exists")
    output.mkdir(parents=True, exist_ok=True)
    pipeline = MaskedEvidence(dataset, fold); data = pipeline.data
    indices = data.test_indices[:5] if smoke else data.test_indices
    ids = [pipeline.ids[i] for i in indices]
    evidence = completed_folder("results_v2/T1_evidence_full", dataset, fold, seed)
    et = completed_folder("results_v2/T4_et_spec_full", dataset, fold, seed)
    expert_meta = json.loads((evidence / "run_meta.json").read_text())
    et_meta = json.loads((et / "run_meta.json").read_text())
    for path, expected in ((evidence / "specific.joblib", expert_meta["specific_model_sha256"]),
                           (evidence / "generic.joblib", expert_meta["generic_model_sha256"]),
                           (et / "model.joblib", et_meta["model_sha256"])):
        if sha256(path) != expected: raise ValueError("frozen model changed")
    specific, generic = joblib.load(evidence / "specific.joblib"), joblib.load(evidence / "generic.joblib")
    et_model = joblib.load(et / "model.joblib")
    beta = expert_meta["selected_beta"]
    base, context = context_scores(dataset, fold, seed, ids)
    prior = np.bincount(data.train_labels, minlength=len(data.services)).astype(float)
    random = np.zeros((len(ids), len(data.services)))
    for i, key in enumerate(ids):
        order = np.random.default_rng(stable_seed(dataset, key, "Random", seed)).permutation(len(data.services))
        random[i, order] = np.arange(len(data.services), 0, -1)
    scenarios, masks_hash, batches, specifics, generics, branches = [], {}, [], [], [], []
    errors = []
    for j, (scenario, param) in enumerate(SCENARIOS):
        path = ROOT / f"masks/{dataset}/{scenario}_{param or 'none'}/seed{seed}.npz"
        masks_hash[str(path.relative_to(ROOT))] = sha256(path)
        with np.load(path) as archive: masks = np.stack([archive[key] for key in ids])
        try:
            x, names, availability = pipeline.recompute(indices, masks, "family_gated")
            spec = predict(specific, transform(x, names, availability, "family_gated"))
            gen = predict(generic, transform(x, names, availability, "family_gated", generic=True))
            branch = fuse_all((1 - beta, spec), (beta, gen))
            final = fuse_all((.25, base[j]), (.75, branch))
            et_scores = predict(et_model, transform(x, names, availability, "recompute_ungated", append_flags=False))
            values = np.stack([final, et_scores, spec, np.repeat(prior[None], len(ids), axis=0), random])
        except Exception as error:
            errors.append({"scenario": scenario, "param": param, "error": repr(error)})
            values = np.full((len(METHODS), len(ids), len(data.services)), np.nan)
            spec = gen = branch = np.full((len(ids), len(data.services)), np.nan)
        batches.append(values); specifics.append(spec); generics.append(gen); branches.append(branch)
        scenarios.append(f"{scenario}_{param or 'none'}")
    scores = np.stack(batches, axis=1)
    # S0 must reproduce the old frozen experts and complete-data ET baseline.
    differences = {}
    for method, path, source in (("ARISE", ROOT / f"results_v2/T1_arise_s0/{dataset}/{dataset}_fold{fold}_seed{seed}/scores.npz", "probabilities"),
                                 ("ET-spec", et / "scores.npz", "probabilities"),
                                 ("ET-spec+gate+aug", evidence / "scores.npz", "specific")):
        with np.load(path) as archive:
            if archive["incident_ids"].tolist()[:len(ids)] != ids: raise ValueError("S0 prerequisite identity changed")
            expected = archive[source][:len(ids)]
        difference = float(np.max(np.abs(scores[METHODS.index(method), 0] - expected)))
        differences[method] = difference
        if not np.isfinite(difference) or difference > 1e-5:
            errors.append({"scenario": "S0", "method": method, "parity_difference": difference})
            scores[METHODS.index(method), 0] = np.nan
    np.savez_compressed(output / "scores.npz", scores=scores, method_names=METHODS,
                        scenario_keys=scenarios, incident_ids=ids, specific=np.stack(specifics),
                        generic=np.stack(generics), evidence=np.stack(branches), context=base)
    config = {"task": "T2", "dataset": dataset, "fold": fold, "seed": seed, "smoke": smoke,
              "methods": METHODS, "pipeline": "family_gated", "alpha": .75, "beta": beta,
              "training_protocol": "reuse unchanged T1 augmentation and models", "split": "kfold5"}
    meta = config_snapshot("python -m exp_revision.infer_evidence_masks", config)
    sources = [ROOT / name for name in ("exp_revision/infer_evidence_masks.py", "exp_revision/masked_evidence.py",
                "exp_revision/raw_spans_v2.py", "exp_revision/fold_data.py", "exp_revision/missingness.py",
                "scripts/care/evidence_reranker.py", "scripts/care/robust_evidence.py")]
    inputs = [evidence / name for name in ("specific.joblib", "generic.joblib", "run_meta.json", "scores.npz")]
    inputs += [et / name for name in ("model.joblib", "run_meta.json", "scores.npz")]
    inputs += [context / "scores.npz", context / "run_meta.json"]
    meta.update(status="failed_scores" if errors else "completed_T2_evidence_sanity", errors=errors,
                start_time=start, end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                elapsed_seconds=time.perf_counter() - started, s0_max_abs_differences=differences,
                mask_sha256=masks_hash, input_sha256={**pipeline.inputs, **{str(p.relative_to(ROOT)): sha256(p) for p in inputs}},
                source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources}, scores_sha256=sha256(output / "scores.npz"))
    json_dump(output / "run_meta.json", meta)
    if errors: raise RuntimeError("masked evidence failed; scores and errors retained")
    return output


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); print(infer(a.dataset, a.fold, a.seed, a.output, a.smoke))


if __name__ == "__main__": main()
