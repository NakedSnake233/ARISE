"""Recover deadline-interrupted variant suites using verified saved model pairs.

Scores are always recomputed. Incomplete historical folders are never presented
as successful runs; their files are frozen as explicit recovery inputs.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q
from . import train_variant_evidence as training
from robust_evidence import fault_type_holdout


def eligible(row):
    cfg = row["spec"]["expected_config"]
    attempts = row["attempts"]
    last = attempts[-1] if attempts else {}
    tail = q.log_tail(ROOT / last["log"]) if last.get("log") else ""
    explicit_error = any(x in tail.lower() for x in ("traceback (most recent call last)", "out of memory", "no module named"))
    deadline_exit = last.get("exit_code") == 124 or (last.get("exit_code") is None and '"reason": "signal"' in tail)
    return (row["status"] == "stuck" and cfg.get("task") in ("T3", "T6")
        and "variants" in cfg and not cfg.get("smoke") and bool(attempts)
        and all(a.get("status") == "interrupted" for a in attempts)
        and not explicit_error and deadline_exit
        and last.get("elapsed_seconds", 0) >= last.get("runtime_timeout_seconds", row["spec"]["timeout_seconds"]))


def create():
    state = json.loads((q.FOLDER / "queue_state.json").read_text())
    covered = set()
    for path in q.MANIFESTS.glob("*.json"):
        covered.update(json.loads(path.read_text()).get("recovery_recipes", {}))
    targets = {name: row for name, row in state["jobs"].items()
        if eligible(row) and name not in covered and not name.startswith("recovered_")}
    if not targets: return None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = str(time.time_ns()); relative = f"manifests/remaining/21_timeout_variant_recovery_{stamp}.json"
    sources, jobs, recipes = {}, [], {}
    for name, row in targets.items():
        spec = row["spec"]; cfg = spec["expected_config"]; partial = {}
        for path in q.MANIFESTS.glob("*.json"):
            manifest = json.loads(path.read_text())
            if any(j["id"] == name for j in manifest["jobs"]):
                sources.update(manifest["source_sha256"]); sources[str(path.relative_to(ROOT))] = sha256(path)
        for index, method in enumerate(cfg["methods"]):
            for attempt in reversed(row["attempts"]):
                directory = ROOT / attempt["output"] / f"variant{index}"
                files = [directory / f for f in ("specific.joblib", "generic.joblib", "inner_beta.json")]
                if all(f.is_file() for f in files):
                    partial[method] = dict(directory=str(directory.relative_to(ROOT)), attempt=attempt["number"],
                        sha256={str(f.relative_to(ROOT)): sha256(f) for f in files})
                    sources.update(partial[method]["sha256"]); break
        replacement = f"recovered_{name}"
        recipes[name] = dict(recovery_job=replacement, original_spec=spec, attempts=row["attempts"], partial_models=partial)
        jobs.append({**spec, "id": replacement, "source_commit": commit, "timeout_seconds": 2400,
            "command": ["-m", "exp_revision.variant_timeout_recovery", "--manifest", relative, "--original-id", name]})
    for name in ("exp_revision/variant_timeout_recovery.py", "exp_revision/adaptive_queue.py", "exp_revision/priority_queue.py"):
        sources[name] = sha256(ROOT / name)
    manifest = dict(backend="local", phase="timeout_variant_recovery", source_commit=commit,
        precharged_hours=q.PRECHARGED, source_sha256=sources, jobs=jobs, recovery_recipes=recipes,
        reason="Explicit recovery after exhausted deadline attempts; saved models verified, all scores recomputed")
    json_dump(ROOT / relative, manifest); return ROOT / relative


def expected_holdouts(trainer):
    data = trainer.evidence.data
    rows = [dict(fault_type=t, overlap_group=g) for t, g in zip(data.train_fault_types, data.train_groups)]
    result = []
    for fault in sorted(set(data.train_fault_types)):
        fit, held, purged, groups = fault_type_holdout(rows, data.train_indices, fault)
        result.append(dict(fault_type=fault, fit_ids=[data.train_ids[i] for i in fit],
            held_ids=[data.train_ids[i] for i in held], purged_ids=[data.train_ids[i] for i in purged], held_groups=groups))
    return result


def load_pair(trainer, method, setting, partial):
    directory = ROOT / partial["directory"]
    for name, digest in partial["sha256"].items():
        if sha256(ROOT / name) != digest: raise ValueError(f"recovery model input changed: {name}")
    selection = json.loads((directory / "inner_beta.json").read_text())
    if selection["holdouts"] != expected_holdouts(trainer): raise ValueError("recovered holdouts differ from original train-only cohorts")
    grid = selection["grid"]
    if [x["beta"] for x in grid] != [0., .25, .5, .75, 1.]: raise ValueError("recovered beta grid differs")
    if not all(np.isfinite(list(x["metrics"].values())).all() and all(0 <= v <= 1 for v in x["metrics"].values()) for x in grid):
        raise ValueError("invalid recovered training-only metrics")
    chosen = max(grid, key=lambda x: (x["metrics"]["PR@1"], x["metrics"]["MAP@5"], -x["beta"]))["beta"]
    if chosen != selection["selected_beta"]: raise ValueError("recovered beta selection differs")
    data = trainer.evidence.data; sample = data.train_indices[:1]
    x, names, a = trainer.evidence.recompute(sample, np.ones_like(data.availability[sample]), setting["pipeline"])
    x = training.remove_group(x, names, setting["drop"])
    models = []
    for generic, name in zip((False, True), ("specific.joblib", "generic.joblib")):
        model = joblib.load(directory / name)
        expected = dict(random_state=trainer.seed, n_estimators=256, min_samples_leaf=3,
            max_features=.7, class_weight="balanced", n_jobs=4, max_depth=None)
        if any(model.get_params().get(k) != v for k, v in expected.items()): raise ValueError("recovered model parameters differ")
        features = training.transform(x, names, a, setting["pipeline"], generic=generic, append_flags=setting["flags"])
        if model.n_features_in_ != features.shape[-1] or not np.array_equal(model.classes_, [0, 1]) or len(model.estimators_) != 256:
            raise ValueError("recovered model shape differs")
        models.append(model)
    return tuple(models), selection


def recover(manifest_path, original_id, output):
    manifest = json.loads(manifest_path.read_text()); recipe = manifest["recovery_recipes"][original_id]
    for name, digest in manifest["source_sha256"].items():
        if sha256(ROOT / name) != digest: raise ValueError(f"frozen recovery input changed: {name}")
    cfg = recipe["original_spec"]["expected_config"]; original_class = training.VariantTrainer
    reused, retrained, discarded = [], [], {}
    class ResumingTrainer(original_class):
        def train_variant(self, setting):
            method = next(n for n, s in cfg["variants"].items() if s == setting)
            partial = recipe["partial_models"].get(method)
            if partial:
                try: result = load_pair(self, method, setting, partial)
                except (OSError, ValueError, EOFError, KeyError, TypeError, pickle.UnpicklingError, AttributeError) as error:
                    discarded[method] = repr(error)
                else:
                    # Register the same deterministic masks, in the same cohort order,
                    # as the original fit. These masks are rebuilt, not historical receipts.
                    index = {name: i for i, name in enumerate(self.evidence.data.train_ids)}
                    for held in result[1]["holdouts"]:
                        self.views(np.asarray([index[name] for name in held["fit_ids"]], dtype=int))
                    self.views(self.evidence.data.train_indices)
                    reused.append(method); return result
            retrained.append(method); return super().train_variant(setting)
    original_dump = training.json_dump
    def provisional(path, value):
        if path == output / "run_meta.json" and value.get("status") == "completed_variant_evidence":
            value = {**value, "status": "pending_recovery_receipt"}
        return original_dump(path, value)
    training.VariantTrainer = ResumingTrainer; training.json_dump = provisional
    try: training.train(cfg["dataset"], cfg["fold"], cfg["seed"], cfg["task"], output, smoke=False)
    finally: training.VariantTrainer = original_class; training.json_dump = original_dump
    meta = json.loads((output / "run_meta.json").read_text())
    if meta["status"] != "pending_recovery_receipt" or meta.get("errors"):
        raise ValueError("recovery training did not produce a valid provisional receipt")
    meta["input_sha256"].update({str(manifest_path.relative_to(ROOT)): sha256(manifest_path),
        **{p: digest for item in recipe["partial_models"].values() for p, digest in item["sha256"].items()}})
    meta["source_sha256"]["exp_revision/variant_timeout_recovery.py"] = sha256(Path(__file__))
    meta["completion_protocol"] = "saved model pairs verified at recovery; original scoring rerun; cohort masks deterministically rebuilt"
    meta["recovery"] = dict(original_id=original_id, preserved_attempts=recipe["attempts"],
        reused_methods=reused, retrained_methods=retrained, rejected_partial_models=discarded,
        partial_models=recipe["partial_models"], historical_final_receipt_present=False)
    meta["status"] = "completed_variant_evidence"
    meta["completed_recovery_receipt"] = True
    json_dump(output / "run_meta.json", meta)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--create", action="store_true")
    p.add_argument("--manifest", type=Path); p.add_argument("--original-id"); p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.create: print(create())
    else: recover(ROOT / a.manifest, a.original_id, a.output)


if __name__ == "__main__": main()
