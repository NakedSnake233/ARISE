"""Compose requested T5 fusions and T6 reuses without fitting any model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import ROOT, SEEDS, config_snapshot, json_dump, sha256
from .fold_data import load_fold
from .generalization_data import part_for, split_file
from .generalization_models import EXPERT_METHODS, scenarios as t5_scenarios
from .generalization_model_manifest import identifier
from .missingness import SCENARIOS
from . import remaining_queue as q
from .score_receipts import archive_axes, check_hashes, fused, read_job
from .train_variant_evidence import T3_VARIANTS, T6_SCENARIOS

T6_METHODS = ["ARISE", "ARISE w/o channel-aggregated expert", "context-only", "ET-spec",
              "Dual-view evidence without context fusion", "ARISE[V-d]"]
T5_METHODS = ["ARISE", "context-only", "ARISE w/o channel-aggregated expert", *EXPERT_METHODS, "Eadro", "MULAN"]
NOAUG = "ARISE w/o missing-view augmentation"


def compose(dataset, task, fold, seed, smoke, output):
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable composed scores exist")
    start, began = time.perf_counter(), q.now(); state = json.loads((q.FOLDER / "queue_state.json").read_text())
    inputs, source_hashes, masks = {}, {"exp_revision/compose_remaining.py": sha256(Path(__file__)),
        "exp_revision/score_receipts.py": sha256(ROOT / "exp_revision/score_receipts.py")}, {}
    def get(name):
        arrays, meta, folder = read_job(name, state)
        for group, destination in (("input_sha256", inputs), ("source_sha256", source_hashes), ("mask_sha256", masks)):
            destination.update(meta.get(group, {}))
        inputs.update({str((folder / file).relative_to(ROOT)): sha256(folder / file) for file in ("scores.npz", "run_meta.json")})
        return arrays, meta
    mode = "smoke" if smoke else "full"
    if task == "T6":
        data = load_fold(dataset, fold); ids = data.test_ids[:5] if smoke else data.test_ids
        base, bm = get(f"full_evidence_masks_{dataset}_fold{fold}_seed{seed}")
        variants, vm = get(f"{mode}_t3_evidence_{dataset}_fold{fold}_seed{seed}")
        all_scenes = [f"{s}_{p or 'none'}" for s, p in SCENARIOS]
        archive_axes(base, data.test_ids, all_scenes, ["ARISE", *EXPERT_METHODS])
        archive_axes(variants, ids, all_scenes, list(T3_VARIANTS))
        scenes = T6_SCENARIOS; ix = [all_scenes.index(key) for key in scenes]
        context = base["context"][ix, :len(ids)]; spec = base["specific"][ix, :len(ids)]
        scores = np.stack([base["scores"][0, ix, :len(ids)], fused(context, spec), context,
            base["scores"][1, ix, :len(ids)], base["evidence"][ix, :len(ids)], variants["scores"][3, ix]])
        methods = T6_METHODS; split = "kfold5"
    else:
        part = part_for(dataset, task, fold); ids = part["test"][:5] if smoke else part["test"]
        scenes = t5_scenarios(task); split = part["split"]
        expert, em = get(identifier(mode, dataset, task, fold, seed, "experts"))
        context, cm = get(identifier(mode, dataset, task, fold, seed, "context"))
        archive_axes(expert, ids, scenes, EXPERT_METHODS); archive_axes(context, ids, scenes)
        values = [fused(context["scores"], expert["evidence"]), context["scores"], fused(context["scores"], expert["specific"])]
        values.extend(expert["scores"])
        for method in ("Eadro", "MULAN"):
            baseline, _ = get(identifier(mode, dataset, task, fold, seed, "baseline", method)); archive_axes(baseline, ids, scenes)
            values.append(baseline["scores"])
        methods = T5_METHODS.copy()
        if task == "T5a":
            methods.append(NOAUG); values.append(fused(context["scores"], expert["noaug_evidence"]))
        scores = np.stack(values)
        path = split_file(dataset, task); inputs[str(path.relative_to(ROOT))] = sha256(path)
    if not np.isfinite(scores).all(): raise ValueError("invalid prerequisite branch scores")
    for key in scenes:
        path = ROOT / f"masks/{dataset}/{key}/seed{seed}.npz"; relative = str(path.relative_to(ROOT))
        if relative not in masks or masks[relative] != sha256(path): raise ValueError("composition missing shared mask receipt")
    check_hashes(inputs); check_hashes(source_hashes)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "scores.npz", scores=scores, method_names=methods, scenario_keys=scenes, incident_ids=ids)
    cfg = dict(task=task, dataset=dataset, fold=fold, seed=seed, smoke=smoke, methods=methods, scenarios=scenes,
        split=split, alpha=.75, component="frozen_score_composition", preregistered=False if task == "T6" else None)
    meta = config_snapshot("python -m exp_revision.compose_remaining", cfg)
    meta.update(status="completed_score_composition", errors=[], start_time=began, end_time=q.now(), elapsed_seconds=time.perf_counter() - start,
        input_sha256=inputs, source_sha256=source_hashes, mask_sha256=masks, scores_sha256=sha256(output / "scores.npz"),
        completion_protocol="original per-scenario fusion of verified frozen scores; no training or held-out label reads")
    json_dump(output / "run_meta.json", meta)


def create(task, smoke=False):
    paths = [ROOT / "manifests/remaining/06_full_evidence_masks.json"]
    if task == "T6": paths.append(ROOT / f"manifests/remaining/{'07_smoke' if smoke else '08_full'}_t3_evidence.json")
    else: paths.append(ROOT / f"manifests/remaining/{'18_smoke' if smoke else '20_full'}_generalization_models.json")
    state = json.loads((q.FOLDER / "queue_state.json").read_text())
    # Phase-level dependency gate: no composition is registered ahead of its inputs.
    for path in paths:
        for job in json.loads(path.read_text())["jobs"]:
            if state["jobs"][job["id"]]["status"] != "completed": raise ValueError("composition prerequisites incomplete")
    if not smoke:
        smoke_path = ROOT / f"manifests/remaining/{22 if task == 'T6' else 24}_smoke_{'t6_reuse' if task == 'T6' else 't5_composition'}.json"
        for job in json.loads(smoke_path.read_text())["jobs"]:
            row = state["jobs"][job["id"]]
            if row["status"] != "completed" or not q.verified(ROOT / row["output"], row["spec"]): raise ValueError("composition smoke not verified")
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    for path in paths: sources.update(json.loads(path.read_text())["source_sha256"])
    for name in ("exp_revision/compose_remaining.py", "exp_revision/score_receipts.py", "exp_revision/generalization_data.py",
                 "exp_revision/generalization_model_manifest.py", "exp_revision/generalization_models.py"):
        sources[name] = sha256(ROOT / name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(); jobs = []
    for dataset in ("tt", "aiops"):
        for current in (["T6"] if task == "T6" else ["T5a", "T5b"]):
            fold_count = 5 if current == "T6" or current == "T5b" or dataset == "tt" else 9
            for fold in ([0] if smoke else range(fold_count)):
                if current == "T6":
                    data = load_fold(dataset, fold); ids = data.test_ids[:5] if smoke else data.test_ids; services = data.services
                    scenes, methods, split = T6_SCENARIOS, T6_METHODS, "kfold5"
                else:
                    part = part_for(dataset, current, fold); ids = part["test"][:5] if smoke else part["test"]
                    services = json.loads((ROOT / f"data/care/{'trainticket-prepared' if dataset == 'tt' else 'aiops2022-paper128-prepared'}/metadata.json").read_text())["services"]
                    scenes, methods, split = t5_scenarios(current), T5_METHODS + ([NOAUG] if current == "T5a" else []), part["split"]
                for seed in ([42] if smoke else SEEDS):
                    mode = "smoke" if smoke else "full"
                    config = dict(task=current, dataset=dataset, fold=fold, seed=seed, smoke=smoke, methods=methods, scenarios=scenes,
                        split=split, alpha=.75, component="frozen_score_composition", preregistered=False if current == "T6" else None)
                    jobs.append(dict(id=f"{mode}_{current.lower()}_composition_{dataset}_fold{fold}_seed{seed}", source_commit=commit,
                        resource="cpu", cores=1, timeout_seconds=300, max_attempts=2,
                        command=["-m", "exp_revision.compose_remaining", "--dataset", dataset, "--task", current, "--fold", str(fold), "--seed", str(seed)] + (["--smoke"] if smoke else []),
                        expected_config=config, expected_status=["completed_score_composition"], test_ids=ids,
                        score_shape=[len(methods), len(scenes), len(ids), len(services)]))
    number = (22 if smoke else 23) if task == "T6" else (24 if smoke else 25)
    name = "t6_reuse" if task == "T6" else "t5_composition"
    path = ROOT / f"manifests/remaining/{number}_{'smoke' if smoke else 'full'}_{name}.json"
    manifest = dict(backend="local", phase=f"{'smoke' if smoke else 'full'}_{name}", source_commit=commit,
        precharged_hours=q.PRECHARGED, source_sha256=sources, jobs=jobs)
    if path.exists(): raise FileExistsError("immutable composition manifest exists")
    json_dump(path, manifest); return path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--create", action="store_true"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--dataset", choices=("tt", "aiops")); p.add_argument("--task", choices=("T6", "T5", "T5a", "T5b"), required=True)
    p.add_argument("--fold", type=int); p.add_argument("--seed", type=int); p.add_argument("--output", type=Path); a = p.parse_args()
    if a.create: print(create(a.task, a.smoke))
    else: compose(a.dataset, a.task, a.fold, a.seed, a.smoke, a.output)


if __name__ == "__main__": main()
