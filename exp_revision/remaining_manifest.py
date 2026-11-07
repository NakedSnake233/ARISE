"""Freeze reviewed baseline/context-missingness smoke and full batches."""
import argparse
import json
import subprocess
from pathlib import Path
from .common import ROOT, SEEDS, json_dump, sha256
from .baseline_fold import METHODS, CONFIG
from .fold_data import load_fold
from .missingness import SCENARIOS
from .remaining_queue import PRECHARGED, verified


def create(smoke=False):
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    jobs=[]
    paths=[ROOT/name for name in (
        "exp_revision/remaining_queue.py","exp_revision/remaining_manifest.py",
        "exp_revision/baseline_fold.py","exp_revision/baseline_settings.json",
        "exp_revision/context_masks_local.py","exp_revision/infer_context_masks.py",
        "exp_revision/common.py","exp_revision/metrics.py","exp_revision/missingness.py",
        "exp_revision/fold_data.py","exp_revision/disk_guard_local.py","exp_revision/disk_guard.py",
        "scripts/care/fse2027_baseline_common.py","scripts/care/fse2027_classical_baselines.py",
        "scripts/care/fse2027_neural_baselines.py","scripts/care/train_base.py",
        "splits/tt_inner_validation_kfold5.json")]
    settings=json.loads(CONFIG.read_text())
    for dataset in ("tt","aiops"):
        paths.append(ROOT/f"splits/{dataset}_kfold5.json")
        for entry in settings[dataset].values(): paths.append(ROOT/entry["source"])
        for fold in range(5):
            paths.extend([ROOT/f"cache/fold_preprocessing/{dataset}/fold{fold}/{name}" for name in ("features.npz","metadata.json")])
            paths.append(ROOT/f"splits/{dataset}_training_labels/fold_{fold}.json")
        paths.extend(ROOT/f"masks/{dataset}/{s}_{p or 'none'}/seed{seed}.npz" for s,p in SCENARIOS for seed in SEEDS)
        for fold in ([0] if smoke else range(5)):
            data=load_fold(dataset,fold)
            ids=data.test_ids[:5] if smoke else data.test_ids
            for seed in ([42] if smoke else SEEDS):
                for method in METHODS:
                    key=f"{'smoke' if smoke else 'full'}_baseline_{dataset}_{method.replace('+','_')}_fold{fold}_seed{seed}"
                    command=["-m","exp_revision.baseline_fold","--dataset",dataset,"--fold",str(fold),"--seed",str(seed),"--method",method]
                    if smoke: command.append("--smoke")
                    jobs.append({"id":key,"command":command,"source_commit":commit,
                                 "resource":"cpu","cores":1,"timeout_seconds":600,"max_attempts":2,
                                 "expected_status":["completed_scores"],
                                 "expected_config":{"dataset":dataset,"fold":fold,"seed":seed,"method":method,"smoke":smoke,"device":"cpu"},
                                 "test_ids":ids,"score_shape":[len(SCENARIOS),len(ids),len(data.services)]})
                if dataset=="tt":
                    key=f"{'smoke' if smoke else 'full'}_context_masks_{dataset}_fold{fold}_seed{seed}"
                    command=["-m","exp_revision.context_masks_local","--dataset",dataset,"--fold",str(fold),"--seed",str(seed)]
                    if smoke: command.append("--smoke")
                    jobs.append({"id":key,"command":command,"source_commit":commit,
                                 "resource":"gpu","cores":4,"timeout_seconds":600,"max_attempts":2,
                                 "expected_status":["partial_T2_context_only"],
                                 "expected_config":{"dataset":dataset,"fold":fold,"seed":seed,"smoke":smoke,"device":"cuda"},
                                 "test_ids":ids,"score_shape":[len(SCENARIOS),len(ids),len(data.services)]})
    context=json.loads((ROOT/"results_v2/T1_context_local_full/tt/queue_state.json").read_text())
    paths.append(ROOT/"results_v2/T1_context_local_full/tt/queue_state.json")
    for row in context["jobs"].values():
        paths.extend(ROOT/row["output"]/name for name in ("run_meta.json","model.pt","scores.npz"))
    if not smoke:
        state=json.loads((ROOT/"results_v2/remaining_queue/queue_state.json").read_text())
        smoke_jobs=[row for key,row in state["jobs"].items() if key.startswith("smoke_")]
        if len(smoke_jobs)!=23 or any(row["status"]!="completed" or not verified(ROOT/row["output"],row["spec"]) for row in smoke_jobs):
            raise ValueError("all23 baseline/context smoke jobs must pass before the full matrix")
    manifest={"backend":"local","phase":"smoke" if smoke else "full_baselines_T1_T2",
              "source_commit":commit,"precharged_hours":PRECHARGED,
              "source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in dict.fromkeys(paths)},"jobs":jobs}
    path=ROOT/f"manifests/remaining/{'01_smoke_baselines' if smoke else '02_full_baselines'}.json"
    if path.exists():
        if json.loads(path.read_text())!=manifest: raise FileExistsError("immutable batch manifest exists")
    else: json_dump(path,manifest)
    return path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    print(create(a.smoke))


if __name__=="__main__": main()
