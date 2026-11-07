"""Smoke then full raw preprocessing of all 24 T5 train/test cohorts."""
import argparse
import json
import subprocess

from .common import ROOT, json_dump, sha256
from .generalization_data import part_for, split_file, training_file
from .remaining_queue import PRECHARGED, verified


def create(smoke=False):
    parent = ROOT / "manifests/remaining/04_full_span_cache.json"
    sources = dict(json.loads(parent.read_text())["source_sha256"])
    for name in ("exp_revision/generalization_data.py", "exp_revision/generalization_prepare_manifest.py",
                 "exp_revision/preprocess_tt.py", "exp_revision/preprocess_aiops.py", "exp_revision/fold_data.py"):
        sources[name] = sha256(ROOT / name)
    sources[str(parent.relative_to(ROOT))] = sha256(parent)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    jobs = []
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    for dataset in ("tt", "aiops"):
        metadata = json.loads((ROOT / f"data/care/{'trainticket-prepared' if dataset == 'tt' else 'aiops2022-paper128-prepared'}/metadata.json").read_text())
        for task in ("T5a", "T5b"):
            path = split_file(dataset, task); parts = json.loads(path.read_text()); sources[str(path.relative_to(ROOT))] = sha256(path)
            if not smoke:
                key = f"smoke_generalization_prepare_{dataset}_{task}_fold0"
                row = state["jobs"][key]
                if row["status"] != "completed" or not verified(ROOT / row["output"], row["spec"]): raise ValueError("T5 preprocessing smoke incomplete")
            for fold in ([0] if smoke else range(len(parts))):
                part = part_for(dataset, task, fold); ids = part["test"][:5] if smoke else part["test"]
                labels = training_file(dataset, task, fold); sources[str(labels.relative_to(ROOT))] = sha256(labels)
                command = ["-m", "exp_revision.generalization_data", "--prepare", "--dataset", dataset, "--task", task, "--fold", str(fold)]
                if smoke: command.append("--smoke")
                jobs.append(dict(id=f"{'smoke' if smoke else 'full'}_generalization_prepare_{dataset}_{task}_fold{fold}", source_commit=commit,
                    resource="cpu", cores=1, timeout_seconds=600 if smoke else 1200, max_attempts=2, command=command,
                    expected_status=["completed_generalization_preprocessing"],
                    expected_config=dict(task=task, component="train_only_preprocessing", dataset=dataset, fold=fold, smoke=smoke, split=part["split"]),
                    test_ids=ids, score_shape=[1, len(ids), len(metadata["services"])]))
    manifest = dict(backend="local", phase="smoke_T5_preprocessing" if smoke else "full_T5_preprocessing", source_commit=commit,
        precharged_hours=PRECHARGED, source_sha256=sources, jobs=jobs)
    path = ROOT / f"manifests/remaining/{'15_smoke' if smoke else '16_full'}_generalization_preprocessing.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest: raise FileExistsError("immutable T5 preprocessing manifest differs")
    else: json_dump(path, manifest)
    return path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true"); a = p.parse_args(); print(create(a.smoke))


if __name__ == "__main__": main()
