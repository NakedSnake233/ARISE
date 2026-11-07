"""Stage original T5 model smoke, local CUDA pilots, then all 480 jobs."""
import argparse
import json
import subprocess

from .common import ROOT, SEEDS, json_dump, sha256
from .generalization_data import part_for, prepared_folder, split_file, training_file
from .generalization_models import EXPERT_METHODS, scenarios
from .remaining_queue import PRECHARGED, verified


def identifier(mode, dataset, task, fold, seed, component, method=None):
    kind = method if component == "baseline" else component
    return f"{mode}_generalization_{kind}_{dataset}_{task}_fold{fold}_seed{seed}"


def create(smoke=False, pilot=False):
    if smoke and pilot: raise ValueError("select one phase")
    parent = ROOT / "manifests/remaining/06_full_evidence_masks.json"
    sources = dict(json.loads(parent.read_text())["source_sha256"])
    names = ("exp_revision/generalization_models.py", "exp_revision/generalization_model_manifest.py", "exp_revision/generalization_data.py",
        "exp_revision/train_context_variant.py", "exp_revision/train_context.py", "exp_revision/train_variant_evidence.py",
        "exp_revision/train_evidence.py", "exp_revision/baseline_fold.py", "exp_revision/baseline_settings.json", "exp_revision/splits.py",
        "scripts/care/fse2027_neural_baselines.py", "scripts/care/fse2027_baseline_common.py", "configs/care/review-response-v1.json")
    sources.update({name: sha256(ROOT / name) for name in names}); sources[str(parent.relative_to(ROOT))] = sha256(parent)
    smoke_path = ROOT / "manifests/remaining/18_smoke_generalization_models.json"
    commit = (json.loads(smoke_path.read_text())["source_commit"] if smoke_path.exists() else
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    def require(key):
        row = state["jobs"][key]
        if row["status"] != "completed" or not verified(ROOT / row["output"], row["spec"]): raise ValueError(f"T5 prerequisite incomplete: {key}")
    if not smoke:
        for dataset in ("tt", "aiops"):
            for task in ("T5a", "T5b"):
                for component, method in (("experts", None), ("context", None), ("baseline", "Eadro"), ("baseline", "MULAN")):
                    require(identifier("smoke", dataset, task, 0, 42, component, method))
    if not smoke and not pilot:
        for task in ("T5a", "T5b"): require(identifier("full", "tt", task, 0, 42, "context"))
    jobs = []; settings = json.loads((ROOT / "exp_revision/baseline_settings.json").read_text())
    for dataset in (["tt"] if pilot else ["tt", "aiops"]):
        services = json.loads((ROOT / f"data/care/{'trainticket-prepared' if dataset == 'tt' else 'aiops2022-paper128-prepared'}/metadata.json").read_text())["services"]
        for task in ("T5a", "T5b"):
            path = split_file(dataset, task); parts = json.loads(path.read_text()); sources[str(path.relative_to(ROOT))] = sha256(path)
            for fold in ([0] if smoke or pilot else range(len(parts))):
                part = part_for(dataset, task, fold)
                folder = prepared_folder(dataset, task, fold, smoke)
                for name in ("features.npz", "metadata.json", "run_meta.json"):
                    path = folder / name; sources[str(path.relative_to(ROOT))] = sha256(path)
                path = training_file(dataset, task, fold); sources[str(path.relative_to(ROOT))] = sha256(path)
                ids = part["test"][:5] if smoke else part["test"]
                for seed in ([42] if smoke or pilot else SEEDS):
                    components = [("context", None)] if pilot else [("experts", None), ("context", None), ("baseline", "Eadro"), ("baseline", "MULAN")]
                    for component, method in components:
                        config = dict(task=task, dataset=dataset, fold=fold, seed=seed, smoke=smoke, component=component,
                            split=part["split"], scenarios=scenarios(task), alpha=.75)
                        device = "cuda" if component == "context" and dataset == "tt" and not smoke else "cpu"
                        shape = [len(scenarios(task)), len(ids), len(services)]
                        if component == "experts":
                            config.update(methods=EXPERT_METHODS, training_protocol="unchanged T1 cached seven-view augmentation", noaug=task == "T5a")
                            shape = [len(EXPERT_METHODS), *shape]
                        else:
                            config.update(method=method or "context-only", device=device)
                            if component == "baseline": config["hyperparameters"] = settings[dataset][method]["configuration"]
                        command = ["-m", "exp_revision.generalization_models", "--dataset", dataset, "--task", task, "--fold", str(fold),
                            "--seed", str(seed), "--component", component]
                        if method: command += ["--method", method]
                        if smoke: command.append("--smoke")
                        jobs.append(dict(id=identifier("smoke" if smoke else "full", dataset, task, fold, seed, component, method),
                            source_commit=commit, resource="gpu" if device == "cuda" else "cpu", cores=1 if component == "baseline" else 4,
                            timeout_seconds=600 if smoke or component == "baseline" or component == "context" and dataset == "aiops" else 1800 if component == "context" else 1200,
                            max_attempts=2, command=command, expected_status=[f"completed_generalization_{component}"],
                            expected_config=config, test_ids=ids, score_shape=shape))
    mode = "smoke" if smoke else "pilot" if pilot else "full"
    manifest = dict(backend="local", phase=f"{mode}_T5_models", source_commit=commit, precharged_hours=PRECHARGED, source_sha256=sources, jobs=jobs)
    path = ROOT / f"manifests/remaining/{18 if smoke else 19 if pilot else 20:02d}_{mode}_generalization_models.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest: raise FileExistsError("immutable T5 model manifest differs")
    else: json_dump(path, manifest)
    return path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true"); p.add_argument("--pilot", action="store_true")
    a = p.parse_args(); print(create(a.smoke, a.pilot))


if __name__ == "__main__": main()
