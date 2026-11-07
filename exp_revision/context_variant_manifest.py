"""CPU sanity, local CUDA pilots, then the complete context ablation matrix."""
import argparse
import json
import subprocess

from .common import ROOT, SEEDS, json_dump, sha256
from .fold_data import load_fold
from .remaining_queue import PRECHARGED, verified
from .train_context_variant import VARIANTS
from .train_variant_evidence import T6_SCENARIOS


def create(smoke=False, pilot=False):
    if smoke and pilot: raise ValueError("select smoke or pilot")
    parent = ROOT / "manifests/remaining/06_full_evidence_masks.json"
    sources = dict(json.loads(parent.read_text())["source_sha256"])
    for name in ("exp_revision/context_variant_manifest.py", "exp_revision/train_context_variant.py",
                 "exp_revision/train_context.py", "exp_revision/train_variant_evidence.py",
                 "scripts/care/train_base.py", "scripts/care/train_robust_base.py",
                 "configs/care/review-response-v1.json", "splits/tt_inner_validation_kfold5.json"):
        sources[name] = sha256(ROOT / name)
    sources[str(parent.relative_to(ROOT))] = sha256(parent)
    smoke_path = ROOT / "manifests/remaining/11_smoke_context_variants.json"
    commit = (json.loads(smoke_path.read_text())["source_commit"] if smoke_path.exists() else
              subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    for dataset in ("tt", "aiops"):
        for fold in range(5):
            for seed in SEEDS:
                row = state["jobs"][f"full_evidence_masks_{dataset}_fold{fold}_seed{seed}"]
                if row["status"] != "completed": raise ValueError("T2 evidence prerequisites incomplete")
                for name in ("scores.npz", "run_meta.json"):
                    path = ROOT / row["output"] / name; sources[str(path.relative_to(ROOT))] = sha256(path)
    def check_phase(name):
        for dataset in ("tt", "aiops"):
            for variant in VARIANTS:
                key = f"{name}_context_variant_{dataset}_{variant}_fold0_seed42"
                row = state["jobs"][key]
                if row["status"] != "completed" or not verified(ROOT / row["output"], row["spec"]):
                    raise ValueError(f"context {name} prerequisite incomplete")
    if not smoke: check_phase("smoke")
    if not smoke and not pilot: check_phase("full")
    jobs = []
    for dataset in ("tt", "aiops"):
        for fold in ([0] if smoke or pilot else range(5)):
            data = load_fold(dataset, fold); ids = data.test_ids[:5] if smoke else data.test_ids
            for seed in ([42] if smoke or pilot else SEEDS):
                for variant, method in VARIANTS.items():
                    device = "cuda" if dataset == "tt" and not smoke else "cpu"
                    key = f"{'smoke' if smoke else 'full'}_context_variant_{dataset}_{variant}_fold{fold}_seed{seed}"
                    command = ["-m", "exp_revision.train_context_variant", "--dataset", dataset, "--fold", str(fold),
                               "--seed", str(seed), "--variant", variant]
                    if smoke: command.append("--smoke")
                    jobs.append(dict(id=key, source_commit=commit, resource="gpu" if device == "cuda" else "cpu", cores=4,
                        timeout_seconds=600 if smoke else 1800, max_attempts=2, command=command,
                        expected_status=["completed_context_variant"],
                        expected_config=dict(task="T6", method=method, variant=variant, dataset=dataset, fold=fold,
                            seed=seed, smoke=smoke, device=device, scenarios=T6_SCENARIOS, alpha=.75, split="kfold5", preregistered=False),
                        test_ids=ids, score_shape=[len(T6_SCENARIOS), len(ids), len(data.services)]))
    mode = "smoke" if smoke else "pilot" if pilot else "full"
    manifest = dict(backend="local", phase=f"{mode}_context_variants", source_commit=commit,
                    precharged_hours=PRECHARGED, source_sha256=sources, jobs=jobs)
    number = 11 if smoke else 12 if pilot else 13
    path = ROOT / f"manifests/remaining/{number:02d}_{mode}_context_variants.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest: raise FileExistsError("immutable context variant manifest differs")
    else: json_dump(path, manifest)
    return path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true"); p.add_argument("--pilot", action="store_true")
    a = p.parse_args(); print(create(a.smoke, a.pilot))


if __name__ == "__main__": main()
