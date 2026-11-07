"""Admit independent RE2 pilots and successful official dataset/method groups.

The frozen all-or-nothing official manifest is intentionally not used after
CIRCA/AIOps failed. Each successful pair has its own smoke gate and manifest;
failed pairs stay visible and are never relabeled complete. Dataset pilots can
use available fold-0 preprocessing without waiting for another system's logs.
"""
from __future__ import annotations
import json
import subprocess

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q
from . import re2_data as data
from . import t8_manifest as original
from . import re2_manifest
from .extended_queue import charged, deadline, scientific_blocks, in_blocked_scope


def state(): return json.loads((q.FOLDER / "queue_state.json").read_text())


def commit(): return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def verified_job(name, current=None):
    current = state() if current is None else current
    row = current["jobs"].get(name)
    return bool(row and row["status"] == "completed" and q.verified(ROOT / row["output"], row["spec"]))


def pending_forecasts(current):
    """Reserve forecasts from every admitted full extension, deduplicating IDs."""
    totals = dict(cpu=0., gpu=0.); seen = set()
    mapping = {name: recipe["recovery_job"] for path in q.MANIFESTS.glob("*.json")
        for name, recipe in json.loads(path.read_text()).get("recovery_recipes", {}).items()}
    blocked = scientific_blocks(current)
    for path in sorted(q.MANIFESTS.glob("*.json")):
        manifest = json.loads(path.read_text()); forecast = manifest.get("forecast")
        details = {(r["dataset"], r.get("method") or r.get("component")): r for r in forecast["details"]} if forecast else {}
        for job in manifest["jobs"]:
            name = job["id"]; row = current["jobs"].get(name)
            if name in seen or row and row["status"] in ("completed", "running"): continue
            seen.add(name); cfg = job["expected_config"]
            if name in mapping or row and (row["status"] == "stuck" or in_blocked_scope(row, blocked)): continue
            entry = details.get((cfg.get("dataset"), cfg.get("method") or cfg.get("component")))
            if entry is None:
                estimate = (deadline(job) + 10) * (job["cores"] if job["resource"] == "cpu" else 1) / 3600
            elif "per_job_hours" in entry: estimate = entry["per_job_hours"]
            else:
                factor = len(job["test_ids"]) if forecast["kind"] in ("official", "llm") else 1
                estimate = entry["smoke_seconds"] * job["cores"] * factor / 3600
            totals[job["resource"]] += estimate
    return totals


def admit_forecast(projection, name):
    current = state(); pending = pending_forecasts(current)
    projection["already_admitted_pending_forecast_hours"] = pending
    projection["charged_and_running_bound_hours"] = {r: charged(current, r, True) for r in q.SLOTS}
    json_dump(ROOT / f"results_v2/T8_inputs/{name}_budget_forecast.json", projection)
    for resource, proposed in projection["forecast_hours"].items():
        limit = q.CPU_BUDGET if resource == "cpu" else q.GPU_BUDGET
        if charged(current, resource, True) + pending[resource] + proposed > limit:
            raise ValueError(f"{name}: measured forecast exceeds {resource} budget; author decision required")


def write_manifest(path, phase, jobs, sources, forecast=None):
    sources["exp_revision/t8_admission.py"] = sha256(ROOT / "exp_revision/t8_admission.py")
    json_dump(path, dict(backend="local", phase=phase, source_commit=commit(), precharged_hours=q.PRECHARGED,
        source_sha256=sources, jobs=jobs, forecast=forecast))
    return path


def pilot_path(kind, dataset): return q.MANIFESTS / f"46_pilot_{kind}_{dataset}.json"


def pilot_jobs(kind, dataset):
    services = len(json.loads(data.CATALOG.read_text())[dataset]["services"])
    ids = json.loads((ROOT / f"splits/{dataset}_kfold5.json").read_text())["fold_0"]["test"]
    jobs = []
    for component in original.COMPONENTS if kind == "models" else ("BARO", "CIRCA", "RCD"):
        cfg = dict(task="T8", dataset=dataset, fold=0, smoke=True, seed=42)
        command = ["-m", "exp_revision." + ("re2_models" if kind == "models" else "official_baselines"),
            "--dataset", dataset, "--fold", "0", "--seed", "42"]
        if kind == "models":
            cfg["component"] = component; command += ["--component", component]
        else:
            cfg.update(component="official_baseline", method=component); command += ["--method", component]
        test_ids = ids[:3 if kind == "models" else 1]; shape = [12, len(test_ids), services]
        if kind == "models" and component == "evidence": shape = [5, *shape]
        jobs.append(dict(id=f"smoke_{kind}_{dataset}_{component.replace('+', 'plus')}_f0_s42",
            resource="gpu" if kind == "models" and component in ("context", "Eadro", "MULAN", "Eadro+MD", "MULAN+MD") else "cpu",
            cores=4 if kind == "models" and component == "evidence" else 1, max_attempts=2,
            timeout_seconds=1800 if component == "evidence" else 900, source_commit=commit(),
            expected_config=cfg, command=command + ["--smoke"], test_ids=test_ids, score_shape=shape,
            expected_status=["completed_re2_scores" if kind == "models" else "completed_official_scores"]))
    return jobs


def create_pilot(kind, dataset):
    path = pilot_path(kind, dataset)
    if path.exists() or not verified_job(f"re2_fold_{dataset}_f0"): return None
    if not (ROOT / "results_v2/T8_inputs/realized_rates.csv").exists(): return None
    if kind == "official" and "PASS" not in (ROOT / "review/T8_environment_followdoc.md").read_text():
        raise ValueError("documented official environment witness incomplete")
    return write_manifest(path, "pilot_" + kind + "_" + dataset, pilot_jobs(kind, dataset), original.closure(kind, [dataset]))


def create_pilot_index(kind):
    path = q.MANIFESTS / original.NAMES[kind, True, False]
    if path.exists() or not all(pilot_path(kind, ds).exists() for ds in data.DATASETS): return None
    jobs, sources = [], {}
    for ds in data.DATASETS:
        source = pilot_path(kind, ds); manifest = json.loads(source.read_text())
        jobs += manifest["jobs"]; sources.update(manifest["source_sha256"])
        sources[str(source.relative_to(ROOT))] = sha256(source)
    return write_manifest(path, "pilot_" + kind, jobs, sources)


def create_preprocessing_full():
    path = q.MANIFESTS / "45_full_re2_fold.json"
    if path.exists() or not original.complete("44_smoke_re2_fold.json"): return None
    current = state(); pilots = json.loads((q.MANIFESTS / "44_smoke_re2_fold.json").read_text())["jobs"]
    jobs = list(pilots); details = []; total = 0.
    for pilot in pilots:
        ds = pilot["expected_config"]["dataset"]; completed = current["jobs"][pilot["id"]]
        receipt = json.loads((ROOT / completed["output"] / "run_meta.json").read_text())
        completed_attempt = current["jobs"][completed.get("completion_job", pilot["id"])]["attempts"][-1]
        elapsed = max(receipt["elapsed_seconds"], completed_attempt["elapsed_seconds"])
        # Each pilot processed every outer-training/test event. The four other
        # grouped folds have almost equal cohort sizes; reserve a 25% margin.
        estimate = elapsed * 1.25 / 3600; total += 4 * estimate
        details.append(dict(dataset=ds, component="fold_preprocessing", per_job_hours=estimate, smoke_seconds=elapsed))
        parts = json.loads((ROOT / f"splits/{ds}_kfold5.json").read_text())
        for fold in range(1, 5):
            jobs.append({**pilot, "id": f"re2_fold_{ds}_f{fold}", "source_commit": commit(),
                "timeout_seconds": max(900, min(7200, int(elapsed * 2) + 60)),
                "expected_config": {**pilot["expected_config"], "fold": fold},
                "command": ["-m", "exp_revision.re2_preprocessing_runtime", "--dataset", ds, "--fold", str(fold)],
                "test_ids": parts[f"fold_{fold}"]["test"], "score_shape": [1, len(parts[f"fold_{fold}"]["test"]), pilot["score_shape"][-1]]})
    projection = dict(kind="preprocessing", forecast_hours=dict(cpu=total, gpu=0.), details=details,
        basis="full-cohort preprocessing smoke; remaining four folds per system with 25% time margin")
    admit_forecast(projection, "preprocessing")
    sources = re2_manifest.closure()
    sources["exp_revision/re2_preprocessing_runtime.py"] = sha256(ROOT / "exp_revision/re2_preprocessing_runtime.py")
    return write_manifest(path, "full_re2_fold", jobs, sources, projection)


def create_models_full():
    path = q.MANIFESTS / original.NAMES["models", False, False]
    if path.exists() or not original.complete("45_full_re2_fold.json") or not original.complete("46_smoke_re2_models.json"): return None
    admit_forecast(original.forecast("models"), "models")
    return original.create("models", False)


def create_official_full(dataset, method):
    path = q.MANIFESTS / f"50_full_official_{dataset}_{method}.json"
    if path.exists(): return None
    name = f"smoke_official_{dataset}_{method}_f0_s42"; current = state()
    if not verified_job(name, current): return None
    if dataset in data.DATASETS and not all(verified_job(f"re2_fold_{dataset}_f{f}", current) for f in range(5)): return None
    pilot = current["jobs"][name]; elapsed = sum(a["elapsed_seconds"] for a in pilot["attempts"])
    parts = json.loads((ROOT / f"splits/{dataset}_kfold5.json").read_text()); cases = sum(len(p["test"]) for p in parts.values())
    cost = elapsed * cases * 5 / 3600
    projection = dict(kind="official", forecast_hours=dict(cpu=cost, gpu=0.),
        details=[dict(dataset=dataset, method=method, smoke_seconds=elapsed, forecast_hours=cost, resource="cpu")],
        basis="actual successful same-dataset/method smoke; all held-out events over five seeds")
    admit_forecast(projection, "official_" + dataset + "_" + method)
    jobs = []
    for fold in range(5):
        for seed in range(42, 47):
            ids = parts[f"fold_{fold}"]["test"]
            jobs.append({**pilot["spec"], "id": f"full_official_{dataset}_{method}_f{fold}_s{seed}",
                "source_commit": commit(), "timeout_seconds": max(3600, int(elapsed * len(ids) * 2) + 60),
                "expected_config": {**pilot["spec"]["expected_config"], "fold": fold, "seed": seed, "smoke": False},
                "command": ["-m", "exp_revision.official_baselines", "--dataset", dataset, "--fold", str(fold), "--seed", str(seed), "--method", method],
                "test_ids": ids, "score_shape": [12, len(ids), pilot["spec"]["score_shape"][-1]]})
    return write_manifest(path, "full_official_" + dataset + "_" + method, jobs, original.closure("official", [dataset]), projection)


def create_llm(smoke=True, legacy=False):
    if not smoke:
        if not original.complete("51_smoke_llm_legacy.json") or not original.complete("52_smoke_llm_re2.json") or not original.complete("45_full_re2_fold.json"): return None
        if (q.MANIFESTS / "53_full_llm_baselines.json").exists(): return None
        admit_forecast(original.forecast("llm"), "llm")
    return original.create("llm", smoke, legacy)
