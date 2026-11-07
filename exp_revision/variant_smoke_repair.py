"""Split two timed TT smoke suites into bounded, scientifically identical jobs.

This one-off recovery keeps both interrupted attempts. Independent variants
use the frozen trainer, seed and cohort masks; their verified results are
assembled into the original smoke matrix without a third training retry.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from . import remaining_queue as q
from . import train_variant_evidence as training
from .common import ROOT, config_snapshot, json_dump, sha256
from .disk_guard_local import DiskGuard

PREFIX = "split_smoke_tt_"
CPU_SLOTS = 4
TARGETS = {task: f"smoke_{task.lower()}_evidence_tt_fold0_seed42" for task in ("T3", "T6")}


def selected_variant(task, index, output):
    name = "T3_VARIANTS" if task == "T3" else "T6_VARIANTS"
    settings = getattr(training, name)
    if index is None or not 0 <= index < len(settings): raise ValueError("invalid variant index")
    method = list(settings)[index]
    setattr(training, name, {method: settings[method]})
    try:
        return training.train("tt", 0, 42, task, output, smoke=True)
    finally:
        setattr(training, name, settings)


def create():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources, jobs = {}, []
    for task in ("T3", "T6"):
        parent_path = ROOT / f"manifests/remaining/{'07_smoke_t3_evidence' if task == 'T3' else '09_smoke_t6_evidence'}.json"
        parent = json.loads(parent_path.read_text()); sources.update(parent["source_sha256"])
        sources[str(parent_path.relative_to(ROOT))] = sha256(parent_path)
        original = next(job for job in parent["jobs"] if job["id"] == TARGETS[task])
        settings = training.T3_VARIANTS if task == "T3" else training.T6_VARIANTS
        for index, (method, setting) in enumerate(settings.items()):
            cfg = {**original["expected_config"], "methods": [method], "variants": {method: setting}}
            jobs.append({**original, "id": f"{PREFIX}{task}_variant{index}", "source_commit": commit,
                "command": ["-m", "exp_revision.variant_smoke_repair", "--task", task, "--variant-index", str(index)],
                "expected_config": cfg, "score_shape": [1, *original["score_shape"][1:]]})
        jobs.append({**original, "id": f"{PREFIX}{task}_assembled", "source_commit": commit, "cores": 1,
            "timeout_seconds": 120, "command": ["-m", "exp_revision.variant_smoke_repair", "--task", task, "--compose"]})
    sources["exp_revision/variant_smoke_repair.py"] = sha256(Path(__file__))
    manifest = dict(backend="local", phase="split_TT_variant_smoke_recovery", source_commit=commit,
                    precharged_hours=q.PRECHARGED, source_sha256=sources, jobs=jobs,
                    reason="Combined TT suites exceeded 600s; preserve attempts and verify each unchanged variant separately")
    path = ROOT / "manifests/remaining/14_split_tt_variant_smoke.json"
    if path.exists(): raise FileExistsError("immutable split-smoke manifest exists")
    json_dump(path, manifest); return path


def verified_child(folder, row, method):
    if row["status"] != "completed" or not q.verified(folder, row["spec"]):
        raise ValueError("individual smoke incomplete")
    meta = json.loads((folder / "run_meta.json").read_text())
    for group in ("input_sha256", "source_sha256", "mask_sha256", "model_sha256"):
        if not meta.get(group): raise ValueError(f"child missing {group}")
        for name, digest in meta[group].items():
            if sha256(ROOT / name) != digest: raise ValueError(f"child artifact changed: {name}")
    for name, field in (("training_masks.npz", "training_masks_sha256"),
                        ("training_mask_cohorts.json", "training_mask_cohorts_sha256")):
        if sha256(folder / name) != meta[field]: raise ValueError(f"child artifact changed: {name}")
    for name in ("specific.joblib", "generic.joblib"):
        relative = str((folder / "variant0" / name).relative_to(ROOT))
        if relative not in meta["model_sha256"]: raise ValueError("child model receipt incomplete")
    if (list(meta["beta_selections"]) != [method] or
            json.loads((folder / "variant0/inner_beta.json").read_text()) != meta["beta_selections"][method]):
        raise ValueError("child beta selection differs")
    return meta


def compose(task, output):
    start = q.now(); timer = time.perf_counter()
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable assembled smoke exists")
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((q.FOLDER / "queue_state.json").read_text())
    original = state["jobs"][TARGETS[task]]["spec"]
    settings = training.T3_VARIANTS if task == "T3" else training.T6_VARIANTS
    scores, selections, inputs, sources, models, mask_hashes, timings = [], {}, {}, {}, {}, {}, {}
    masks, cohorts, first = None, None, None
    for index, method in enumerate(settings):
        row = state["jobs"][f"{PREFIX}{task}_variant{index}"]
        folder = ROOT / row["output"]; meta = verified_child(folder, row, method)
        with np.load(folder / "scores.npz") as archive:
            if archive["method_names"].tolist() != [method]: raise ValueError("variant order differs")
            if archive["incident_ids"].tolist() != original["test_ids"]: raise ValueError("test identity differs")
            if archive["scenario_keys"].tolist() != original["expected_config"]["scenarios"]: raise ValueError("scenarios differ")
            scores.append(archive["scores"][0].copy())
        with np.load(folder / "training_masks.npz") as archive: current = {key: archive[key].copy() for key in archive.files}
        current_cohorts = json.loads((folder / "training_mask_cohorts.json").read_text())
        if masks is None: masks, cohorts, first = current, current_cohorts, folder
        elif (cohorts != current_cohorts or masks.keys() != current.keys()
              or any(not np.array_equal(masks[key], current[key]) for key in masks)):
            raise ValueError("independent variants did not use identical original cohort masks")
        directory = output / f"variant{index}"; directory.mkdir()
        for name in ("specific.joblib", "generic.joblib", "inner_beta.json"):
            shutil.copyfile(folder / "variant0" / name, directory / name)
            models[str((directory / name).relative_to(ROOT))] = sha256(directory / name)
        selections.update(meta["beta_selections"])
        inputs.update(meta["input_sha256"]); sources.update(meta["source_sha256"])
        mask_hashes.update(meta["mask_sha256"]); timings[method] = meta["elapsed_seconds"]
        for name in ("scores.npz", "run_meta.json", "training_masks.npz", "training_mask_cohorts.json"):
            inputs[str((folder / name).relative_to(ROOT))] = sha256(folder / name)
    shutil.copyfile(first / "training_masks.npz", output / "training_masks.npz")
    shutil.copyfile(first / "training_mask_cohorts.json", output / "training_mask_cohorts.json")
    np.savez_compressed(output / "scores.npz", scores=np.stack(scores), method_names=list(settings),
                        incident_ids=original["test_ids"], scenario_keys=original["expected_config"]["scenarios"])
    meta = config_snapshot("python -m exp_revision.variant_smoke_repair --compose", original["expected_config"])
    meta.update(status="completed_variant_evidence", start_time=start, end_time=q.now(),
        elapsed_seconds=time.perf_counter() - timer, errors=[], beta_selections=selections,
        input_sha256=inputs, source_sha256={**sources, "exp_revision/variant_smoke_repair.py": sha256(Path(__file__))},
        model_sha256=models, mask_sha256=mask_hashes, scores_sha256=sha256(output / "scores.npz"),
        training_masks_sha256=sha256(output / "training_masks.npz"), training_mask_cohorts_sha256=sha256(output / "training_mask_cohorts.json"),
        completion_protocol="assembled verified per-variant smoke; no third original training attempt",
        child_elapsed_seconds=timings, training_elapsed_seconds=sum(timings.values()),
        recovered_job=TARGETS[task], preserved_attempts=state["jobs"][TARGETS[task]]["attempts"])
    json_dump(output / "run_meta.json", meta)


def repair():
    guard = DiskGuard(ROOT)
    with (q.FOLDER / ".queue.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return {"status": "scheduler_busy"}
        path = q.FOLDER / "queue_state.json"; state = json.loads(path.read_text())
        q.load_manifests(state, {})
        processes = {}; stop = False
        def stopping(*_):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGTERM, stopping); signal.signal(signal.SIGINT, stopping)
        for row in state["jobs"].values():
            if row["status"] == "running" and q.alive(row["attempts"][-1]):
                attempt = row["attempts"][-1]
                guard.reserve(attempt["key"], ROOT / attempt["output"], ROOT / attempt["log"], child_pid=attempt["pid"])
        while not stop:
            storage = guard.poll()
            for key, row in state["jobs"].items():
                if row["status"] == "running" and not q.alive(row["attempts"][-1]):
                    q.finish(row["spec"], row, row["attempts"][-1], processes.pop(key, None), guard)
            active = Counter(row["spec"]["resource"] for row in state["jobs"].values() if row["status"] == "running")
            selected = {key: row for key, row in state["jobs"].items() if key.startswith(PREFIX)}
            if any(row["status"] == "stuck" for row in selected.values()):
                state["status"] = "stuck"; json_dump(path, state); return {"status": "recovery_failed"}
            candidates = {**selected, **{key: row for key, row in state["jobs"].items()
                if key.startswith("full_context_variant_") and row["spec"]["resource"] == "gpu"}}
            limited = False
            for key, row in candidates.items():
                job = row["spec"]; resource = job["resource"]
                slots = CPU_SLOTS if resource == "cpu" else q.SLOTS[resource]
                if (row["status"] != "pending" or time.time() < row.get("retry_after", 0)
                        or active[resource] >= slots or not storage["safe_to_launch"]): continue
                if key.endswith("_assembled"):
                    task = row["spec"]["expected_config"]["task"]
                    count = len(training.T3_VARIANTS if task == "T3" else training.T6_VARIANTS)
                    if any(selected[f"{PREFIX}{task}_variant{i}"]["status"] != "completed" for i in range(count)): continue
                bound = (job["timeout_seconds"] + 10) * (job["cores"] if resource == "cpu" else 1) / 3600
                if q.charged(state, resource, True) + bound > (q.CPU_BUDGET if resource == "cpu" else q.GPU_BUDGET):
                    limited = True; continue
                if resource == "gpu":
                    free = subprocess.check_output(["/usr/bin/nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
                    if int(free.strip()) < 8192: continue
                try: processes[key] = q.start(job, row, state, guard)
                except q.InsufficientSpace:
                    storage = guard.poll(); break
                active[resource] += 1; storage = guard.poll()
            for task, target in TARGETS.items():
                assembled = selected[f"{PREFIX}{task}_assembled"]
                if assembled["status"] == "completed" and state["jobs"][target]["status"] != "completed":
                    folder = ROOT / assembled["output"]
                    if not q.verified(folder, state["jobs"][target]["spec"]): raise ValueError("assembled original smoke fails verification")
                    state["jobs"][target].update(status="completed", output=assembled["output"],
                        completion_job=f"{PREFIX}{task}_assembled", completion_protocol="verified split-smoke composition")
            state.update(updated=q.now(), status="running", storage=storage,
                         counts=dict(Counter(row["status"] for row in state["jobs"].values())),
                         charged_hours={r: q.charged(state, r) for r in q.SLOTS},
                         admission_hours={r: q.charged(state, r, True) for r in q.SLOTS})
            json_dump(path, state)
            if all(row["status"] == "completed" for row in selected.values()):
                receipt = {"time": q.now(), "status": "completed", "targets": TARGETS,
                           "completion_jobs": [f"{PREFIX}{task}_assembled" for task in TARGETS],
                           "source_sha256": sha256(Path(__file__)), "charged_hours": state["charged_hours"]}
                json_dump(ROOT / "results_v2/variant_smoke_repair/receipt.json", receipt)
                break
            if not sum(active.values()) and (limited or not storage["safe_to_launch"]):
                state["status"] = "budget_limited" if limited else "storage_limited"
                json_dump(path, state); return {"status": state["status"]}
            time.sleep(2)
        if stop: return {"status": "interrupted", "children_preserved": True}
    return q.check()


def check():
    """Resume this actual recovery before handing monitoring back to q.check."""
    manifest = ROOT / "manifests/remaining/14_split_tt_variant_smoke.json"
    receipt = ROOT / "results_v2/variant_smoke_repair/receipt.json"
    if not manifest.exists() or receipt.exists(): return q.check()
    storage = DiskGuard(ROOT).poll(); busy = False
    with (q.FOLDER / ".queue.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: busy = True
    state = json.loads((q.FOLDER / "queue_state.json").read_text())
    failed = any(row["status"] == "stuck" for key, row in state["jobs"].items() if key.startswith(PREFIX))
    action = "none"
    if not busy and storage["safe_to_launch"] and not failed and state["status"] != "budget_limited":
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        log = ROOT / f"logs/variant_smoke_repair_{stamp}.log"
        with log.open("x") as stream:
            child = subprocess.Popen([str(ROOT / "cache/exp_runtime/venv/bin/python"), "-m", "exp_revision.variant_smoke_repair", "--repair"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        action = "resume_variant_smoke_repair"
    result = dict(time=q.now(), backend="local", status=state["status"], scheduler_alive=busy,
        jobs=dict(Counter(row["status"] for row in state["jobs"].values())), action=action, storage=storage)
    if action != "none": result.update(pid=child.pid, log=str(log.relative_to(ROOT)))
    json_dump(q.FOLDER / "monitor.json", result)
    with (q.FOLDER / "monitor_history.jsonl").open("a") as history: history.write(json.dumps(result) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--create", action="store_true"); p.add_argument("--repair", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--task", choices=("T3", "T6")); p.add_argument("--variant-index", type=int); p.add_argument("--compose", action="store_true")
    p.add_argument("--output", type=Path); a = p.parse_args()
    if a.create: print(create())
    elif a.repair: print(json.dumps(repair()))
    elif a.check: print(json.dumps(check()))
    elif a.compose: compose(a.task, a.output)
    else: selected_variant(a.task, a.variant_index, a.output)


if __name__ == "__main__": main()
