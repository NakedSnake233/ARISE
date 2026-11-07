"""Continue independent local phases while preserving scoped scientific failures.

The real CIRCA/AIOps failure must not strand RE2 preprocessing and other methods.
Ordinary code failures still stop admission. All attempts, costs and disk leases
use the original queue contract, with no experiment result relabeled successful.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q
from .disk_guard_local import DiskGuard, InsufficientSpace, guarded_command

CPU_SLOTS = 4


def deadline(job, attempt=None):
    if attempt is not None: return attempt.get("runtime_timeout_seconds", job["timeout_seconds"])
    cfg = job["expected_config"]
    if not cfg.get("smoke") and cfg.get("task") in ("T3", "T6") and "variants" in cfg:
        return max(2400, job["timeout_seconds"])
    return job["timeout_seconds"]


def charged(state, resource, include_running=False):
    amount = q.PRECHARGED[resource]
    for row in state["jobs"].values():
        job = row["spec"]
        if job["resource"] != resource: continue
        factor = job["cores"] if resource == "cpu" else 1
        for a in row["attempts"]:
            if "elapsed_seconds" in a: amount += a["elapsed_seconds"] * factor / 3600
            elif include_running: amount += (deadline(job, a) + 10) * factor / 3600
    return amount


def start(job, row, state, guard):
    number = len(row["attempts"]) + 1; key = f"remaining_{job['id']}_attempt{number}"
    output = ROOT / "results_v2/remaining_runs" / job["id"] / f"attempt{number}"
    log = ROOT / "logs" / f"{key}.log"
    if output.exists() or log.exists(): raise FileExistsError("immutable attempt path occupied")
    env = {**os.environ, "ARISE_SOURCE_COMMIT": job["source_commit"], "OMP_NUM_THREADS": str(job["cores"]),
        "MKL_NUM_THREADS": str(job["cores"]), "OPENBLAS_NUM_THREADS": str(job["cores"]), "PYTHONUNBUFFERED": "1"}
    env.update(guard.reserve(key, output, log))
    command = [str(ROOT / "cache/exp_runtime/venv/bin/python"), *job["command"], "--output", str(output)]
    effective = deadline(job); launched = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", str(effective) + "s", *guarded_command(ROOT, key, command)]
    attempt = dict(number=number, key=key, pid=None, identity=None, start_time=q.now(), start_epoch=time.time(),
        output=str(output.relative_to(ROOT)), log=str(log.relative_to(ROOT)), status="launching",
        runtime_timeout_seconds=effective, runtime_command=launched, scheduler_source_sha256=sha256(Path(__file__)),
        runtime_reason="evidence suites exceeded original deadline under concurrency" if effective != job["timeout_seconds"] else "original deadline")
    row["attempts"].append(attempt); row["status"] = "running"; json_dump(q.FOLDER / "queue_state.json", state)
    gate = "import os,sys; token=sys.stdin.readline(); sys.exit(74) if token!='GO\\n' else os.execv(sys.argv[1],sys.argv[1:])"
    with log.open("x") as stream:
        process = subprocess.Popen([sys.executable, "-c", gate, *launched], cwd=ROOT, env=env,
            stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    attempt.update(pid=process.pid, identity=q.process_identity(process.pid), status="running")
    json_dump(q.FOLDER / "queue_state.json", state); guard.attach(key, process.pid)
    process.stdin.write("GO\n"); process.stdin.close(); return process


def aliases(state):
    result = {}
    for path in q.MANIFESTS.glob("*.json"):
        manifest = json.loads(path.read_text())
        for original, recipe in manifest.get("recovery_recipes", {}).items():
            if original in result: raise ValueError("multiple recovery jobs for same original suite")
            if state["jobs"][original]["spec"] != recipe["original_spec"]: raise ValueError("recovery original spec changed")
            replacement = recipe["recovery_job"]; spec = state["jobs"][replacement]["spec"]
            for key in ("expected_config", "expected_status", "test_ids", "score_shape", "resource", "cores"):
                if spec[key] != recipe["original_spec"][key]: raise ValueError("recovery scientific configuration differs")
            result[original] = replacement
    return result


def reconcile(state, mapping):
    for original, replacement in mapping.items():
        row = state["jobs"][original]; restored = state["jobs"][replacement]
        if row["status"] == "completed" or restored["status"] != "completed": continue
        if not q.verified(ROOT / restored["output"], row["spec"]): raise ValueError("recovered suite not verified")
        meta = json.loads((ROOT / restored["output"] / "run_meta.json").read_text())
        if not meta.get("completed_recovery_receipt") or meta.get("recovery", {}).get("original_id") != original:
            raise ValueError("recovered suite missing final recovery provenance")
        row.update(status="completed", output=restored["output"], completion_job=replacement,
            completion_protocol="verified explicit deadline recovery; original attempts retained and charged")



def isolatable_failure(row):
    cfg = row["spec"]["expected_config"]
    if row["status"] != "stuck" or cfg.get("component") != "official_baseline" or not row["attempts"]: return False
    path = ROOT / row["attempts"][-1]["output"] / "run_meta.json"
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError): return False
    return (meta.get("status") == "blocked_algorithm_failure_rate" and bool(meta.get("errors"))
        and all(meta.get("config", {}).get(k) == v for k, v in cfg.items())
        and any(float(x) > .1 for x in meta.get("failure_rates", {}).values()))


def scientific_blocks(state):
    """Only actual official-algorithm failure-rate receipts permit scope isolation."""
    return {(cfg["dataset"], cfg["method"]) for row in state["jobs"].values()
        for cfg in [row["spec"]["expected_config"]] if isolatable_failure(row)}


def in_blocked_scope(row, blocked):
    cfg = row["spec"]["expected_config"]
    return cfg.get("component") == "official_baseline" and (cfg.get("dataset"), cfg.get("method")) in blocked


def run():
    guard = DiskGuard(ROOT); guard.poll(); processes = {}; cache = {}
    with (q.FOLDER / ".queue.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return {"status": "already_running"}
        path = q.FOLDER / "queue_state.json"; state = json.loads(path.read_text()); q.load_manifests(state, cache)
        for row in state["jobs"].values():
            if row["status"] == "running" and q.alive(row["attempts"][-1]):
                a = row["attempts"][-1]; guard.reserve(a["key"], ROOT / a["output"], ROOT / a["log"], child_pid=a["pid"])
        stop = False
        def stopping(*_):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGTERM, stopping); signal.signal(signal.SIGINT, stopping)
        print(json.dumps(dict(scheduler="extended_queue", source_sha256=sha256(Path(__file__)), cpu_slots=CPU_SLOTS,
            gpu_slots=q.SLOTS["gpu"], evidence_deadline_seconds=2400)), flush=True)
        while not stop:
            q.load_manifests(state, cache); mapping = aliases(state); reconcile(state, mapping)
            pilots = {j["id"] for p in q.MANIFESTS.glob("*.json") for m in [json.loads(p.read_text())] if "pilot" in m["phase"].lower() for j in m["jobs"]}
            def order(item):
                name, row = item; cfg = row["spec"]["expected_config"]
                return (0 if cfg.get("smoke") or name in pilots or name.startswith("recovered_") else 1 if cfg.get("component") == "train_only_preprocessing" else 2, name)
            state["jobs"] = dict(sorted(state["jobs"].items(), key=order))
            storage = guard.poll(); state["storage"] = storage
            for name, row in state["jobs"].items():
                if row["status"] == "running" and not q.alive(row["attempts"][-1]):
                    a = row["attempts"][-1]; effective = {**row["spec"], "timeout_seconds": deadline(row["spec"], a)}
                    q.finish(effective, row, a, processes.pop(name, None), guard)
            reconcile(state, mapping)
            active = Counter(r["spec"]["resource"] for r in state["jobs"].values() if r["status"] == "running")
            blocked = scientific_blocks(state)
            stuck = any(r["status"] == "stuck" and n not in mapping and not isolatable_failure(r)
                for n, r in state["jobs"].items())
            limited = False
            for name, row in state["jobs"].items():
                job = row["spec"]; resource = job["resource"]; slots = CPU_SLOTS if resource == "cpu" else q.SLOTS[resource]
                if (row["status"] != "pending" or name in mapping or in_blocked_scope(row, blocked)
                        or time.time() < row.get("retry_after", 0) or stuck): continue
                if active[resource] >= slots or not storage["safe_to_launch"]: continue
                bound = (deadline(job) + 10) * (job["cores"] if resource == "cpu" else 1) / 3600
                if charged(state, resource, True) + bound > (q.CPU_BUDGET if resource == "cpu" else q.GPU_BUDGET): limited = True; continue
                if resource == "gpu":
                    free = subprocess.check_output(["/usr/bin/nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
                    if int(free.strip()) < 8192: continue
                try: processes[name] = start(job, row, state, guard)
                except InsufficientSpace: storage = guard.poll(); break
                active[resource] += 1; storage = guard.poll()
            state.update(updated=q.now(), counts=dict(Counter(r["status"] for r in state["jobs"].values())),
                charged_hours={r: charged(state, r) for r in q.SLOTS}, admission_hours={r: charged(state, r, True) for r in q.SLOTS},
                scheduler_policy=dict(cpu_slots=CPU_SLOTS, gpu_slots=q.SLOTS["gpu"], source_sha256=sha256(Path(__file__))))
            if sum(active.values()): state["status"] = "running"
            elif stuck: state["status"] = "stuck"
            elif blocked and not any(r["status"] == "pending" and n not in mapping and not in_blocked_scope(r, blocked)
                    for n, r in state["jobs"].items()): state["status"] = "partial_blocked"
            elif limited: state["status"] = "budget_limited"
            elif not storage["safe_to_launch"]: state["status"] = "storage_limited"
            elif all(r["status"] == "completed" for r in state["jobs"].values()): state["status"] = "completed"
            else: state["status"] = "running"
            state["scientific_blocks"] = [dict(dataset=ds, method=method,
                reason="actual official failure rate exceeds 10%; author decision pending") for ds, method in sorted(blocked)]
            json_dump(path, state)
            if state["status"] != "running": break
            time.sleep(2)
        return state


def check():
    from .re2_deadline_recovery import create
    # Publish only while holding the shared scheduler lock. A live scheduler
    # owns admission; this checker never adds a replacement behind its back.
    storage = DiskGuard(ROOT).poll(); busy = False; recovery = None
    with (q.FOLDER / ".recovery.lock").open("a") as recovery_lock:
        fcntl.flock(recovery_lock, fcntl.LOCK_EX)
        with (q.FOLDER / ".queue.lock").open("a") as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: busy = True
            if not busy: recovery = create()
    state = json.loads((q.FOLDER / "queue_state.json").read_text()); action = "none"
    pending = {str(p.relative_to(ROOT)) for p in q.MANIFESTS.glob("*.json")} - set(state["manifests"])
    blocking = scientific_blocks(state)
    mapping = {name: recipe["recovery_job"] for path in q.MANIFESTS.glob("*.json")
        for name, recipe in json.loads(path.read_text()).get("recovery_recipes", {}).items()}
    runnable = any(r["status"] == "pending" and n not in mapping and not in_blocked_scope(r, blocking)
        for n, r in state["jobs"].items())
    unresolved = any(r["status"] == "stuck" and n not in mapping and not isolatable_failure(r)
        for n, r in state["jobs"].items())
    if not busy and storage["safe_to_launch"] and not unresolved and (pending or runnable
            or any(r["status"] == "running" for r in state["jobs"].values())):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f"); log = ROOT / f"logs/extended_resume_{stamp}.log"
        with log.open("x") as stream:
            child = subprocess.Popen([str(ROOT / "cache/exp_runtime/venv/bin/python"), "-m", "exp_revision.extended_queue", "--run"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        action = "resume_extended_queue"
    result = dict(time=q.now(), backend="local", status=state["status"], scheduler_alive=busy, jobs=state.get("counts"),
        storage=storage, action=action, recovery_manifest=str(recovery.relative_to(ROOT)) if recovery else None,
        scheduler_source_sha256=sha256(Path(__file__)))
    if action != "none": result.update(pid=child.pid, log=str(log.relative_to(ROOT)))
    json_dump(q.FOLDER / "monitor.json", result)
    with (q.FOLDER / "monitor_history.jsonl").open("a") as stream: stream.write(json.dumps(result) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", action="store_true"); p.add_argument("--check", action="store_true"); a = p.parse_args()
    result = run() if a.run else check(); print(json.dumps({k: result[k] for k in ("status", "counts", "jobs", "action", "pid") if k in result}))


if __name__ == "__main__": main()
