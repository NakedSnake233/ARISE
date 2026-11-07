"""Local append-only experiment batches with the existing shared disk guard.

New, immutable manifests can join the same queue while an earlier batch runs.
The single scheduler owns compute admission; watchdogs survive its interruption.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
from collections import Counter

import numpy as np
from .common import ROOT, json_dump, sha256
from .disk_guard import process_identity
from .disk_guard_local import DiskGuard, InsufficientSpace, guarded_command

FOLDER = ROOT / "results_v2/remaining_queue"
MANIFESTS = ROOT / "manifests/remaining"
CPU_BUDGET = 200.
GPU_BUDGET = 48.
# Conservatively covers all previous queues, preprocessing, migration probes,
# and interrupted attempts. Individual old receipts remain the authoritative
# breakdown; this reservation avoids counting only the newest matrix.
PRECHARGED = {"cpu": 40., "gpu": 8.}
SLOTS = {"cpu": 8, "gpu": 3}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def alive(attempt):
    return bool(attempt.get("identity") and process_identity(attempt.get("pid")) == attempt["identity"])


def verified(output: Path, job: dict) -> bool:
    try:
        meta = json.loads((output / "run_meta.json").read_text())
        cfg = meta["config"]
        if (meta["status"] not in job["expected_status"] or meta.get("errors")
                or meta["scores_sha256"] != sha256(output / "scores.npz")
                or any(cfg.get(k) != v for k,v in job["expected_config"].items())
                or not meta.get("end_time") or not meta.get("start_time")):
            return False
        with np.load(output / "scores.npz") as archive:
            if archive["incident_ids"].tolist() != job["test_ids"]:
                return False
            key = "scores" if "scores" in archive else "probabilities"
            scores = archive[key]
            return bool(np.isfinite(scores).all() and scores.shape == tuple(job["score_shape"]))
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
        return False


def load_manifests(state, cache):
    for path in sorted(MANIFESTS.glob("*.json")):
        manifest = json.loads(path.read_text())
        if manifest["backend"] != "local" or manifest["precharged_hours"] != PRECHARGED:
            raise ValueError("wrong compute backend or inherited compute accounting")
        if manifest["source_sha256"].get("exp_revision/remaining_queue.py") != sha256(Path(__file__)):
            raise ValueError("remaining scheduler changed since review")
        digest = sha256(path)
        if str(path.relative_to(ROOT)) in state["manifests"] and state["manifests"][str(path.relative_to(ROOT))] != digest:
            raise ValueError("registered manifest was modified")
        for relative, expected in manifest["source_sha256"].items():
            source = ROOT / relative
            stat = source.stat()
            identity = (stat.st_mtime_ns, stat.st_size, expected)
            if cache.get(relative) != identity:
                if sha256(source) != expected:
                    raise ValueError(f"frozen experiment input changed: {relative}")
                cache[relative] = identity
        for job in manifest["jobs"]:
            if job["resource"] not in SLOTS or job["max_attempts"] != 2:
                raise ValueError("unsupported job resource or retry policy")
            if job["id"] in state["jobs"]:
                if state["jobs"][job["id"]]["spec"] != job:
                    raise ValueError("existing job specification changed")
            else:
                state["jobs"][job["id"]] = {"spec": job, "status": "pending", "attempts": []}
        state["manifests"][str(path.relative_to(ROOT))] = digest


def start(job, row, state, guard):
    number = len(row["attempts"]) + 1
    key = f"remaining_{job['id']}_attempt{number}"
    output = ROOT / "results_v2/remaining_runs" / job["id"] / f"attempt{number}"
    log = ROOT / "logs" / f"{key}.log"
    if output.exists() or log.exists():
        raise FileExistsError("immutable attempt path occupied")
    env = {**os.environ, "ARISE_SOURCE_COMMIT":job["source_commit"],
           "OMP_NUM_THREADS": str(job["cores"]), "MKL_NUM_THREADS": str(job["cores"]),
           "OPENBLAS_NUM_THREADS": str(job["cores"]), "PYTHONUNBUFFERED": "1"}
    env.update(guard.reserve(key, output, log))
    command = [str(ROOT / "cache/exp_runtime/venv/bin/python"), *job["command"], "--output", str(output)]
    deadline = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s",
                str(job["timeout_seconds"])+"s", *guarded_command(ROOT, key, command)]
    attempt = {"number": number, "key": key, "pid": None, "identity": None,
               "start_time": now(), "start_epoch": time.time(), "output": str(output.relative_to(ROOT)),
               "log": str(log.relative_to(ROOT)), "status": "launching"}
    row["attempts"].append(attempt); row["status"] = "running"
    json_dump(FOLDER / "queue_state.json", state)
    gate = "import os,sys; token=sys.stdin.readline(); sys.exit(74) if token!='GO\\n' else os.execv(sys.argv[1],sys.argv[1:])"
    with log.open("x") as stream:
        process = subprocess.Popen([sys.executable, "-c", gate, *deadline], cwd=ROOT, env=env,
                                   stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
    attempt.update(pid=process.pid, identity=process_identity(process.pid), status="running")
    json_dump(FOLDER / "queue_state.json", state)
    guard.attach(key, process.pid)
    process.stdin.write("GO\n"); process.stdin.close()
    return process


def charged(state, resource, include_running=False):
    amount = PRECHARGED[resource]
    for row in state["jobs"].values():
        job = row["spec"]
        if job["resource"] != resource:
            continue
        factor = job["cores"] if resource == "cpu" else 1
        for attempt in row["attempts"]:
            if "elapsed_seconds" in attempt:
                amount += attempt["elapsed_seconds"]*factor/3600
            elif include_running:
                amount += (job["timeout_seconds"] + 10)*factor/3600
    return amount


def log_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-6000))
        return stream.read().decode(errors="replace")


def finish(job, row, attempt, process, guard):
    output, log = ROOT / attempt["output"], ROOT / attempt["log"]
    completed = verified(output, job)
    meta_path = output / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    duration = meta.get("elapsed_seconds") if completed else None
    # Lost scheduler timestamps are bounded by the actual deadline; completed
    # jobs use their recorded work plus the observed spawn/watchdog overhead.
    wall = min(time.time()-attempt["start_epoch"], job["timeout_seconds"]+10)
    attempt.update(end_time=now(), elapsed_seconds=max(wall, float(duration or 0)))
    if process is not None:
        attempt["exit_code"] = process.poll()
    guard.release(attempt["key"])
    if completed:
        attempt["status"] = row["status"] = "completed"
        row["output"] = attempt["output"]
        return
    tail = log_tail(log)
    code=attempt.get("exit_code")
    explicit_error=any(token in tail for token in ("Traceback (most recent call last)", "error:", "No module named", "can't open file"))
    retryable = ("out of memory" in tail.lower() or '"reason": "disk pressure"' in tail
                 or code in (124,137,143,-9,-15)
                 or code is None and not explicit_error and not meta)
    attempt["status"] = "interrupted" if retryable else "failed"
    if retryable and len(row["attempts"]) < job["max_attempts"]:
        row["status"] = "pending"
        row["retry_after"] = time.time() + (120 if "out of memory" in tail.lower() else 0)
    else:
        row["status"] = "stuck"; row["error_tail"] = tail


def run():
    guard = DiskGuard(ROOT); guard.poll()
    FOLDER.mkdir(parents=True, exist_ok=True); (ROOT / "logs").mkdir(exist_ok=True)
    with (FOLDER / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        path = FOLDER / "queue_state.json"
        state = json.loads(path.read_text()) if path.exists() else {
            "created": now(), "backend": "local", "status": "running", "manifests": {}, "jobs": {}}
        cache, processes = {}, {}
        load_manifests(state, cache)
        # Adopt children that outlived a stopped scheduler; their timeout PID
        # identity is stable across exec, and the existing watchdog remains.
        for row in state["jobs"].values():
            if row["status"] == "running" and alive(row["attempts"][-1]):
                a = row["attempts"][-1]
                guard.reserve(a["key"], ROOT/a["output"], ROOT/a["log"], child_pid=a["pid"])
        stop = False
        def stopping(*_):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGTERM, stopping); signal.signal(signal.SIGINT, stopping)
        while not stop:
            load_manifests(state, cache)
            storage = guard.poll(); state["storage"] = storage
            for key,row in state["jobs"].items():
                if row["status"] == "running" and not alive(row["attempts"][-1]):
                    finish(row["spec"], row, row["attempts"][-1], processes.pop(key,None), guard)
            active = Counter(row["spec"]["resource"] for row in state["jobs"].values() if row["status"] == "running")
            stuck = any(row["status"] == "stuck" for row in state["jobs"].values())
            limited = False
            for key,row in state["jobs"].items():
                job = row["spec"]; resource = job["resource"]
                if row["status"] != "pending" or time.time() < row.get("retry_after",0) or stuck:
                    continue
                if active[resource] >= SLOTS[resource] or not storage["safe_to_launch"]:
                    continue
                bound = (job["timeout_seconds"]+10)*(job["cores"] if resource == "cpu" else 1)/3600
                budget = CPU_BUDGET if resource == "cpu" else GPU_BUDGET
                if charged(state,resource,True)+bound > budget:
                    limited = True; continue
                if resource == "gpu":
                    free = subprocess.check_output(["/usr/bin/nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
                    if int(free.strip()) < 8192:
                        continue
                try:
                    processes[key] = start(job,row,state,guard)
                except InsufficientSpace:
                    storage = guard.poll(); break
                active[resource] += 1
                storage = guard.poll()
            state.update(updated=now(), charged_hours={r:charged(state,r) for r in SLOTS},
                         admission_hours={r:charged(state,r,True) for r in SLOTS},
                         counts=dict(Counter(row["status"] for row in state["jobs"].values())))
            if not sum(active.values()):
                if stuck: state["status"] = "stuck"
                elif limited: state["status"] = "budget_limited"
                elif not storage["safe_to_launch"]: state["status"] = "storage_limited"
                elif all(row["status"] == "completed" for row in state["jobs"].values()): state["status"] = "completed"
                else: state["status"] = "running"
            else:
                state["status"] = "running"
            json_dump(path,state)
            if state["status"] != "running":
                break
            time.sleep(2)
        return state


def check():
    guard = DiskGuard(ROOT); storage = guard.poll()
    FOLDER.mkdir(parents=True,exist_ok=True)
    busy = False
    with (FOLDER/".queue.lock").open("a") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: busy = True
    path=FOLDER/"queue_state.json"
    state=json.loads(path.read_text()) if path.exists() else {"status":"new","jobs":{},"manifests":{}}
    registered={str(p.relative_to(ROOT)) for p in MANIFESTS.glob("*.json")}
    pending_manifest=registered-set(state["manifests"])
    action="none"
    if not busy and storage["safe_to_launch"] and (state["status"] in ("new","running","storage_limited") or pending_manifest):
        stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        log=ROOT/f"logs/remaining_resume_{stamp}.log"
        with log.open("x") as stream:
            child=subprocess.Popen([str(ROOT/"cache/exp_runtime/venv/bin/python"),"-m","exp_revision.remaining_queue","--run"],
                                   cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        action="resume_local_remaining_queue"
    result={"time":now(),"backend":"local","status":state["status"],"scheduler_alive":busy,
            "jobs":dict(Counter(row["status"] for row in state["jobs"].values())),"action":action,
            "storage":storage}
    json_dump(FOLDER/"monitor.json",result)
    with (FOLDER/"monitor_history.jsonl").open("a") as history: history.write(json.dumps(result)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",action="store_true"); p.add_argument("--check",action="store_true")
    a=p.parse_args()
    if not a.run and not a.check: p.error("choose --run or --check")
    result=run() if a.run else check()
    print(json.dumps({k:result[k] for k in ("status","counts","jobs","action") if k in result},ensure_ascii=False))


if __name__ == "__main__": main()
