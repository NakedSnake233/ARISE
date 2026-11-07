"""Twenty-minute host checks and bounded recovery of the local ARISE queue."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .common import ROOT, json_dump, sha256
from .disk_guard import process_identity
from .disk_guard_local import DiskGuard, guarded_command


MONITOR = ROOT / "results_v2/local_monitor"
QUEUE = ROOT / "results_v2/T1_context_local_full/tt"
CONFIG = ROOT / "exp_revision/local_monitor_config.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify_sources(config: dict) -> None:
    if config["backend"] != "local" or config["interval_minutes"] != 20:
        raise ValueError("monitor is restricted to the requested local 20-minute schedule")
    for relative, digest in config["source_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"reviewed monitor input changed: {relative}")


def _lock_held(path: Path) -> bool:
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def _alive(record: dict | None) -> bool:
    return bool(record and record.get("identity") is not None
                and process_identity(record.get("pid")) == record["identity"])


def _log_tail(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 4000))
        return stream.read().decode(errors="replace")


def _environment() -> dict[str, str]:
    return {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}


def _scheduler_command(config: dict) -> list[str]:
    return [str(ROOT / "cache/exp_runtime/venv/bin/python"), "-m",
            "exp_revision.context_namespace_resume", "--manifest",
            str(ROOT / "manifests/tt_context_full.json"), "--wrapper-sha256",
            config["source_sha256"]["exp_revision/context_namespace_resume.py"],
            "--local-disk-guard-sha256",
            config["source_sha256"]["exp_revision/disk_guard_local.py"]]


def _start_scheduler(config: dict) -> dict:
    # The namespace-aware scheduler takes its own queue and GPU locks and
    # adopts surviving trainers; completed attempts are never retrained.
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    log = ROOT / f"logs/arise_local_monitor_resume_{stamp}.log"
    command = _scheduler_command(config)
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=_environment(),
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    return {"started": _now(), "pid": process.pid,
            "identity": process_identity(process.pid), "command": command,
            "log": str(log.relative_to(ROOT))}


def _start_collection(config: dict, phase: str, number: int,
                      guard: DiskGuard) -> dict:
    stage = "T1_context_local_full" if phase == "context" else "T1_arise_s0"
    output = ROOT / f"results_v2/{stage}/tt"
    key = f"T1_local_collect_{phase}_attempt{number}"
    log = ROOT / f"logs/{key}.log"
    receipt = output / f"monitor_collection_attempt{number}.json"
    if log.exists() or receipt.exists():
        raise FileExistsError("immutable collection attempt already exists")
    command = [str(ROOT / "cache/exp_runtime/venv/bin/python"), "-m",
               "exp_revision.local_collect", "--phase", phase,
               "--config", str(CONFIG), "--receipt", str(receipt)]
    env = _environment()
    env.update(guard.reserve(key, output, log))
    record = {"started": _now(), "phase": phase, "number": number, "pid": None,
              "identity": None, "command": command,
              "log": str(log.relative_to(ROOT)), "lease": key,
              "receipt": str(receipt.relative_to(ROOT)), "status": "launching"}
    # Persist the attempt before creating any immutable log or child. Even a
    # crash before Popen returns must not strand an occupied attempt1 path.
    json_dump(MONITOR / f"launching_{phase}.json", record)
    # A durable launch gate records the timeout PID and attaches its lease
    # before the watchdog can start writing.
    deadline = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "300s",
                *guarded_command(ROOT, key, command)]
    gate = ("import os,sys; "
            "token=sys.stdin.readline(); "
            "sys.exit(74) if token!='GO\\n' else os.execv(sys.argv[1],sys.argv[1:])")
    with log.open("x") as stream:
        process = subprocess.Popen([sys.executable, "-c", gate, *deadline],
                                   cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
    record.update(pid=process.pid, identity=process_identity(process.pid), status="running")
    json_dump(MONITOR / f"launching_{phase}.json", record)
    guard.attach(key, process.pid)
    assert process.stdin is not None
    process.stdin.write("GO\n")
    process.stdin.close()
    return record


def _collect(config: dict, previous: dict, guard: DiskGuard) -> dict:
    collected = previous.get("collections", {})
    for phase in ("context", "arise"):
        attempts = collected.setdefault(phase, [])
        # Recover the durable launch record if the check process died between
        # opening the gate and persisting the combined monitor state.
        launch_path = MONITOR / f"launching_{phase}.json"
        if launch_path.exists():
            launched = json.loads(launch_path.read_text())
            if launched["number"] == len(attempts) + 1:
                attempts.append(launched)
            elif (launched["number"] > len(attempts)
                  or attempts[launched["number"] - 1]["pid"] != launched["pid"]):
                raise ValueError("collection launch record differs from monitor state")
        if attempts:
            last = attempts[-1]
            if _alive(last):
                continue
            receipt = ROOT / last["receipt"]
            if receipt.exists():
                result = json.loads(receipt.read_text())
                if (result["status"] == "completed" and result["phase"] == phase
                        and sha256(ROOT / result["prediction_path"]) == result["predictions_sha256"]):
                    last["status"] = "completed"
                    continue
                # A real evaluation/configuration error needs inspection;
                # retrying unchanged must not turn it into a result.
                last["status"] = "needs_repair"
                continue
            last["status"] = "interrupted"
            if len(attempts) >= 2:
                last["status"] = "needs_repair"
                continue
        if not guard.poll()["safe_to_launch"]:
            continue
        attempts.append(_start_collection(config, phase, len(attempts) + 1, guard))
        # Persist each launched child before another phase can be launched.
        json_dump(MONITOR / "state.json", {**previous, "collections": collected})
    return collected


def check(config: dict) -> dict:
    verify_sources(config)
    # The PID namespace is checked before writing monitor or queue state.
    guard = DiskGuard(ROOT)
    storage = guard.poll()
    MONITOR.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with (MONITOR / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"time": _now(), "status": "another_check_running"}
        state_path = MONITOR / "state.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        queue = json.loads((QUEUE / "queue_state.json").read_text())
        busy = (_lock_held(QUEUE / ".namespace_scheduler.lock")
                or _lock_held(QUEUE / ".queue.lock")
                or _alive(previous.get("restart")))
        result = {**previous, "time": _now(), "interval_minutes": 20,
                  "backend": "local", "queue_status": queue["status"],
                  "jobs": dict(Counter(row["status"] for row in queue["jobs"].values())),
                  "charged_gpu_hours": queue["charged_core_or_gpu_hours"],
                  "scheduler_alive": busy, "storage": storage, "action": "none"}
        gpu = subprocess.run(["/usr/bin/nvidia-smi", "--query-gpu=name,memory.used,memory.free,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        result["gpu"] = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
        scheduler_log = previous.get("restart", {}).get(
            "log", "logs/arise_tt_context_local_namespace_20260930.log")
        result["recent_logs"] = {"scheduler": _log_tail(ROOT / scheduler_log)}
        for job_id, job in queue["jobs"].items():
            if job.get("attempts") and job["status"] not in ("completed", "pending"):
                result["recent_logs"][job_id] = _log_tail(ROOT / job["attempts"][-1]["log"])
        if storage["abort_job_keys"] or not storage["safe_to_launch"]:
            result["action"] = "wait_for_safe_storage"
        elif not busy and queue["status"] in ("running", "storage_limited"):
            result["restart"] = _start_scheduler(config)
            result["action"] = "resume_frozen_local_queue"
        elif not busy and queue["status"] == "completed":
            result["collections"] = _collect(config, result, guard)
            result["action"] = "collect_verified_T1_results"
        elif not busy:
            result["action"] = "needs_repair" if queue["status"] == "stuck" else "respect_compute_limit"
            # Preserve a concise primary error trace for the next inspection.
            result["error_tail"] = result["recent_logs"]
        json_dump(state_path, result)
        with (MONITOR / "history.jsonl").open("a") as history:
            history.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    result = check(json.loads(args.config.read_text()))
    print(json.dumps({key: result[key] for key in
                      ("time", "jobs", "queue_status", "scheduler_alive", "action")
                      if key in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
