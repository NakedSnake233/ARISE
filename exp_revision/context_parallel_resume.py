"""Resume the frozen TrainTicket CUDA matrix with three concurrent workers.

Only scheduling changes. The original context manifest, model source, train
parameters, immutable attempt outputs, and source-commit receipts stay fixed.
An isolated duplicate-job benchmark found bit-identical CUDA probabilities
for seeds 42 and 43 with three simultaneous processes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from .common import ROOT, SEEDS, json_dump, sha256
from .context_queue import (
    FOLDS, GPU_BUDGET_HOURS, KILL_GRACE_SECONDS, MAX_ATTEMPTS,
    _kill_group, _log_retryable, _pid_matches, _stamp_seconds,
    _verified_output, _verify_files_unchanged, verify_manifest,
)


WORKERS = 3
GPU_MEMORY_LAUNCH_CEILING_MIB = 6000


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _verified(output: Path, manifest: dict, fold: int, seed: int, data: object) -> bool:
    try:
        if not _verified_output(output, manifest, fold, seed, data):
            return False
        with np.load(output / "scores.npz") as archive:
            scores = archive["probabilities"]
            return (np.isfinite(scores).all() and (scores >= 0).all()
                    and np.allclose(scores.sum(axis=1), 1, atol=1e-5))
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
        return False


def _gpu_capacity() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    values = [int(line) for line in result.stdout.splitlines() if line.strip()]
    return len(values) == 1 and values[0] < GPU_MEMORY_LAUNCH_CEILING_MIB


def run(manifest_path: Path, expected_scheduler_sha256: str) -> dict:
    if (len(expected_scheduler_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in expected_scheduler_sha256) or
            sha256(Path(__file__)) != expected_scheduler_sha256):
        raise ValueError("parallel scheduler does not match reviewed deployment hash")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"] != "tt" or manifest["device"] != "cuda":
        raise ValueError("parallel resume is restricted to frozen TrainTicket CUDA")
    folds = verify_manifest(manifest)
    _verify_files_unchanged(manifest)
    root = ROOT / "results_v2/T1_context_full/tt"
    state_path = root / "queue_state.json"
    queue_lock = (root / ".queue.lock").open("w")
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = Path("/tmp/arise_gpu0.lock").open("w")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(state_path.read_text())
    if state["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("queue state does not match original frozen manifest")
    job_ids = [f"tt_fold{fold}_seed{seed}" for fold in FOLDS for seed in SEEDS]
    if set(state["jobs"]) != set(job_ids):
        raise ValueError("queue state does not contain the exact 5×5 matrix")
    if state["status"] not in ("running", "completed"):
        raise ValueError(f"cannot resume queue state {state['status']}")
    scheduler = {"name": "context_parallel_resume", "workers": WORKERS,
                 "gpu_memory_launch_ceiling_mib": GPU_MEMORY_LAUNCH_CEILING_MIB,
                 "source_sha256": expected_scheduler_sha256,
                 "started": _now(),
                 "note": "CUDA seed 42/43 duplicate benchmark was bit-identical at three-way concurrency"}
    if state.get("parallel_scheduler") and state["parallel_scheduler"]["source_sha256"] != scheduler["source_sha256"]:
        raise ValueError("parallel scheduler source changed after resume")
    state["parallel_scheduler"] = state.get("parallel_scheduler", scheduler)
    state["workers"] = WORKERS
    timeout = int(manifest["max_job_seconds"])
    if timeout != 3600 or timeout * len(job_ids) > GPU_BUDGET_HOURS * 3600:
        raise ValueError("frozen matrix could exceed GPU-hour budget")
    active: dict[str, tuple[subprocess.Popen | None, object | None]] = {}
    charged_seconds = 0.0
    for job_id in job_ids:
        row = state["jobs"][job_id]
        fold, seed = map(int, job_id.split("fold", 1)[1].split("_seed"))
        for index, attempt in enumerate(row["attempts"]):
            if attempt["status"] == "launching":
                attempt.update(status="interrupted", ended=_now(), elapsed_seconds=0.0)
            if attempt["status"] == "running":
                output = ROOT / attempt["output"]
                if index != len(row["attempts"]) - 1:
                    row["status"] = "stuck"
                elif _pid_matches(attempt["pid"], str(output)):
                    active[job_id] = (None, None)
                else:
                    attempt.update(status="interrupted", ended=_now(),
                                   elapsed_seconds=min(timeout + KILL_GRACE_SECONDS,
                                                       max(0, time.time() - _stamp_seconds(attempt["started"]))),
                                   billing_basis="watchdog_upper_bound")
            if attempt["status"] != "running":
                charged_seconds += float(attempt.get("elapsed_seconds", 0))
        if row["status"] == "stuck":
            continue
        if not row["attempts"]:
            row["status"] = "pending"
            continue
        last = row["attempts"][-1]
        output = ROOT / last["output"]
        if last["status"] == "running":
            row["status"] = "running"
        elif _verified(output, manifest, fold, seed, folds[fold]):
            last["status"] = row["status"] = "completed"
            row["output"] = last["output"]
        elif (last["status"] == "interrupted" or
              (last["status"] == "failed" and
               (last.get("reason") == "job timeout" or
                _log_retryable(ROOT / last["log"])))) and len(row["attempts"]) < MAX_ATTEMPTS:
            row["status"] = "pending"
        else:
            row["status"] = "stuck"
    state["charged_core_or_gpu_hours"] = charged_seconds / 3600
    json_dump(state_path, state)
    if any(row["status"] == "stuck" for row in state["jobs"].values()):
        state["status"] = "stuck"
        json_dump(state_path, state)
        raise RuntimeError("queue has a stuck attempt")
    if state["status"] == "completed":
        if any(row["status"] != "completed" for row in state["jobs"].values()):
            raise ValueError("completed queue state contains unfinished jobs")
        return state
    logs = root / "logs"
    halted = False
    while True:
        for job_id, (process, stream) in list(active.items()):
            row = state["jobs"][job_id]
            attempt = row["attempts"][-1]
            fold, seed = map(int, job_id.split("fold", 1)[1].split("_seed"))
            output = ROOT / attempt["output"]
            elapsed = time.time() - _stamp_seconds(attempt["started"])
            running = process.poll() is None if process is not None else _pid_matches(attempt["pid"], str(output))
            if running and elapsed < timeout:
                continue
            if running:
                _kill_group(attempt["pid"])
                attempt["reason"] = "job timeout"
                elapsed = time.time() - _stamp_seconds(attempt["started"])
            if stream is not None:
                stream.close()
            valid = _verified(output, manifest, fold, seed, folds[fold])
            attempt.update(status="completed" if valid else "failed", ended=_now(),
                           elapsed_seconds=max(0, elapsed),
                           returncode=process.returncode if process is not None else None)
            charged_seconds += attempt["elapsed_seconds"]
            row["status"] = "completed" if valid else (
                "pending" if len(row["attempts"]) < MAX_ATTEMPTS and
                (attempt.get("reason") == "job timeout" or _log_retryable(ROOT / attempt["log"]))
                else "stuck")
            if valid:
                row["output"] = attempt["output"]
            elif row["status"] == "stuck":
                halted = True
                state["status"] = "stuck"
            del active[job_id]
            state["charged_core_or_gpu_hours"] = charged_seconds / 3600
            json_dump(state_path, state)
            print(json.dumps({"job": job_id, "status": row["status"],
                              "charged_hours": charged_seconds / 3600}), flush=True)
        pending = [job_id for job_id in job_ids if state["jobs"][job_id]["status"] == "pending"]
        if halted and not active:
            raise RuntimeError("parallel context queue stopped after failed job")
        if not pending and not active:
            break
        while pending and not halted and len(active) < WORKERS:
            reserved = (len(active) + 1) * (timeout + KILL_GRACE_SECONDS)
            if charged_seconds + reserved > GPU_BUDGET_HOURS * 3600:
                state["status"] = "budget_limited"
                json_dump(state_path, state)
                halted = True
                break
            if not _gpu_capacity():
                break
            _verify_files_unchanged(manifest)
            job_id = pending.pop(0)
            row = state["jobs"][job_id]
            fold, seed = map(int, job_id.split("fold", 1)[1].split("_seed"))
            number = len(row["attempts"]) + 1
            output = root / job_id / f"attempt{number}"
            log = logs / f"{job_id}_attempt{number}.log"
            if output.exists() or log.exists():
                raise FileExistsError("immutable CUDA attempt path already exists")
            command = [sys.executable, "-m", "exp_revision.train_context",
                       "--dataset", "tt", "--fold", str(fold), "--seed", str(seed),
                       "--device", "cuda", "--output", str(output)]
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       ARISE_SOURCE_COMMIT=manifest["source_commit"], ARISE_SOURCE_DIRTY="true")
            row["attempts"].append({"number": number, "status": "launching",
                                    "started": _now(), "output": str(output.relative_to(ROOT)),
                                    "log": str(log.relative_to(ROOT)), "command": command})
            row["status"] = "running"
            json_dump(state_path, state)
            stream = log.open("w")
            gate = ('IFS= read -r token || exit 0; '
                    '[ "$token" = GO ] || exit 0; '
                    'exec timeout --signal=TERM --kill-after=5s "$@"')
            process = subprocess.Popen(["bash", "-c", gate, "arise-gate",
                                        f"{timeout}s", *command], cwd=ROOT, env=env,
                                       stdin=subprocess.PIPE, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       text=True)
            row["attempts"][-1].update(pid=process.pid, status="running")
            active[job_id] = (process, stream)
            json_dump(state_path, state)
            assert process.stdin is not None
            process.stdin.write("GO\n")
            process.stdin.close()
            print(json.dumps({"launched": job_id, "pid": process.pid,
                              "workers_active": len(active)}), flush=True)
        time.sleep(2 if active else 20)
    state.update(status="completed", ended=_now(),
                 charged_core_or_gpu_hours=charged_seconds / 3600)
    json_dump(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scheduler-sha256", required=True)
    args = parser.parse_args()
    state = run(args.manifest, args.scheduler_sha256)
    print(json.dumps({"status": state["status"],
                      "charged_hours": state["charged_core_or_gpu_hours"]}))


if __name__ == "__main__":
    main()
