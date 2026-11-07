"""Recover the one TT evidence job interrupted by two environment resets.

The two original attempts and frozen manifest remain intact. This permits one
explicitly recorded third attempt, with the same model and conservative CPU
accounting. It never extends retries for other jobs or for model failures.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, SEEDS, json_dump, sha256
from .disk_guard import DiskGuard, InsufficientSpace, POLICY, guarded_command
from .evidence_queue import (
    DATASET_BUDGET_CORE_HOURS, KILL_GRACE_SECONDS, THREADS, TIMEOUT_SECONDS,
    _kill_group, _now, _pid_matches, _seconds, _verified, verify_manifest,
)


JOB_ID = "tt_fold4_seed46"
LEASE = "T1_evidence_tt_fold4_seed46_attempt3"


def recover(manifest_path: Path, scheduler_digest: str, guard_digest: str) -> dict:
    if (sha256(Path(__file__)) != scheduler_digest
            or sha256(ROOT / "exp_revision/disk_guard.py") != guard_digest):
        raise ValueError("recovery code differs from independently reviewed files")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"] != "tt":
        raise ValueError("environment recovery is restricted to the interrupted TT job")
    folds = verify_manifest(manifest)
    folder = ROOT / "results_v2/T1_evidence_full/tt"
    with (folder / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = folder / "queue_state.json"
        state = json.loads(path.read_text())
        if (state["manifest_sha256"] != sha256(manifest_path)
                or set(state["jobs"]) != {f"tt_fold{f}_seed{s}" for f in range(5) for s in SEEDS}):
            raise ValueError("recovery state differs from frozen 5×5 matrix")
        for key, row in state["jobs"].items():
            if key == JOB_ID:
                continue
            fold, seed = map(int, key.split("fold", 1)[1].split("_seed"))
            if row["status"] != "completed" or not _verified(
                    ROOT / row["output"], manifest, fold, seed, folds[fold]):
                raise ValueError(f"recovery requires all other jobs to be complete: {key}")
        row = state["jobs"][JOB_ID]
        if len(row["attempts"]) not in (2, 3):
            raise ValueError("recovery requires exactly two historical interrupted attempts")
        if len(row["attempts"]) == 2 and (row["status"] not in ("running", "stuck")
                or state["status"] not in ("running", "stuck", "storage_limited")):
            raise ValueError("target job is not in the documented environment-interruption state")
        for number, attempt in enumerate(row["attempts"][:2], start=1):
            output, log = ROOT / attempt["output"], ROOT / attempt["log"]
            expected_output = folder / JOB_ID / f"attempt{number}"
            expected_log = folder / "logs" / f"{JOB_ID}_attempt{number}.log"
            expected_command = ["/tmp/eadro-care-runtime/bin/python", "-m",
                                "exp_revision.train_evidence", "--dataset", "tt", "--fold", "4",
                                "--seed", "46", "--output", str(expected_output)]
            if (attempt["number"] != number or output != expected_output or log != expected_log
                    or attempt["command"] != expected_command
                    or attempt["status"] not in ("running", "interrupted")
                    or _pid_matches(attempt.get("pid", -1), output)
                    or (output.exists() and any(output.iterdir()))
                    or (log.exists() and log.stat().st_size)):
                raise ValueError("historical attempts are not empty environment interruptions")
            if attempt["status"] == "running":
                attempt.update(status="interrupted", ended=_now(),
                               elapsed_seconds=TIMEOUT_SECONDS + KILL_GRACE_SECONDS,
                               billing_basis="watchdog_upper_bound")
        receipt = {"reason": "local environment reset destroyed /tmp runtime and all local experiment processes",
                   "scope": JOB_ID, "additional_attempts": 1,
                   "scheduler_sha256": scheduler_digest, "disk_guard_sha256": guard_digest,
                   "disk_policy": POLICY, "recorded": _now()}
        if state.get("environment_recovery"):
            if any(state["environment_recovery"].get(k) != v for k, v in receipt.items() if k != "recorded"):
                raise ValueError("environment-recovery receipt changed")
        else:
            state["environment_recovery"] = receipt
        charged = sum(THREADS * float(a.get("elapsed_seconds", 0))
                      for job in state["jobs"].values() for a in job["attempts"]
                      if a["status"] != "running")
        state["charged_core_hours"] = charged / 3600
        guard = DiskGuard(ROOT)
        output = folder / JOB_ID / "attempt3"
        log = folder / "logs" / f"{JOB_ID}_attempt3.log"
        process = stream = None
        if len(row["attempts"]) == 3:
            attempt = row["attempts"][-1]
            if (attempt["number"] != 3 or ROOT / attempt["output"] != output
                    or ROOT / attempt["log"] != log):
                raise ValueError("third attempt differs from recorded recovery paths")
            if _pid_matches(attempt.get("pid", -1), output):
                guard.reserve(LEASE, output, log, child_pid=attempt["pid"])
            elif _verified(output, manifest, 4, 46, folds[4]):
                if attempt["status"] == "running":
                    attempt.update(status="completed", ended=_now(),
                                   elapsed_seconds=TIMEOUT_SECONDS + KILL_GRACE_SECONDS,
                                   billing_basis="watchdog_upper_bound")
                    charged += THREADS * attempt["elapsed_seconds"]
            else:
                if attempt["status"] in ("running", "launching"):
                    attempt.update(status="interrupted", ended=_now(),
                                   elapsed_seconds=TIMEOUT_SECONDS + KILL_GRACE_SECONDS,
                                   billing_basis="watchdog_upper_bound")
                    charged += THREADS * attempt["elapsed_seconds"]
                row["status"] = state["status"] = "stuck"
                state["charged_core_hours"] = charged / 3600
                json_dump(path, state)
                raise RuntimeError("the one permitted recovery attempt was interrupted; inspect it")
        else:
            if output.exists() or log.exists():
                raise FileExistsError("preserve existing recovery output/log")
            if charged + THREADS * (TIMEOUT_SECONDS + KILL_GRACE_SECONDS) > DATASET_BUDGET_CORE_HOURS * 3600:
                raise RuntimeError("recovery exceeds conservative dataset CPU slice")
            if (sha256(Path(__file__)) != scheduler_digest
                    or sha256(ROOT / "exp_revision/disk_guard.py") != guard_digest):
                raise ValueError("reviewed recovery code changed before launch")
            verify_manifest(manifest)
            try:
                storage_env = guard.reserve(LEASE, output, log)
            except InsufficientSpace:
                state["status"] = "storage_limited"
                json_dump(path, state)
                raise
            command = [sys.executable, "-m", "exp_revision.train_evidence", "--dataset", "tt",
                       "--fold", "4", "--seed", "46", "--output", str(output)]
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       ARISE_SOURCE_COMMIT=manifest["source_commit"], ARISE_SOURCE_DIRTY="true")
            env.update(storage_env)
            attempt = {"number": 3, "status": "launching", "started": _now(),
                       "output": str(output.relative_to(ROOT)), "log": str(log.relative_to(ROOT)),
                       "command": command, "reason": "environment reset recovery"}
            row["attempts"].append(attempt)
            row["status"] = state["status"] = "running"
            json_dump(path, state)
            stream = log.open("w")
            gate = ('IFS= read -r token || exit 0; [ "$token" = GO ] || exit 0; '
                    'ulimit -f 524288 || exit 1; '
                    'exec timeout --signal=TERM --kill-after=5s "$@"')
            process = subprocess.Popen(["bash", "-c", gate, "arise-gate", f"{TIMEOUT_SECONDS}s",
                                        *guarded_command(ROOT, LEASE, command)],
                                       cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True, text=True)
            attempt.update(pid=process.pid, status="running")
            json_dump(path, state)
            guard.attach(LEASE, process.pid)
            process.stdin.write("GO\n")
            process.stdin.close()
        if attempt["status"] == "running":
            while True:
                snapshot = guard.poll()
                state["storage"] = snapshot
                alive = process.poll() is None if process else _pid_matches(attempt["pid"], output)
                elapsed = max(0, time.time() - _seconds(attempt["started"]))
                if alive and (elapsed >= TIMEOUT_SECONDS or LEASE in snapshot["abort_job_keys"]):
                    _kill_group(attempt["pid"], output)
                    attempt["reason"] = "disk guard" if LEASE in snapshot["abort_job_keys"] else "job timeout"
                    continue
                if not alive:
                    if stream:
                        stream.close()
                    valid = _verified(output, manifest, 4, 46, folds[4])
                    attempt.update(status="completed" if valid else "failed", ended=_now(),
                                   elapsed_seconds=elapsed, returncode=process.returncode if process else None)
                    charged += THREADS * elapsed
                    guard.release(LEASE)
                    break
                time.sleep(2)
        valid = _verified(output, manifest, 4, 46, folds[4])
        row["status"] = "completed" if valid else "stuck"
        if valid:
            row["output"] = str(output.relative_to(ROOT))
        # Recheck the full frozen closure and all outputs before completion.
        if (sha256(Path(__file__)) != scheduler_digest
                or sha256(ROOT / "exp_revision/disk_guard.py") != guard_digest):
            raise ValueError("reviewed recovery code changed before completion")
        verify_manifest(manifest)
        for key, job in state["jobs"].items():
            fold, seed = map(int, key.split("fold", 1)[1].split("_seed"))
            if job["status"] == "completed" and not _verified(
                    ROOT / job["output"], manifest, fold, seed, folds[fold]):
                raise RuntimeError(f"evidence output changed before completion: {key}")
        state.update(status="completed" if valid else "stuck", ended=_now(),
                     charged_core_hours=charged / 3600)
        json_dump(path, state)
        if not valid:
            raise RuntimeError("environment-recovery attempt failed")
        return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scheduler-sha256", required=True)
    parser.add_argument("--disk-guard-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in recover(args.manifest, args.scheduler_sha256,
                                             args.disk_guard_sha256).items()
                      if k in ("status", "charged_core_hours")}))


if __name__ == "__main__":
    main()
