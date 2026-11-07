"""Bounded, resumable full-fold context-only experiment queue.

All 5 folds and 5 frozen model seeds are required. The queue sees training
labels only; saved scores are evaluated against held-out labels separately.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import ROOT, SEEDS, json_dump, sha256
from .fold_data import load_fold


FOLDS = tuple(range(5))
MAX_ATTEMPTS = 2
THREADS_PER_CPU_JOB = 4
GPU_BUDGET_HOURS = 48
CPU_CORE_BUDGET_HOURS = 200
KILL_GRACE_SECONDS = 5


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _stamp_seconds(stamp: str) -> float:
    return dt.datetime.fromisoformat(stamp).timestamp()


def _expected_paths(dataset: str) -> list[Path]:
    paths = [ROOT / name for name in (
        "exp_revision/train_context.py", "exp_revision/context_queue.py",
        "exp_revision/fold_data.py", "exp_revision/common.py",
        "exp_revision/metrics.py", "scripts/care/train_base.py",
        "configs/care/review-response-v1.json",
        f"splits/{dataset}_kfold5.json",
    )]
    if dataset == "tt":
        paths.append(ROOT / "splits/tt_inner_validation_kfold5.json")
        paths.append(ROOT / "exp_revision/preprocess_tt.py")
        paths.append(ROOT / "splits/tt_unlabeled_events.json")
    else:
        paths.append(ROOT / "exp_revision/preprocess_aiops.py")
        paths.append(ROOT / "splits/aiops_unlabeled_events.json")
    for fold in FOLDS:
        paths.extend([
            ROOT / f"splits/{dataset}_training_labels/fold_{fold}.json",
            ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}/features.npz",
            ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}/metadata.json",
        ])
    return paths


def make_manifest(dataset: str, source_commit: str) -> dict:
    if dataset not in ("tt", "aiops"):
        raise ValueError(dataset)
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be the full Git SHA")
    if (ROOT / ".git").exists():
        actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=ROOT, text=True).strip()
        if source_commit != actual_commit:
            raise ValueError("source_commit differs from local Git HEAD")
    config = json.loads((ROOT / "configs/care/review-response-v1.json").read_text())
    device = config["datasets"][{"tt": "trainticket", "aiops": "aiops2022"}[dataset]]["device"]
    fold_counts = {}
    for fold in FOLDS:
        data = load_fold(dataset, fold, smoke=False)
        fold_counts[str(fold)] = {"train": len(data.train_ids), "test": len(data.test_ids),
                                  "services": len(data.services)}
    return {
        "protocol": "T1_context_only_full_kfold5_v1", "dataset": dataset,
        "device": device, "source_commit": source_commit,
        "folds": list(FOLDS), "seeds": list(SEEDS),
        "files_sha256": {str(path.relative_to(ROOT)): sha256(path)
                         for path in _expected_paths(dataset)},
        "fold_counts": fold_counts,
        "budget_hours": {"gpu": GPU_BUDGET_HOURS, "cpu_core": CPU_CORE_BUDGET_HOURS},
        "threads_per_cpu_job": THREADS_PER_CPU_JOB,
        "max_job_seconds": 3600 if device == "cuda" else 600,
    }


def verify_manifest(manifest: dict) -> dict[int, object]:
    dataset = manifest["dataset"]
    source_commit = manifest.get("source_commit", "")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("manifest source_commit is not a full Git SHA")
    if manifest.get("protocol") != "T1_context_only_full_kfold5_v1":
        raise ValueError("wrong context queue protocol")
    if manifest.get("folds") != list(FOLDS) or manifest.get("seeds") != list(SEEDS):
        raise ValueError("full queue requires exactly the frozen 5 folds and 5 seeds")
    config = json.loads((ROOT / "configs/care/review-response-v1.json").read_text())
    expected_device = config["datasets"][{"tt": "trainticket", "aiops": "aiops2022"}[dataset]]["device"]
    if manifest["device"] != expected_device:
        raise ValueError("queue device differs from frozen dataset configuration")
    if (manifest.get("budget_hours") != {"gpu": GPU_BUDGET_HOURS,
                                         "cpu_core": CPU_CORE_BUDGET_HOURS}
            or manifest.get("threads_per_cpu_job") != THREADS_PER_CPU_JOB
            or manifest.get("max_job_seconds") != (3600 if expected_device == "cuda" else 600)):
        raise ValueError("queue compute limits differ from the authorized fixed limits")
    expected = {str(path.relative_to(ROOT)): sha256(path) for path in _expected_paths(dataset)}
    if manifest["files_sha256"] != expected:
        changed = [path for path in expected if manifest["files_sha256"].get(path) != expected[path]]
        raise ValueError(f"source or fold input differs from frozen manifest: {changed[:5]}")
    return {fold: load_fold(dataset, fold, smoke=False) for fold in FOLDS}


def _verify_files_unchanged(manifest: dict) -> None:
    for relative, expected in manifest["files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen queue input changed before job launch: {relative}")


def _verified_output(output: Path, manifest: dict, fold: int, seed: int, data: object) -> bool:
    meta_path, scores_path, model_path = (output / name for name in
                                          ("run_meta.json", "scores.npz", "model.pt"))
    if not all(path.is_file() for path in (meta_path, scores_path, model_path)):
        return False
    try:
        meta = json.loads(meta_path.read_text())
        config = meta["config"]
        if (meta.get("status") != "partial_T1_context_only"
                or meta.get("git_commit") != manifest["source_commit"]
                or (config["dataset"], config["fold"], config["seed"], config["smoke"])
                != (manifest["dataset"], fold, seed, False)
                or meta.get("scores_sha256") != sha256(scores_path)
                or meta.get("model_sha256") != sha256(model_path)):
            return False
        for relative, digest in meta["input_sha256"].items():
            if manifest["files_sha256"].get(relative) != digest:
                return False
        required = {
            "exp_revision/train_context.py", "exp_revision/fold_data.py",
            "scripts/care/train_base.py", "configs/care/review-response-v1.json",
            f"splits/{manifest['dataset']}_kfold5.json",
            f"splits/{manifest['dataset']}_training_labels/fold_{fold}.json",
            f"cache/fold_preprocessing/{manifest['dataset']}/fold{fold}/features.npz",
            f"cache/fold_preprocessing/{manifest['dataset']}/fold{fold}/metadata.json",
        }
        if manifest["dataset"] == "tt":
            required.add("splits/tt_inner_validation_kfold5.json")
        if set(meta["input_sha256"]) != required:
            return False
        with np.load(scores_path) as archive:
            return (archive["incident_ids"].tolist() == data.train_ids + data.test_ids
                    and archive["probabilities"].shape ==
                    (len(data.train_ids) + len(data.test_ids), len(data.services)))
    except (OSError, KeyError, ValueError):
        return False


def _pid_matches(pid: int, output: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return "exp_revision.train_context" in command and output in command
    except OSError:
        return False


def _gpu_free() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    values = [int(line) for line in result.stdout.splitlines() if line.strip()]
    return len(values) == 1 and values[0] < 500


def _log_retryable(path: Path) -> bool:
    try:
        text = path.read_text(errors="replace")[-4000:].lower()
    except OSError:
        return False
    return "out of memory" in text or "resource temporarily unavailable" in text


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    if _pid_matches(pid, ""):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(manifest_path: Path, *, workers: int, output_root: Path = ROOT / "results_v2/T1_context_full") -> dict:
    manifest = json.loads(manifest_path.read_text())
    folds = verify_manifest(manifest)
    dataset, device = manifest["dataset"], manifest["device"]
    if workers < 1 or workers > 16 or (device == "cuda" and workers != 1):
        raise ValueError("CPU workers must be 1–16; the single GPU permits one CUDA worker")
    root = output_root / dataset
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    lock = (root / ".queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = None
    if device == "cuda":
        gpu_lock = Path("/tmp/arise_gpu0.lock").open("w")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_hash = sha256(manifest_path)
    state_path = root / "queue_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["manifest_sha256"] != manifest_hash:
            raise ValueError("existing queue belongs to a different frozen manifest")
    else:
        state = {"manifest_sha256": manifest_hash, "dataset": dataset, "device": device,
                 "workers": workers, "created": _now(), "jobs": {}, "status": "running"}
    ids = [f"{dataset}_fold{fold}_seed{seed}" for fold in FOLDS for seed in SEEDS]
    for fold in FOLDS:
        for seed in SEEDS:
            state["jobs"].setdefault(f"{dataset}_fold{fold}_seed{seed}",
                                     {"status": "pending", "attempts": []})
    if set(state["jobs"]) != set(ids):
        raise ValueError("queue state does not contain exactly the full matrix")

    factor = 1 if device == "cuda" else THREADS_PER_CPU_JOB
    budget_seconds = 3600 * manifest["budget_hours"]["gpu" if device == "cuda" else "cpu_core"]
    timeout = int(manifest["max_job_seconds"])
    if timeout <= 0 or factor * timeout * len(ids) > budget_seconds:
        raise ValueError("worst-case full matrix exceeds the compute-hour limit")
    active: dict[str, tuple[subprocess.Popen | None, object | None]] = {}
    charged = 0.0
    for job_id in ids:
        row = state["jobs"][job_id]
        fold, seed = map(int, job_id.split("fold", 1)[1].split("_seed"))
        data = folds[fold]
        for index, attempt in enumerate(row["attempts"]):
            if attempt["status"] == "launching":
                # The launch gate cannot start training until the PID has been
                # durably recorded and the parent writes GO to its stdin.
                attempt["status"] = "interrupted"
                attempt["ended"] = _now()
                attempt["elapsed_seconds"] = 0.0
            if attempt["status"] == "running":
                output = ROOT / attempt["output"]
                if index != len(row["attempts"]) - 1:
                    row["status"] = "stuck"
                elif _pid_matches(attempt["pid"], str(output)):
                    active[job_id] = (None, None)
                else:
                    attempt["status"] = "interrupted"
                    attempt["ended"] = _now()
                    # An independent GNU timeout stops the child within this
                    # limit even if the queue was down for days. Conservatively
                    # charge the upper bound, never the offline wall interval.
                    attempt["elapsed_seconds"] = min(
                        timeout + KILL_GRACE_SECONDS,
                        max(0, time.time() - _stamp_seconds(attempt["started"])))
                    attempt["billing_basis"] = "watchdog_upper_bound"
            if attempt["status"] != "running":
                charged += factor * float(attempt.get("elapsed_seconds", 0))
        if row["status"] != "stuck":
            if not row["attempts"]:
                row["status"] = "pending"
            else:
                last = row["attempts"][-1]
                output = ROOT / last["output"]
                if last["status"] == "running":
                    row["status"] = "running"
                elif _verified_output(output, manifest, fold, seed, data):
                    last["status"] = "completed"
                    row["status"] = "completed"
                    row["output"] = last["output"]
                elif last["status"] in ("interrupted", "failed") and len(row["attempts"]) < MAX_ATTEMPTS and (
                        last["status"] == "interrupted" or _log_retryable(ROOT / last["log"])):
                    row["status"] = "pending"
                else:
                    row["status"] = "stuck"
        if row["status"] == "stuck":
            state["status"] = "stuck"
    json_dump(state_path, state)
    if state["status"] == "stuck":
        raise RuntimeError("queue contains a stuck job; inspect immutable attempts")

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
            verified = _verified_output(output, manifest, fold, seed, folds[fold])
            attempt["ended"] = _now()
            attempt["elapsed_seconds"] = max(0, elapsed)
            charged += factor * attempt["elapsed_seconds"]
            attempt["status"] = "completed" if verified else "failed"
            attempt["returncode"] = process.returncode if process is not None else None
            row["status"] = "completed" if verified else (
                "pending" if len(row["attempts"]) < MAX_ATTEMPTS
                and _log_retryable(ROOT / attempt["log"]) else "stuck")
            if verified:
                row["output"] = attempt["output"]
            del active[job_id]
            state["charged_core_or_gpu_hours"] = charged / 3600
            json_dump(state_path, state)
            print(json.dumps({"job": job_id, "status": row["status"],
                              "charged_hours": charged / 3600}), flush=True)
            if row["status"] == "stuck":
                state["status"] = "stuck"
                json_dump(state_path, state)
                halted = True

        pending = [job_id for job_id in ids if state["jobs"][job_id]["status"] == "pending"]
        if halted and not active:
            raise RuntimeError(f"queue stopped after a failed job; inspect {state_path}")
        if not pending and not active:
            break
        while pending and not halted and len(active) < workers:
            max_attempt_seconds = timeout + KILL_GRACE_SECONDS
            reserved = factor * max_attempt_seconds * len(active)
            if charged + reserved + factor * max_attempt_seconds > budget_seconds:
                state["status"] = "budget_limited"
                json_dump(state_path, state)
                halted = True
                break
            if device == "cuda" and not _gpu_free():
                break
            _verify_files_unchanged(manifest)
            job_id = pending.pop(0)
            row = state["jobs"][job_id]
            fold, seed = map(int, job_id.split("fold", 1)[1].split("_seed"))
            number = len(row["attempts"]) + 1
            output = root / job_id / f"attempt{number}"
            log = logs / f"{job_id}_attempt{number}.log"
            if output.exists() or log.exists():
                raise FileExistsError(f"immutable attempt path already exists: {output} or {log}")
            command = [sys.executable, "-m", "exp_revision.train_context",
                       "--dataset", dataset, "--fold", str(fold), "--seed", str(seed),
                       "--device", device, "--output", str(output)]
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       ARISE_SOURCE_COMMIT=manifest["source_commit"], ARISE_SOURCE_DIRTY="true")
            row["attempts"].append({"number": number, "status": "launching",
                                    "started": _now(),
                                    "output": str(output.relative_to(ROOT)),
                                    "log": str(log.relative_to(ROOT)),
                                    "command": command})
            row["status"] = "running"
            json_dump(state_path, state)
            stream = log.open("w")
            # The child waits for a one-line start gate. If this parent dies
            # before persisting its PID, pipe EOF makes the child exit without
            # touching the output. GNU timeout then enforces the job limit even
            # if the queue process later dies.
            gate = ('IFS= read -r token || exit 0; '
                    '[ "$token" = GO ] || exit 0; '
                    'exec timeout --signal=TERM --kill-after=5s "$@"')
            process = subprocess.Popen(["bash", "-c", gate, "arise-gate",
                                        f"{timeout}s", *command],
                                       cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, text=True)
            row["attempts"][-1]["pid"] = process.pid
            row["attempts"][-1]["status"] = "running"
            active[job_id] = (process, stream)
            json_dump(state_path, state)
            assert process.stdin is not None
            process.stdin.write("GO\n")
            process.stdin.close()
            print(json.dumps({"launched": job_id, "pid": process.pid,
                              "workers_active": len(active)}), flush=True)
        time.sleep(2 if active else 20)
    state["status"] = "completed"
    state["ended"] = _now()
    state["charged_core_or_gpu_hours"] = charged / 3600
    json_dump(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("manifest", "run"))
    parser.add_argument("--dataset", choices=("tt", "aiops"))
    parser.add_argument("--source-commit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "manifest":
        if args.dataset is None or args.source_commit is None:
            parser.error("manifest mode requires --dataset and --source-commit")
        json_dump(args.manifest, make_manifest(args.dataset, args.source_commit))
        print(args.manifest)
    else:
        state = run(args.manifest, workers=args.workers)
        print(json.dumps({"status": state["status"],
                          "charged_hours": state["charged_core_or_gpu_hours"]}))


if __name__ == "__main__":
    main()
