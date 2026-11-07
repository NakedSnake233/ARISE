"""Bounded, resumable 5-fold × 5-seed S0 evidence-expert training queue."""

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
import zipfile
from pathlib import Path

import numpy as np

from .common import ROOT, SEEDS, json_dump, sha256
from .context_queue import FOLDS
from .fold_data import load_fold


TIMEOUT_SECONDS = 900
KILL_GRACE_SECONDS = 5
THREADS = 4
MAX_ATTEMPTS = 2
DATASET_BUDGET_CORE_HOURS = 100


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _seconds(stamp: str) -> float:
    return dt.datetime.fromisoformat(stamp).timestamp()


def _paths(dataset: str) -> list[Path]:
    paths = [ROOT / name for name in (
        "exp_revision/train_evidence.py", "exp_revision/evidence_fold.py",
        "exp_revision/audit_rpc_alignment.py", "exp_revision/evidence_queue.py",
        "exp_revision/context_queue.py", "exp_revision/fold_data.py",
        "exp_revision/common.py", "exp_revision/metrics.py",
        "scripts/care/robust_evidence.py", "scripts/care/evidence_reranker.py",
        "scripts/care/train_base.py", "configs/care/review-response-v1.json",
        f"splits/{dataset}_kfold5.json", f"splits/{dataset}_unlabeled_events.json",
        f"cache/rpc_alignment/{dataset}.json",
    )]
    for fold in FOLDS:
        evidence_meta = ROOT / f"cache/fold_evidence_v2/{dataset}/fold{fold}/metadata.json"
        paths.extend([
            ROOT / f"splits/{dataset}_training_labels/fold_{fold}.json",
            ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}/features.npz",
            ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}/metadata.json",
            ROOT / f"cache/fold_evidence_v2/{dataset}/fold{fold}/features.npz",
            evidence_meta,
        ])
        # The trainer checks these transitive feature sources at startup.
        # Freeze the same closure so a manifest can stage all required files.
        paths.extend(ROOT / name for name in json.loads(
            evidence_meta.read_text())["input_sha256"])
    return list(dict.fromkeys(paths))


def make_manifest(dataset: str, source_commit: str) -> dict:
    if dataset not in ("tt", "aiops"):
        raise ValueError(dataset)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    if source_commit != actual:
        raise ValueError("evidence source commit differs from local Git HEAD")
    source_files = [str(path.relative_to(ROOT)) for path in _paths(dataset)
                    if path.suffix == ".py" or path == ROOT / "configs/care/review-response-v1.json"]
    for relative in source_files:
        if subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative],
                          cwd=ROOT, capture_output=True).returncode:
            raise ValueError(f"evidence source is not tracked in Git: {relative}")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *source_files],
                      cwd=ROOT).returncode:
        raise ValueError("evidence source differs from the supplied Git HEAD")
    counts = {}
    for fold in FOLDS:
        data = load_fold(dataset, fold, smoke=False)
        counts[str(fold)] = {"train": len(data.train_ids), "test": len(data.test_ids),
                             "services": len(data.services)}
    return {"protocol": "T1_evidence_S0_full_kfold5_v1", "dataset": dataset,
            "source_commit": source_commit, "folds": list(FOLDS),
            "seeds": list(SEEDS), "device": "cpu",
            "budget_core_hours": DATASET_BUDGET_CORE_HOURS,
            "threads": THREADS, "max_attempts": MAX_ATTEMPTS,
            "timeout_seconds": TIMEOUT_SECONDS, "kill_grace_seconds": KILL_GRACE_SECONDS,
            "fold_counts": counts,
            "files_sha256": {str(path.relative_to(ROOT)): sha256(path)
                             for path in _paths(dataset)}}


def verify_manifest(manifest: dict) -> dict[int, object]:
    dataset = manifest["dataset"]
    if dataset not in ("tt", "aiops") or manifest.get("protocol") != "T1_evidence_S0_full_kfold5_v1":
        raise ValueError("wrong evidence queue protocol")
    if manifest.get("folds") != list(FOLDS) or manifest.get("seeds") != list(SEEDS):
        raise ValueError("evidence queue requires the exact 5-fold × 5-seed matrix")
    if (manifest.get("device") != "cpu"
            or manifest.get("budget_core_hours") != DATASET_BUDGET_CORE_HOURS
            or manifest.get("threads") != THREADS
            or manifest.get("max_attempts") != MAX_ATTEMPTS
            or manifest.get("timeout_seconds") != TIMEOUT_SECONDS
            or manifest.get("kill_grace_seconds") != KILL_GRACE_SECONDS):
        raise ValueError("evidence compute limits differ from fixed authorization")
    commit = manifest.get("source_commit", "")
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("invalid evidence source commit")
    expected = {str(path.relative_to(ROOT)): sha256(path) for path in _paths(dataset)}
    if manifest.get("files_sha256") != expected:
        raise ValueError("evidence source or fold input differs from frozen manifest")
    folds = {fold: load_fold(dataset, fold, smoke=False) for fold in FOLDS}
    if manifest["fold_counts"] != {str(fold): {
            "train": len(data.train_ids), "test": len(data.test_ids),
            "services": len(data.services)} for fold, data in folds.items()}:
        raise ValueError("evidence fold counts differ from manifest")
    return folds


def _verified(output: Path, manifest: dict, fold: int, seed: int, data: object) -> bool:
    meta_path = output / "run_meta.json"
    scores_path = output / "scores.npz"
    beta_path = output / "inner_beta.json"
    models = [output / "specific.joblib", output / "generic.joblib"]
    if (not meta_path.is_file() or not scores_path.is_file()
            or not beta_path.is_file() or not all(p.is_file() for p in models)):
        return False
    try:
        meta = json.loads(meta_path.read_text())
        cfg = meta["config"]
        if (meta["status"] != "partial_T1_evidence_S0"
                or meta["git_commit"] != manifest["source_commit"]
                or (cfg["dataset"], cfg["fold"], cfg["seed"], cfg["smoke"])
                != (manifest["dataset"], fold, seed, False)
                or meta["scores_sha256"] != sha256(scores_path)
                or meta["inner_beta_sha256"] != sha256(beta_path)
                or meta["specific_model_sha256"] != sha256(models[0])
                or meta["generic_model_sha256"] != sha256(models[1])
                or not meta.get("start_time") or not meta.get("end_time")):
            return False
        beta = json.loads(beta_path.read_text())
        if (beta["selected_beta"] != meta["selected_beta"]
                or beta["selected_beta"] not in (0, 0.25, 0.5, 0.75, 1)
                or len(beta["grid"]) != 5
                or not all(set(row["fit_ids"] + row["held_ids"] + row["purged_ids"])
                           <= set(data.train_ids) for row in beta["holdouts"])):
            return False
        required = {
            f"cache/fold_preprocessing/{manifest['dataset']}/fold{fold}/features.npz",
            f"cache/fold_preprocessing/{manifest['dataset']}/fold{fold}/metadata.json",
            f"cache/fold_evidence_v2/{manifest['dataset']}/fold{fold}/features.npz",
            f"cache/fold_evidence_v2/{manifest['dataset']}/fold{fold}/metadata.json",
            f"splits/{manifest['dataset']}_training_labels/fold_{fold}.json",
            f"splits/{manifest['dataset']}_kfold5.json",
            "exp_revision/train_evidence.py", "exp_revision/fold_data.py",
            "scripts/care/robust_evidence.py", "scripts/care/evidence_reranker.py",
            "configs/care/review-response-v1.json",
        }
        if set(meta["input_sha256"]) != required:
            return False
        if any(manifest["files_sha256"].get(key) != value
               for key, value in meta["input_sha256"].items()):
            return False
        with np.load(scores_path) as archive:
            if set(archive.files) != {"incident_ids", "specific", "generic", "probabilities"}:
                return False
            if archive["incident_ids"].tolist() != data.test_ids:
                return False
            shape = (len(data.test_ids), len(data.services))
            return all(archive[name].shape == shape
                       and np.isfinite(archive[name]).all()
                       and (archive[name] >= 0).all()
                       and np.allclose(archive[name].sum(axis=1), 1.0, atol=1e-6)
                       for name in ("specific", "generic", "probabilities"))
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
        return False


def _pid_matches(pid: int, output: Path) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return "exp_revision.train_evidence" in command and str(output) in command
    except OSError:
        return False


def _kill_group(pid: int, output: Path) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    if _pid_matches(pid, output):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _retryable(attempt: dict) -> bool:
    if attempt.get("reason") == "job timeout" or attempt.get("returncode") in (124, 137):
        return True
    try:
        tail = (ROOT / attempt["log"]).read_text(errors="replace")[-4000:].lower()
    except OSError:
        return False
    return "out of memory" in tail or "resource temporarily unavailable" in tail


def run(manifest_path: Path, *, workers: int = 8,
        root: Path = ROOT / "results_v2/T1_evidence_full") -> dict:
    manifest = json.loads(manifest_path.read_text())
    folds = verify_manifest(manifest)
    if workers < 1 or workers > 16:
        raise ValueError("evidence CPU workers must be 1–16")
    dataset = manifest["dataset"]
    folder = root / dataset
    folder.mkdir(parents=True, exist_ok=True)
    logs = folder / "logs"
    logs.mkdir(exist_ok=True)
    lock = (folder / ".queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = folder / "queue_state.json"
    manifest_hash = sha256(manifest_path)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["manifest_sha256"] != manifest_hash:
            raise ValueError("existing evidence queue belongs to another manifest")
    else:
        state = {"status": "running", "dataset": dataset,
                 "manifest_sha256": manifest_hash, "created": _now(), "jobs": {}}
    ids = [f"{dataset}_fold{fold}_seed{seed}" for fold in FOLDS for seed in SEEDS]
    for job_id in ids:
        state["jobs"].setdefault(job_id, {"status": "pending", "attempts": []})
    if set(state["jobs"]) != set(ids):
        raise ValueError("evidence queue state has an incomplete matrix")
    # Even two timeout-length attempts for every job in both datasets consume
    # under 101 CPU core-hours, below the user's 200 core-hour total ceiling.
    worst_case = len(ids) * MAX_ATTEMPTS * (TIMEOUT_SECONDS + KILL_GRACE_SECONDS) * THREADS / 3600
    if worst_case > DATASET_BUDGET_CORE_HOURS:
        raise ValueError("evidence matrix could exceed its compute budget")
    active: dict[str, tuple[subprocess.Popen | None, object | None]] = {}
    charged = 0.0
    for job_id in ids:
        row = state["jobs"][job_id]
        fold, seed = map(int, job_id.split("fold", 1)[1].split("_seed"))
        for index, attempt in enumerate(row["attempts"]):
            if attempt["status"] == "launching":
                attempt.update(status="interrupted", ended=_now(), elapsed_seconds=0.0)
            if attempt["status"] == "running":
                output = ROOT / attempt["output"]
                if index != len(row["attempts"]) - 1:
                    row["status"] = "stuck"
                elif _pid_matches(attempt["pid"], output):
                    active[job_id] = (None, None)
                else:
                    attempt.update(status="interrupted", ended=_now(),
                                   elapsed_seconds=min(TIMEOUT_SECONDS + KILL_GRACE_SECONDS,
                                                       max(0, time.time() - _seconds(attempt["started"]))),
                                   billing_basis="watchdog_upper_bound")
            if attempt["status"] != "running":
                charged += THREADS * float(attempt.get("elapsed_seconds", 0))
        if row["status"] != "stuck":
            if not row["attempts"]:
                row["status"] = "pending"
            else:
                last = row["attempts"][-1]
                output = ROOT / last["output"]
                if last["status"] == "running":
                    row["status"] = "running"
                elif _verified(output, manifest, fold, seed, folds[fold]):
                    last["status"] = row["status"] = "completed"
                    row["output"] = last["output"]
                elif (last["status"] == "interrupted" or
                      (last["status"] == "failed" and _retryable(last))) and len(row["attempts"]) < MAX_ATTEMPTS:
                    row["status"] = "pending"
                else:
                    row["status"] = "stuck"
    state["charged_core_hours"] = charged / 3600
    json_dump(state_path, state)
    if any(row["status"] == "stuck" for row in state["jobs"].values()):
        state["status"] = "stuck"
        json_dump(state_path, state)
        raise RuntimeError("evidence queue has a stuck job; inspect immutable attempts")
    halted = False
    while True:
        for job_id, (process, stream) in list(active.items()):
            row = state["jobs"][job_id]
            attempt = row["attempts"][-1]
            fold, seed = map(int, job_id.split("fold", 1)[1].split("_seed"))
            output = ROOT / attempt["output"]
            elapsed = time.time() - _seconds(attempt["started"])
            running = process.poll() is None if process is not None else _pid_matches(attempt["pid"], output)
            if running and elapsed < TIMEOUT_SECONDS:
                continue
            if running:
                _kill_group(attempt["pid"], output)
                attempt["reason"] = "job timeout"
                elapsed = time.time() - _seconds(attempt["started"])
            if stream is not None:
                stream.close()
            valid = _verified(output, manifest, fold, seed, folds[fold])
            attempt.update(status="completed" if valid else "failed",
                           ended=_now(), elapsed_seconds=max(0, elapsed),
                           returncode=process.returncode if process is not None else None)
            charged += THREADS * attempt["elapsed_seconds"]
            row["status"] = "completed" if valid else (
                "pending" if len(row["attempts"]) < MAX_ATTEMPTS and _retryable(attempt)
                else "stuck")
            if valid:
                row["output"] = attempt["output"]
            elif row["status"] == "stuck":
                halted = True
                state["status"] = "stuck"
            del active[job_id]
            state["charged_core_hours"] = charged / 3600
            json_dump(state_path, state)
            print(json.dumps({"job": job_id, "status": row["status"],
                              "charged_core_hours": charged / 3600}), flush=True)
        pending = [job_id for job_id in ids if state["jobs"][job_id]["status"] == "pending"]
        if halted and not active:
            raise RuntimeError("evidence queue stopped after a failed job")
        if not pending and not active:
            break
        while pending and not halted and len(active) < workers:
            if charged + THREADS * (TIMEOUT_SECONDS + KILL_GRACE_SECONDS) * (len(active) + 1) > DATASET_BUDGET_CORE_HOURS * 3600:
                state["status"] = "budget_limited"
                json_dump(state_path, state)
                halted = True
                break
            for relative, digest in manifest["files_sha256"].items():
                if sha256(ROOT / relative) != digest:
                    raise RuntimeError(f"frozen evidence input changed: {relative}")
            job_id = pending.pop(0)
            row = state["jobs"][job_id]
            fold, seed = map(int, job_id.split("fold", 1)[1].split("_seed"))
            number = len(row["attempts"]) + 1
            output = folder / job_id / f"attempt{number}"
            log = logs / f"{job_id}_attempt{number}.log"
            if output.exists() or log.exists():
                raise FileExistsError("immutable evidence attempt path already exists")
            command = [sys.executable, "-m", "exp_revision.train_evidence",
                       "--dataset", dataset, "--fold", str(fold), "--seed", str(seed),
                       "--output", str(output)]
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       ARISE_SOURCE_COMMIT=manifest["source_commit"], ARISE_SOURCE_DIRTY="true")
            row["attempts"].append({"number": number, "status": "launching",
                                    "started": _now(),
                                    "output": str(output.relative_to(ROOT)),
                                    "log": str(log.relative_to(ROOT)), "command": command})
            row["status"] = "running"
            json_dump(state_path, state)
            stream = log.open("w")
            gate = ('IFS= read -r token || exit 0; '
                    '[ "$token" = GO ] || exit 0; '
                    'exec timeout --signal=TERM --kill-after=5s "$@"')
            process = subprocess.Popen(["bash", "-c", gate, "arise-gate",
                                        f"{TIMEOUT_SECONDS}s", *command],
                                       cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, text=True)
            row["attempts"][-1].update(pid=process.pid, status="running")
            active[job_id] = (process, stream)
            json_dump(state_path, state)
            assert process.stdin is not None
            process.stdin.write("GO\n")
            process.stdin.close()
            print(json.dumps({"launched": job_id, "pid": process.pid,
                              "workers_active": len(active)}), flush=True)
        time.sleep(2 if active else 20)
    state.update(status="completed", ended=_now(), charged_core_hours=charged / 3600)
    json_dump(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("manifest", "run"))
    parser.add_argument("--dataset", choices=("tt", "aiops"))
    parser.add_argument("--source-commit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "manifest":
        if not args.dataset or not args.source_commit:
            parser.error("manifest mode requires dataset and source commit")
        json_dump(args.manifest, make_manifest(args.dataset, args.source_commit))
    else:
        state = run(args.manifest, workers=args.workers)
        print(json.dumps({"status": state["status"],
                          "charged_core_hours": state["charged_core_hours"]}))


if __name__ == "__main__":
    main()
