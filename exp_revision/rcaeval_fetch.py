"""Download only the author's pinned RE2 Parquet files, one guarded case/job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import requests

from .common import ROOT, config_snapshot, json_dump, sha256
from . import remaining_queue as q

SOURCES = ROOT / "exp_revision/rcaeval_sources.json"
MAX_CASE_BYTES = 440 * 1024**2


def create(smoke=True):
    sources = json.loads(SOURCES.read_text())
    index = ROOT / sources["data"]["index"]
    records = pd.read_parquet(index)
    records = records[records.dataset.isin(sources["datasets"])].sort_values(["dataset", "case"])
    if smoke:
        records = records.groupby("dataset", sort=True).head(1)
    path = q.MANIFESTS / ("40_smoke_re2_download.json" if smoke else "41_full_re2_download.json")
    if path.exists():
        return None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    jobs = []
    for row in records.itertuples():
        dataset = sources["datasets"][row.dataset]
        # The original case name is needed only for acquisition, never a model prompt.
        config = dict(task="T8", component="official_re2_download", dataset=dataset,
            case=row.case, revision=sources["data"]["revision"], witness="file byte counts; not RCA scores")
        jobs.append(dict(id="re2_download_" + row.case, resource="cpu", cores=1, max_attempts=2,
            timeout_seconds=900, source_commit=commit, expected_config=config,
            command=["-m", "exp_revision.rcaeval_fetch", "--case", row.case],
            test_ids=[row.case], score_shape=[1, 1], expected_status=["completed_official_re2_download"]))
    closure = {str(p.relative_to(ROOT)): sha256(p) for p in (SOURCES, index, Path(__file__),
        ROOT / "exp_revision/remaining_queue.py", ROOT / "exp_revision/common.py", ROOT / "exp_revision/disk_guard.py", ROOT / "exp_revision/disk_guard_local.py")}
    json_dump(path, dict(backend="local", phase="pilot_re2_download" if smoke else "official_re2_download",
        source_commit=commit, precharged_hours=q.PRECHARGED, source_sha256=closure, jobs=jobs))
    return path


def download(case, output):
    clock, began = time.perf_counter(), q.now()
    sources = json.loads(SOURCES.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("immutable RE2 attempt exists")
    index = pd.read_parquet(ROOT / sources["data"]["index"])
    entry = index[index.case == case]
    if len(entry) != 1 or entry.iloc[0].dataset not in sources["datasets"]:
        raise ValueError("case is outside the authorized RE2 suite")
    row = entry.iloc[0]
    base = f"https://huggingface.co/datasets/{sources['data']['repository']}/resolve/{sources['data']['revision']}/"
    names = ["metrics.parquet", "inject_time.txt"]
    if row.has_logs:
        names.append("logs.parquet")
    if row.has_traces:
        names.append("traces.parquet")
    folder = output / "raw"
    folder.mkdir(parents=True)
    files, total = {}, 0
    transfers = []
    for name in names:
        target = folder / name
        partial = folder / (name + ".partial")
        for trial in range(3):
            response, written = None, 0
            try:
                response = requests.get(base + case + "/" + name, stream=True, timeout=(30, 90))
                response.raise_for_status()
                if total + int(response.headers.get("content-length", 0)) > MAX_CASE_BYTES:
                    raise ValueError("case exceeds its disk reservation; split acquisition is required")
                with partial.open("xb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        written += len(block)
                        if total + written > MAX_CASE_BYTES:
                            raise ValueError("case exceeds its disk reservation")
                        stream.write(block)
                partial.replace(target)
                total += written
                transfers.append(dict(file=name, attempts=trial + 1, bytes=written))
                break
            except requests.RequestException as error:
                partial.unlink(missing_ok=True)
                status = getattr(error.response, "status_code", None)
                if trial == 2 or (status is not None and status < 500 and status != 429):
                    raise
                print(json.dumps(dict(file=name, network_retry=trial + 1, error=type(error).__name__)), flush=True)
                time.sleep(2 ** (trial + 1))
            finally:
                if response is not None:
                    response.close()
        files[str(target.relative_to(ROOT))] = sha256(target)
    injection = int((folder / "inject_time.txt").read_text().strip())
    if injection != int(row.inject_time):
        raise ValueError("injection time differs from the author's index")
    frame = pd.read_parquet(folder / "metrics.parquet")
    if "time" not in frame or frame.time.duplicated().any() or not frame.time.is_monotonic_increasing:
        raise ValueError("invalid official metric timestamps")
    if not (frame.time < injection).any() or not (frame.time >= injection).any():
        raise ValueError("case lacks pre/post observations")
    np.savez_compressed(output / "scores.npz", scores=np.asarray([[total]], dtype=float), incident_ids=[case])
    config = dict(task="T8", component="official_re2_download", dataset=sources["datasets"][row.dataset],
        case=case, revision=sources["data"]["revision"], witness="file byte counts; not RCA scores")
    meta = config_snapshot("python -m exp_revision.rcaeval_fetch", config)
    meta.update(status="completed_official_re2_download", errors=[], start_time=began, end_time=q.now(),
        elapsed_seconds=time.perf_counter()-clock, bytes=total, input_sha256=files,
        transfers=transfers,
        source_sha256={"exp_revision/rcaeval_fetch.py": sha256(Path(__file__)), "exp_revision/rcaeval_sources.json": sha256(SOURCES)},
        scores_sha256=sha256(output / "scores.npz"))
    json_dump(output / "run_meta.json", meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--create", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--case")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.create:
        print(create(not args.full))
    else:
        download(args.case, args.output)


if __name__ == "__main__":
    main()
