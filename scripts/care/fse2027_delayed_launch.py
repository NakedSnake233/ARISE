"""Launch a frozen experiment after a UTC time, with durable status and logs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--start-at-utc", required=True)
    args = parser.parse_args()
    target = dt.datetime.fromisoformat(args.start_at_utc.replace("Z", "+00:00"))
    if target.tzinfo is None:
        raise ValueError("--start-at-utc must include a UTC offset")
    directory = ROOT / "artifacts/care/fse2027-opencode" / args.campaign
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "DELAYED_LAUNCH.json"
    state = {"campaign": args.campaign, "launcher_pid": os.getpid(), "status": "scheduled",
             "scheduled_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "start_at_utc": target.astimezone(dt.timezone.utc).isoformat()}
    write(status_path, state)
    while True:
        remaining = (target - dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(60.0, remaining))
    state.update(status="running", started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(status_path, state)
    log_path = directory / "pipeline.log"
    with log_path.open("a") as log:
        completed = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/care/fse2027_llm_budget_pipeline.py"),
             "--campaign", args.campaign], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    state.update(status="complete" if completed.returncode == 0 else "failed",
                 returncode=completed.returncode,
                 finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 pipeline_log=str(log_path.relative_to(ROOT)))
    write(status_path, state)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
