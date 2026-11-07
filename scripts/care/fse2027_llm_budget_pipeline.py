"""Durable supervisor for the separately frozen output-budget campaign."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
import re
import subprocess
import sys
from pathlib import Path

from fse2027_baseline_common import ROOT, atomic_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="opencode-go-budget-v2-20260927")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.campaign):
        parser.error("campaign must be a simple identifier")
    directory = ROOT / "artifacts/care/fse2027-opencode" / args.campaign
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A supervisor is already running for this campaign")
        state = {"campaign": args.campaign, "pid": os.getpid(), "status": "running",
                 "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "supervisor_sha256": sha256(Path(__file__)), "completed_stages": []}
        commands = [
            ("experiments", ["fse2027_llm_budget_recovery.py", "--campaign", args.campaign]),
            ("aggregate", ["aggregate_fse2027_results.py", "--api-output", str(directory),
                           "--report-output", str(directory)]),
            ("accounting", ["fse2027_llm_accounting.py", "--output", str(directory)]),
            ("cost_and_completion", ["fse2027_opencode_cost.py", "--output", str(directory)]),
            ("offline_replay", ["fse2027_llm_budget_replay.py", "--output", str(directory)]),
            ("staged_tables", ["generate_fse2027_tables.py", "--results-dir", str(directory),
                               "--output-dir", str(directory / "manuscript-tables")]),
        ]
        for stage, command in commands:
            state["current_stage"] = stage
            atomic_json(directory / "PIPELINE_STATUS.json", state)
            print("Starting stage: " + stage, flush=True)
            completed = subprocess.run([sys.executable, "-u", str(ROOT / "scripts/care" / command[0]),
                                        *command[1:]], cwd=ROOT)
            if completed.returncode:
                state.update(status="failed", failed_stage=stage, returncode=completed.returncode)
                atomic_json(directory / "PIPELINE_STATUS.json", state)
                raise SystemExit(completed.returncode)
            state["completed_stages"].append(stage)
        state.update(status="complete", finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(directory / "PIPELINE_STATUS.json", state)
        print("Complete: budget-v2 experiments, metrics, accounting, read-only replay, and tables.", flush=True)


if __name__ == "__main__":
    main()
