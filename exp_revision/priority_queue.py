"""Give ready smoke and pilot jobs priority without changing frozen q controls."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q
from .disk_guard_local import DiskGuard
from . import variant_smoke_repair as recovery


def prioritize(state, pilots):
    def key(item):
        name, row = item; config = row["spec"]["expected_config"]
        order = 0 if config.get("smoke") or name in pilots else 1 if config.get("component") == "train_only_preprocessing" else 2
        return order, name
    state["jobs"] = dict(sorted(state["jobs"].items(), key=key))


def run():
    original = q.load_manifests
    def load(state, cache):
        original(state, cache)
        pilots = set()
        for path in q.MANIFESTS.glob("*.json"):
            manifest = json.loads(path.read_text())
            if "pilot" in manifest["phase"].lower(): pilots.update(job["id"] for job in manifest["jobs"])
        prioritize(state, pilots)
    print(json.dumps(dict(scheduler="priority_queue", source_sha256=sha256(Path(__file__)),
        controls="original queue lock, budgets, retry policy, GO gate and disk guard")), flush=True)
    q.load_manifests = load
    try: return q.run()
    finally: q.load_manifests = original


def check():
    if (ROOT / "manifests/remaining/14_split_tt_variant_smoke.json").exists() and not (ROOT / "results_v2/variant_smoke_repair/receipt.json").exists():
        return recovery.check()
    storage = DiskGuard(ROOT).poll(); busy = False
    with (q.FOLDER / ".queue.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: busy = True
    path = q.FOLDER / "queue_state.json"
    state = json.loads(path.read_text()) if path.exists() else {"status": "new", "jobs": {}, "manifests": {}}
    registered = {str(p.relative_to(ROOT)) for p in q.MANIFESTS.glob("*.json")}
    pending_manifest = registered - set(state["manifests"]); action = "none"
    if not busy and storage["safe_to_launch"] and (state["status"] in ("new", "running", "storage_limited") or pending_manifest):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f"); log = ROOT / f"logs/priority_resume_{stamp}.log"
        with log.open("x") as stream:
            child = subprocess.Popen([str(ROOT / "cache/exp_runtime/venv/bin/python"), "-m", "exp_revision.priority_queue", "--run"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        action = "resume_priority_queue"
    result = dict(time=q.now(), backend="local", status=state["status"], scheduler_alive=busy,
        jobs=dict(Counter(row["status"] for row in state["jobs"].values())), action=action, storage=storage,
        scheduler_source_sha256=sha256(Path(__file__)))
    if action != "none": result.update(pid=child.pid, log=str(log.relative_to(ROOT)))
    json_dump(q.FOLDER / "monitor.json", result)
    with (q.FOLDER / "monitor_history.jsonl").open("a") as history: history.write(json.dumps(result) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", action="store_true"); p.add_argument("--check", action="store_true"); a = p.parse_args()
    result = run() if a.run else check()
    print(json.dumps({k: result[k] for k in ("status", "counts", "jobs", "action", "pid") if k in result}))


if __name__ == "__main__": main()
