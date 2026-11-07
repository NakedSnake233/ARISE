"""Advance completed scientific phases and recover the local queue every20min."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q
from . import adaptive_queue
from . import compose_remaining as composition
from . import collect_remaining as collection
from . import report_queue
from . import artifact_admission


def ready(name, state):
    path = q.MANIFESTS / name
    if not path.exists(): return False
    for job in json.loads(path.read_text())["jobs"]:
        row = state["jobs"].get(job["id"])
        if row is None or row["status"] != "completed" or not q.verified(ROOT / row["output"], job): return False
    return True


def check():
    q.FOLDER.mkdir(parents=True, exist_ok=True)
    with (q.FOLDER / ".stages.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return dict(status="stage_check_already_running")
        state = json.loads((q.FOLDER / "queue_state.json").read_text()); actions, errors = [], []
        def add(target, prerequisites, function):
            if (q.MANIFESTS / target).exists() or not all(ready(name, state) for name in prerequisites): return
            try:
                path = function()
                if path is not None: actions.append(str(path.relative_to(ROOT)))
            except Exception as error: errors.append(dict(stage=target, error=repr(error)))
        add("23_full_t6_reuse.json", ["08_full_t3_evidence.json", "22_smoke_t6_reuse.json"], lambda: composition.create("T6"))
        add("25_full_t5_composition.json", ["20_full_generalization_models.json", "24_smoke_t5_composition.json"], lambda: composition.create("T5"))
        for kind in ("main", "T3", "T5", "T6"):
            add(f"30_collect_{kind}.json", collection.PHASES[kind], lambda kind=kind: collection.create(kind))
        main = state["jobs"].get("collect_main")
        if main and main["status"] == "completed":
            report_rows = [row for row in state["jobs"].values() if row["spec"]["expected_config"].get("component") == "statistical_report"]
            passed_smoke = any(row["status"] == "completed" and row["spec"]["expected_config"]["smoke"] and q.verified(ROOT / row["output"], row["spec"]) for row in report_rows)
            try:
                path = report_queue.create(smoke=not passed_smoke)
                if path is not None: actions.append(str(path.relative_to(ROOT)))
            except Exception as error: errors.append(dict(stage="statistical_report", error=repr(error)))
        # New registered phases join the same single-owner execution queue.
        try:
            actions.extend(str(path.relative_to(ROOT)) for path in artifact_admission.create())
        except Exception as error:
            errors.append(dict(stage="artifact_admission", error=repr(error)))
        monitor = adaptive_queue.check()
        latest = json.loads((q.FOLDER / "queue_state.json").read_text())
        attention = []
        recovery_map = {name: recipe["recovery_job"] for path in q.MANIFESTS.glob("*.json")
            for name, recipe in json.loads(path.read_text()).get("recovery_recipes", {}).items()}
        if latest["status"] == "budget_limited": attention.append("compute budget reached; author decision required")
        for name, row in latest["jobs"].items():
            recovering = name in recovery_map and latest["jobs"].get(recovery_map[name], {}).get("status") != "stuck"
            if row["status"] == "stuck" and not recovering:
                attention.append(dict(job=name, reason=row.get("error_tail", "")[-2000:]))
        result = dict(time=q.now(), status="stage_errors" if errors else "checked", backend="local", created_manifests=actions,
            errors=errors, attention=attention, queue=monitor, controller_sha256=sha256(Path(__file__)))
        json_dump(q.FOLDER / "stages_monitor.json", result)
        with (q.FOLDER / "stages_history.jsonl").open("a") as stream: stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        json_dump(q.FOLDER / "attention.json", dict(time=q.now(), status="needs_attention" if attention or errors else "clear", attention=attention, errors=errors))
        return result


def status(task, dataset, smoke=False):
    state = json.loads((q.FOLDER / "queue_state.json").read_text())
    rows = []
    task_set = {"T5a", "T5b"} if task == "T5" else {task}
    for name, row in state["jobs"].items():
        cfg = row["spec"]["expected_config"]
        if cfg.get("task") in task_set and cfg.get("dataset") == dataset and bool(cfg.get("smoke")) == smoke:
            rows.append(dict(id=name, status=row["status"], output=row.get("output")))
    files = {t: str((ROOT / f"results_v2/{t}/predictions.parquet").relative_to(ROOT))
        for t in task_set if (ROOT / f"results_v2/{t}/predictions.parquet").exists()}
    from collections import Counter
    return dict(task=task, dataset=dataset, smoke=smoke, backend="local", counts=dict(Counter(r["status"] for r in rows)),
        running=[r for r in rows if r["status"] == "running"], predictions=files)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--check", action="store_true"); a = p.parse_args()
    if not a.check: p.error("choose --check")
    result = check(); print(json.dumps({k: result[k] for k in ("status", "created_manifests", "errors", "attention") if k in result}, ensure_ascii=False))


if __name__ == "__main__": main()
