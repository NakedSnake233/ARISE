"""Advance reviewed RE2/official/LLM extensions at the existing 20-minute cadence."""
from __future__ import annotations
import argparse
import fcntl
import json

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q
from . import remaining_stages as old
from . import extended_queue_v2 as queue
from . import t8_admission as admission
from . import circa_variant_admission as variant
from . import re2_data as data
from . import t8_manifest as original
from . import re2_fast_admission as fast
from . import t8_collect as collection


def check():
    with (q.FOLDER / ".stages.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return dict(status="extended_stage_check_already_running")
        actions = []; errors = []; decisions = []

        def add(name, function):
            try:
                path = function()
                if path is not None: actions.append(str(path.relative_to(ROOT)))
            except ValueError as error:
                item = dict(stage=name, reason=str(error))
                if "budget" in str(error) or "author approval" in str(error): decisions.append(item)
                else: errors.append(item)
            except Exception as error: errors.append(dict(stage=name, reason=repr(error)))

        for ds in data.DATASETS:
            add("RE2_full_preprocessing_" + ds, lambda ds=ds: fast.create_dataset(ds))
        add("RE2_full_preprocessing_index", fast.create_index)
        for kind in ("models", "official"):
            for ds in data.DATASETS:
                add(kind + "_pilot_" + ds, lambda kind=kind, ds=ds: admission.create_pilot(kind, ds))
            add(kind + "_pilot_index", lambda kind=kind: admission.create_pilot_index(kind))
        add("RE2_full_models", admission.create_models_full)
        for ds in ("tt", "aiops", *data.DATASETS):
            for method in ("BARO", "CIRCA", "RCD"):
                add("full_" + ds + "_" + method, lambda ds=ds, method=method: admission.create_official_full(ds, method))
            add("variant_pilot_" + ds, lambda ds=ds: variant.create(ds))
            add("variant_full_" + ds, lambda ds=ds: variant.create(ds, False))
        approval = ROOT / "docs/llm_external_authorization.json"
        if approval.exists() and json.loads(approval.read_text()).get("approved") is True:
            if original.complete("43_full_re2_raw.json"):
                add("LLM_legacy_pilot", lambda: admission.create_llm(True, True))
            if original.complete("44_smoke_re2_fold.json"):
                add("LLM_RE2_pilot", lambda: admission.create_llm(True, False))
            add("LLM_full", lambda: admission.create_llm(False))
        else: decisions.append(dict(stage="LLM", reason="explicit external OpenCode authorization pending"))
        for source in sorted(q.MANIFESTS.glob("*.json")):
            if source.name.startswith(("47_full_re2_models", "50_full_official_", "53_full_llm_baselines")):
                add("collect_" + source.stem, lambda source=source: collection.create(source.name))
        # The original completed phases still use the same verified collectors
        # and report builders, under the shared phase lock held by this checker.
        state = admission.state()
        def old_add(target, prerequisites, function):
            if (q.MANIFESTS / target).exists() or not all(old.ready(n, state) for n in prerequisites): return
            add(target, function)
        old_add("23_full_t6_reuse.json", ["08_full_t3_evidence.json", "22_smoke_t6_reuse.json"], lambda: old.composition.create("T6"))
        old_add("25_full_t5_composition.json", ["20_full_generalization_models.json", "24_smoke_t5_composition.json"], lambda: old.composition.create("T5"))
        for kind in ("main", "T3", "T5", "T6"):
            old_add(f"30_collect_{kind}.json", old.collection.PHASES[kind], lambda kind=kind: old.collection.create(kind))
        main = state["jobs"].get("collect_main")
        if main and main["status"] == "completed":
            reports = [r for r in state["jobs"].values() if r["spec"]["expected_config"].get("component") == "statistical_report"]
            passed = any(r["status"] == "completed" and r["spec"]["expected_config"]["smoke"] and q.verified(ROOT / r["output"], r["spec"]) for r in reports)
            add("original_statistical_report", lambda: old.report_queue.create(smoke=not passed))
        try: actions.extend(str(p.relative_to(ROOT)) for p in old.artifact_admission.create())
        except Exception as error: errors.append(dict(stage="artifact_admission", reason=repr(error)))
        monitor = queue.check()
        state = admission.state()
        recovery_map = {name: recipe["recovery_job"] for path in q.MANIFESTS.glob("*.json")
            for name, recipe in json.loads(path.read_text()).get("recovery_recipes", {}).items()}
        if state["status"] == "budget_limited": decisions.append("compute budget reached; author decision required")
        for name, row in state["jobs"].items():
            recovering = name in recovery_map and state["jobs"].get(recovery_map[name], {}).get("status") != "stuck"
            if row["status"] == "stuck" and not recovering:
                decisions.append(dict(job=name, reason=row.get("error_tail", "")[-1500:]))
        current = admission.state()
        # A completed default CIRCA failure remains visible even after a variant
        # was approved. The variant has a different name, inputs and smoke gate.
        result = dict(time=q.now(), status="stage_errors" if errors else "checked", backend="local",
            created_manifests=actions, errors=errors,
            attention=decisions, queue=monitor, jobs=current.get("counts"),
            charged_hours=current.get("charged_hours"), controller_sha256=sha256(ROOT / "exp_revision/extended_stages_v2.py"))
        json_dump(q.FOLDER / "extended_stages_v2_monitor.json", result)
        with (q.FOLDER / "extended_stages_v2_history.jsonl").open("a") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        json_dump(q.FOLDER / "attention.json", dict(time=q.now(), status="needs_attention" if decisions or errors else "clear",
            attention=decisions, errors=errors))
        return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true"); args = parser.parse_args()
    if not args.check: parser.error("choose --check")
    result = check(); print(json.dumps({k: result[k] for k in ("status", "created_manifests", "errors", "attention") if k in result}, ensure_ascii=False))


if __name__ == "__main__": main()
