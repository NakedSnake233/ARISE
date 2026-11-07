"""Evaluate immutable multi-method score batches, retaining terminal failures."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .common import ROOT, json_dump, load_bundle, sha256
from .missingness import SCENARIOS
from .remaining_queue import verified


def evaluate(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest["source_sha256"].items():
        if sha256(ROOT / relative) != digest: raise ValueError(f"frozen input changed: {relative}")
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    batches, failures = [], {}
    for job in manifest["jobs"]:
        row = state["jobs"][job["id"]]
        failed = row["status"] in ("stuck", "failed")
        if not failed and row["status"] != "completed": raise ValueError(f"unfinished job: {job['id']}")
        relative = row.get("output") or (row["attempts"][-1]["output"] if row.get("attempts") else None)
        folder = ROOT / relative if relative else None
        meta_path = folder / "run_meta.json" if folder else None
        meta = json.loads(meta_path.read_text()) if meta_path and meta_path.exists() else {}
        if not failed and not verified(folder, job): raise ValueError(f"unverified success: {job['id']}")
        cfg = job["expected_config"]
        if meta and any(meta["config"].get(k) != v for k, v in cfg.items()): raise ValueError("score configuration changed")
        for group in ("input_sha256", "source_sha256", "mask_sha256"):
            for name, digest in meta.get(group, {}).items():
                if sha256(ROOT / name) != digest: raise ValueError(f"score input changed: {name}")
        methods = cfg.get("methods", [cfg.get("method", "ARISE")])
        scenarios = cfg.get("scenarios", [f"{s}_{p or 'none'}" for s, p in SCENARIOS])
        scores = np.full(job["score_shape"], np.nan, dtype=np.float32)
        score_path = folder / "scores.npz" if folder else None
        if score_path and score_path.exists():
            if meta.get("scores_sha256") and meta["scores_sha256"] != sha256(score_path): raise ValueError("score archive changed")
            try:
                with np.load(score_path) as archive:
                    if archive["incident_ids"].tolist() != job["test_ids"]: raise ValueError("incident order changed")
                    if archive["scenario_keys"].tolist() != scenarios: raise ValueError("scenario order changed")
                    if "method_names" in archive and archive["method_names"].tolist() != methods: raise ValueError("method order changed")
                    values = archive["scores" if "scores" in archive else "probabilities"]
                    if values.shape != tuple(job["score_shape"]): raise ValueError("score shape changed")
                    scores = values.copy()
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
                if not failed: raise
                failures[job["id"]] = {"invalid_archive": repr(error)}
        if scores.ndim == 3: scores = scores[None]
        if scores.shape[:2] != (len(methods), len(scenarios)): raise ValueError("unexpected score matrix dimensions")
        if failed:
            failures.setdefault(job["id"], {}).update(output=relative, errors=meta.get("errors", []), error_tail=row.get("error_tail", ""))
        batches.append((job, cfg, methods, scenarios, scores))
    # All model/scoring verification precedes this first held-out label read.
    labels = {}
    for dataset in {cfg["dataset"] for _, cfg, _, _, _ in batches}:
        bundle = load_bundle(dataset)
        labels[dataset] = (bundle.services, {r["incident_id"]: (int(bundle.labels[i]), r["fault_type"])
                                          for i, r in enumerate(bundle.rows)})
    with (ROOT / "results_v2/T2/realized_rates.csv").open(newline="") as stream:
        rates = {(r["dataset"], r["incident_id"], int(r["seed"]), r["scenario"], r["scenario_param"]): float(r["realized_rate"])
                 for r in csv.DictReader(stream)}
    rows = []
    for job, cfg, methods, scenarios, scores in batches:
        services, truth = labels[cfg["dataset"]]
        if scores.shape[-1] != len(services): raise ValueError("service count changed")
        for j, method in enumerate(methods):
            for k, key in enumerate(scenarios):
                scenario, param = key.split("_", 1); param = "" if param == "none" else param
                for i, incident in enumerate(job["test_ids"]):
                    root, fault = truth[incident]; valid = bool(np.isfinite(scores[j, k, i]).all())
                    order = np.argsort(-scores[j, k, i], kind="stable") if valid else []
                    task = cfg.get("task", "T2")
                    if task == "T2" and scenario == "S0": task = "T1"
                    rows.append({"task": task, "dataset": cfg["dataset"], "method": method,
                        "split": cfg.get("split", "kfold5"), "fold": cfg["fold"], "seed": cfg["seed"],
                        "scenario": scenario, "scenario_param": param,
                        "realized_rate": rates[(cfg["dataset"], incident, cfg["seed"], scenario, param)],
                        "incident_id": incident, "fault_type": fault, "root_true": services[root],
                        "rank_of_root": float(np.flatnonzero(order == root)[0] + 1) if valid else float("inf"),
                        "top5": json.dumps([services[int(x)] for x in order[:5]], ensure_ascii=False) if valid else "[]",
                        "status": "ok" if valid else "failed", "invalid_by_design": method == "ARISE[V-a]",
                        "runtime_ms": None})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["dataset", "split", "method", "incident_id", "seed", "scenario", "scenario_param"]).any():
        raise ValueError("duplicate predictions")
    output.mkdir(parents=True, exist_ok=True); path = output / "predictions.parquet"
    if path.exists():
        if not pd.read_parquet(path).equals(frame): raise FileExistsError("immutable predictions differ")
    else: frame.to_parquet(path, index=False)
    failure_rates = {"|".join(key): float((group.status != "ok").mean()) for key, group in frame.groupby(
        ["dataset", "method", "scenario", "scenario_param"], dropna=False)}
    summary = {"time": dt.datetime.now(dt.timezone.utc).isoformat(), "phase": manifest["phase"],
        "rows": len(frame), "failed": int((frame.status != "ok").sum()), "terminal_failures": failures,
        "failure_rates": failure_rates, "failure_threshold_exceeded": [key for key, value in failure_rates.items() if value > .1],
        "predictions_sha256": sha256(path), "manifest_sha256": sha256(manifest_path)}
    receipt = output / "evaluation_meta.json"
    if receipt.exists():
        if json.loads(receipt.read_text())["predictions_sha256"] != summary["predictions_sha256"]: raise FileExistsError("immutable receipt differs")
    else: json_dump(receipt, summary)
    if summary["failure_threshold_exceeded"]: raise ValueError("prediction failure rate exceeds 10%: " + ", ".join(summary["failure_threshold_exceeded"]))
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); result = evaluate(a.manifest, a.output); print(json.dumps({k: result[k] for k in ("phase", "rows", "failed", "predictions_sha256")}))


if __name__ == "__main__": main()
