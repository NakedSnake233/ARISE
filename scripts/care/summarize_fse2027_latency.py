"""Summarize frozen local scoring times and per-decision API HTTP times."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fse2027_baseline_common import ROOT, sha256


LOCAL = ROOT / "artifacts/care/fse2027-latency-20260927"
API = ROOT / "artifacts/care/fse2027-opencode/opencode-go-20260927"
ORDER = [
    "MicroScope", "MicroRCA", "SBLD", "LogFaultFlagger", "TraceAnomaly",
    "MicroRank", "Nezha", "Eadro", "MULAN", "CARE", "CARE+GPT", "ReAct", "mABC",
]
DATASETS = ("trainticket", "aiops2022")


def summary(values, scope):
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Missing or nonfinite latency observations")
    return {
        "scope": scope, "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)), "p95_ms": float(np.quantile(array, .95)),
        "observations": int(len(array)),
    }


def main():
    raw_path = LOCAL / "results.json"
    local = json.loads(raw_path.read_text())
    if len(local) != 52:
        raise AssertionError(f"Expected 52 dataset/method/seed rows, got {len(local)}")
    result = {"source_sha256": {"local_results": sha256(raw_path)}, "methods": {}}
    for dataset in DATASETS:
        result["methods"][dataset] = {}
        for method in ORDER[:10]:
            records = [r for r in local if r["dataset"] == dataset and r["method"] == method]
            expected = 5 if method in ("TraceAnomaly", "Eadro", "MULAN", "CARE") else 1
            if len(records) != expected:
                raise AssertionError((dataset, method, len(records), expected))
            if any(r["validation"]["top1_agreement"] != 1 for r in records):
                raise AssertionError((dataset, method, "score parity"))
            times = [t for record in records for t in record["latency_ms"]]
            item = summary(times, "local_loaded_scoring")
            item["seeds"] = [r["seed"] for r in records]
            item["unique_incidents"] = records[0]["incident_count"]
            result["methods"][dataset][method] = item
        for method in ORDER[10:]:
            wall = []
            call_count = []
            run_hashes = {}
            for run in range(3):
                audit_file = API / dataset / method / f"api-run-{run}" / "llm_audit.json"
                audit = json.loads(audit_file.read_text())
                run_hashes[str(run)] = sha256(audit_file)
                run_wall = 0.0
                for incident in audit:
                    calls = incident["call_paths"]
                    seconds = 0.0
                    for relative_path in calls:
                        path = ROOT / relative_path
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        record = json.loads(path.read_text())
                        seconds += float(record["wall_seconds"])
                    wall.append(seconds * 1000)
                    run_wall += seconds
                    call_count.append(len(calls))
                saved = json.loads((audit_file.parent / "result.json").read_text())
                expected_wall = float(saved["provenance"]["cost"]["summed_call_wall_seconds"])
                if abs(run_wall - expected_wall) > 1e-6:
                    raise AssertionError((dataset, method, run, run_wall, expected_wall))
            item = summary(wall, "successful_http_time_only")
            item["run_count"] = 3
            item["unique_incidents"] = int(len(wall) / 3)
            item["mean_logical_calls"] = float(np.mean(call_count))
            item["audit_sha256"] = run_hashes
            result["methods"][dataset][method] = item
    result["protocol"] = json.loads((LOCAL / "provenance.json").read_text())
    (LOCAL / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Method & TrainTicket mean/p95 (ms) & AIOps mean/p95 (ms)")
    for method in ORDER:
        tt = result["methods"]["trainticket"][method]
        ai = result["methods"]["aiops2022"][method]
        print(f"{method:16s} {tt['mean_ms']:9.2f}/{tt['p95_ms']:9.2f} {ai['mean_ms']:9.2f}/{ai['p95_ms']:9.2f}")


if __name__ == "__main__":
    main()
