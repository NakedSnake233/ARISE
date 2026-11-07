"""Generate measured API cost and gate accounting from retained request caches."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

from fse2027_baseline_common import OUTPUT, ROOT, atomic_json, load_dataset, ranking_metrics
from fse2027_llm_baselines import care_base_scores, cost_summary


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    campaign_status = OUTPUT / 'status.json'
    complete_sets = None
    if (OUTPUT / 'API_CAMPAIGN_FROZEN.json').is_file():
        complete_sets = {tuple(item) for item in json.loads(campaign_status.read_text()).get('completed_run_sets', [])} if campaign_status.is_file() else set()
    report = {"datasets": {}, "currency_cost": None,
              "timing_scope": "sum of successful HTTP-attempt wall times per incident; excludes retry/backoff and preparation"}
    for dataset in ("trainticket", "aiops2022"):
        bundle = load_dataset(dataset)
        dataset_report = {"methods": {}, "care_ensemble_reference":
                          ranking_metrics(care_base_scores(bundle), bundle.labels[bundle.test])}
        for method in ("ReAct", "mABC", "CARE+GPT"):
            run_audits = []
            missing = []
            failed_cache_records = 0
            missing_response_records = 0
            for run in range(3):
                if complete_sets is not None and (dataset, method, run) not in complete_sets:
                    missing.append(run)
                    continue
                path = OUTPUT / dataset / method / f"api-run-{run}" / "llm_audit.json"
                if not path.exists():
                    missing.append(run)
                    continue
                run_audits.append(json.loads(path.read_text()))
                cache_dir = OUTPUT / 'llm-cache' / method / dataset / f'api-run-{run}'
                cache_files = list(cache_dir.rglob('*.json')) if cache_dir.is_dir() else []
                if not cache_files:
                    missing.append(run)
                    continue
                for cache_file in cache_files:
                    record = json.loads(cache_file.read_text())
                    failed_cache_records += int(bool(record.get('error')))
                    missing_response_records += int(record.get('response') is None)
            if missing or failed_cache_records or missing_response_records:
                dataset_report["methods"][method] = {
                    "status": "incomplete", "required_runs": 3,
                    "retained_run_sets": len(run_audits), "missing_runs": sorted(set(missing)),
                    "failed_cache_records": failed_cache_records,
                    "missing_response_records": missing_response_records,
                }
                continue
            audits = [a for run in run_audits for a in run]
            costs = cost_summary(audits)
            latencies, tokens, calls = [], [], []
            reasons = Counter()
            gate_failures = Counter()
            returned_proposals, changed_proposals, accepted = 0, 0, 0
            for audit in audits:
                latencies.append(sum(float(json.loads((ROOT / path).read_text()).get("wall_seconds") or 0)
                                     for path in audit["call_paths"]))
                tokens.append(sum(int((json.loads((ROOT / path).read_text()).get("response") or {}).get("usage", {}).get("total_tokens") or 0)
                                  for path in audit["call_paths"]))
                calls.append(len(audit["call_paths"]))
                reasons[audit.get("reason") or ("accepted" if audit.get("accepted") else "preserved_ensemble")] += 1
                for gate, passed in audit.get("gates", {}).items():
                    if not passed:
                        gate_failures[gate] += 1
                returned_proposals += bool(audit.get("valid"))
                changed_proposals += bool(audit.get("gates", {}).get("changed_top"))
                accepted += bool(audit.get("accepted"))
            result = {
                "status": "complete",
                "unique_incidents": len(bundle.test), "api_runs": 3,
                "incident_run_decisions": len(audits), "cost": costs,
                "calls_per_incident": float(np.mean(calls)),
                "tokens_per_incident": float(np.mean(tokens)),
                "successful_call_sum_seconds_per_incident": float(np.mean(latencies)),
                "incident_call_sum_seconds_p50": float(np.quantile(latencies, .5)),
                "incident_call_sum_seconds_p95": float(np.quantile(latencies, .95)),
                "invalid_incident_outputs": len(audits) - returned_proposals,
                "valid_role_outputs": sum(int(a.get("valid_role_outputs", 0)) for a in audits),
                "reason_counts": dict(reasons),
                "changed_top_proposals": changed_proposals if method == "CARE+GPT" else None,
                "accepted_corrections": accepted if method == "CARE+GPT" else None,
                "gate_failure_counts": dict(gate_failures) if method == "CARE+GPT" else None,
            }
            dataset_report["methods"][method] = result
        report["datasets"][dataset] = dataset_report
    cache_totals = Counter()
    for path in (OUTPUT / "llm-cache").rglob("*.json"):
        record = json.loads(path.read_text())
        cache_totals["logical_calls"] += 1
        if record.get('submitted') is False:
            cache_totals['unsubmitted_infrastructure_skips'] += 1
        cache_totals["http_attempts"] += int(record.get("attempts") or 0)
        cache_totals["failed_calls" if record.get("error") else "successful_calls"] += 1
        cache_totals["total_tokens"] += int((record.get("response") or {}).get("usage", {}).get("total_tokens") or 0)
    report["entire_campaign_including_incomplete_runs"] = dict(cache_totals)
    atomic_json(OUTPUT / "llm_accounting.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
