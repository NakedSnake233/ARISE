"""Post-hoc fault-type RCA latency and LLM token accounting.

Uses the frozen test-cohort order, local timing observations, and retained API
call usage. Fault labels are used only for this analysis, never for diagnosis.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fse2027_baseline_common import ROOT, load_dataset, sha256


LATENCY = ROOT / "artifacts/care/fse2027-latency-20260927"
API = ROOT / "artifacts/care/fse2027-opencode/opencode-go-20260927"
OUTPUT = ROOT / "artifacts/care/fse2027-fault-type-efficiency-20260928"
DATASETS = ("trainticket", "aiops2022")
LOCAL_METHODS = (
    "MicroScope", "MicroRCA", "SBLD", "LogFaultFlagger", "TraceAnomaly",
    "MicroRank", "Nezha", "Eadro", "MULAN", "CARE",
)
LLM_METHODS = ("CARE+GPT", "ReAct", "mABC")
DISPLAY = {
    "cpu_load": "CPU load", "disk_io_load": "Disk I/O", "memory_ramp": "Memory ramp",
    "network_delay": "Network delay", "packet_loss": "Packet loss",
    "k8s容器cpu负载": "CPU load", "k8s容器内存负载": "Memory load",
    "k8s容器写io负载": "Write I/O", "k8s容器读io负载": "Read I/O",
    "k8s容器网络丢包": "Packet loss", "k8s容器网络延迟": "Network delay",
    "k8s容器网络资源包损坏": "Packet corruption",
    "k8s容器网络资源包重复发送": "Packet duplication",
    "k8s容器进程中止": "Process abort",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float] | list[int], digits: int = 6) -> dict:
    a = np.asarray(values, dtype=float)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("empty/non-finite stratum")
    return {
        "count": int(len(a)), "mean": round(float(a.mean()), digits),
        "median": round(float(np.median(a)), digits),
        "p95": round(float(np.quantile(a, .95)), digits),
        "sum": round(float(a.sum()), digits),
    }


def render_report(results: dict) -> str:
    lines = [
        "# Fault-type RCA latency and LLM token use",
        "",
        "Frozen TrainTicket and AIOps2022 test cohorts; generated from the "
        "per-incident timing records and provider-reported API usage. Fault labels "
        "are used only for post-hoc grouping. Times are mean / pooled p95. "
        "Five-seed and three-repeat local measurements, and three API runs, "
        "reuse the same unique incidents; they do not enlarge the fault-type sample size.",
        "",
        "Local time begins from prepared numeric inputs. API time sums successful "
        "HTTP calls and excludes local work. Token cells give mean total "
        "[input / output] per incident--run; reasoning tokens are included in "
        "output tokens. Displayed token means are rounded independently, so "
        "the displayed components may differ from the total by one. "
        "Full per-call and per-decision values are in the CSV files.",
        "",
    ]
    for dataset in DATASETS:
        types = results["datasets"][dataset]
        title = "TrainTicket" if dataset == "trainticket" else "AIOps2022"
        lines += [f"## {title}", "", "Local scoring latency (ms):", ""]
        lines.append("| Fault type (unique n) | " + " | ".join(LOCAL_METHODS) + " |")
        lines.append("|---" + "|---:" * len(LOCAL_METHODS) + "|")
        for item in types.values():
            label = f"{item['display']} ({item['unique_incidents']})"
            cells = [
                f"{item['local'][method]['latency_ms']['mean']:.2f} / {item['local'][method]['latency_ms']['p95']:.2f}"
                for method in LOCAL_METHODS
            ]
            lines.append("| " + " | ".join([label, *cells]) + " |")
        lines += ["", "LLM HTTP latency (s):", ""]
        lines.append("| Fault type (unique n) | " + " | ".join(LLM_METHODS) + " |")
        lines.append("|---" + "|---:" * len(LLM_METHODS) + "|")
        for item in types.values():
            label = f"{item['display']} ({item['unique_incidents']})"
            cells = [
                f"{item['llm'][method]['wall_seconds']['mean']:.2f} / {item['llm'][method]['wall_seconds']['p95']:.2f}"
                for method in LLM_METHODS
            ]
            lines.append("| " + " | ".join([label, *cells]) + " |")
        lines += ["", "LLM tokens per incident--run, mean total [input / output]:", ""]
        lines.append("| Fault type (unique n) | " + " | ".join(LLM_METHODS) + " |")
        lines.append("|---" + "|---:" * len(LLM_METHODS) + "|")
        for item in types.values():
            label = f"{item['display']} ({item['unique_incidents']})"
            cells = []
            for method in LLM_METHODS:
                values = item["llm"][method]
                cells.append(
                    f"{values['total_tokens']['mean']:,.0f} "
                    f"[{values['input_tokens']['mean']:,.0f} / {values['output_tokens']['mean']:,.0f}]"
                )
            lines.append("| " + " | ".join([label, *cells]) + " |")
        lines += [""]
        if any(item["unique_incidents"] <= 2 for item in types.values()):
            lines += ["Types with one or two unique incidents have especially unstable p95 values.", ""]
    return "\n".join(lines)


def render_supplement(results: dict) -> str:
    lines = [
        r"\documentclass[acmsmall,screen,review,anonymous]{acmart}",
        r"\usepackage{booktabs,float}",
        r"\usepackage[T1]{fontenc}",
        r"\setcopyright{none}",
        r"\settopmatter{printfolios=true,printacmref=false,printccs=false}",
        r"\renewcommand\footnotetextcopyrightpermission[1]{}",
        r"\acmYear{2027}\acmMonth{7}\acmDOI{}\acmISBN{}",
        r"\setlength{\intextsep}{4pt}",
        r"\setlength{\floatsep}{4pt}",
        r"\setlength{\abovecaptionskip}{2pt}",
        r"\setlength{\belowcaptionskip}{2pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\title{Supplementary Fault-Type Efficiency Results}",
        r"\author{Anonymous Author(s)}",
        r"\affiliation{\institution{Anonymous Institution}\country{Anonymous}}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Measurement Scope}",
        "The strata use the frozen test incidents. $n$ counts unique incidents, "
        "while local observations repeat each incident three times per seed "
        "(five seeds for stochastic methods) and API decisions repeat it in "
        "three runs. Values are descriptive mean/p95; repeated timings are "
        "not independent incidents. Local timing starts from prepared numeric "
        "inputs. API time sums HTTP calls and excludes local work. Token values "
        "are per decision and use provider-reported total [input/output] tokens; "
        "reasoning tokens are part of output. Displayed token components are "
        "rounded independently and may differ from the displayed total by one.",
        "",
    ]
    for dataset in DATASETS:
        title = "TrainTicket" if dataset == "trainticket" else "AIOps2022"
        types = results["datasets"][dataset]
        lines += [r"\section{" + title + "}", ""]
        for part, methods in enumerate((LOCAL_METHODS[:5], LOCAL_METHODS[5:]), start=1):
            lines += [
                r"\begin{table}[H]\centering\footnotesize",
                rf"\caption{{Local scoring latency by fault type, part {part} (ms, mean/p95).}}",
                r"\setlength{\tabcolsep}{3pt}",
                r"\begin{tabular}{@{}lr" + "c" * len(methods) + r"@{}}",
                r"\toprule",
                "Fault type & $n$ & " + " & ".join(methods) + r" \\",
                r"\midrule",
            ]
            for item in types.values():
                cells = [
                    f"{item['local'][method]['latency_ms']['mean']:.2f}/{item['local'][method]['latency_ms']['p95']:.2f}"
                    for method in methods
                ]
                lines.append(f"{item['display']} & {item['unique_incidents']} & " + " & ".join(cells) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        lines += [
            r"\begin{table}[H]\centering\footnotesize",
            r"\caption{LLM HTTP latency by fault type (s, mean/p95).}",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{@{}lrccc@{}}",
            r"\toprule",
            "Fault type & $n$ & " + " & ".join(LLM_METHODS) + r" \\",
            r"\midrule",
        ]
        for item in types.values():
            cells = [
                f"{item['llm'][method]['wall_seconds']['mean']:.2f}/{item['llm'][method]['wall_seconds']['p95']:.2f}"
                for method in LLM_METHODS
            ]
            lines.append(f"{item['display']} & {item['unique_incidents']} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        lines += [
            r"\begin{table}[H]\centering\footnotesize",
            r"\caption{Mean LLM tokens per decision by fault type: total [input/output].}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}lrccc@{}}",
            r"\toprule",
            "Fault type & $n$ & " + " & ".join(LLM_METHODS) + r" \\",
            r"\midrule",
        ]
        for item in types.values():
            cells = []
            for method in LLM_METHODS:
                x = item["llm"][method]
                cells.append(
                    f"{x['total_tokens']['mean']:,.0f} "
                    f"[{x['input_tokens']['mean']:,.0f}/{x['output_tokens']['mean']:,.0f}]"
                )
            lines.append(f"{item['display']} & {item['unique_incidents']} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    lines += [r"\end{document}", ""]
    return "\n".join(lines)


def load_incidents():
    bundles = {name: load_dataset(name) for name in DATASETS}
    positions = {}
    for dataset, bundle in bundles.items():
        positions[dataset] = [
            {
                "incident_id": bundle.rows[int(index)]["incident_id"],
                "fault_type": bundle.rows[int(index)]["fault_type"],
                "fault_type_display": DISPLAY[bundle.rows[int(index)]["fault_type"]],
            }
            for index in bundle.test
        ]
    return bundles, positions


def local_measurements(positions: dict) -> list[dict]:
    records = json.loads((LATENCY / "results.json").read_text())
    if len(records) != 52:
        raise AssertionError(f"Expected 52 method/seed records, got {len(records)}")
    rows = []
    for record in records:
        dataset, method = record["dataset"], record["method"]
        if method not in LOCAL_METHODS:
            raise AssertionError(method)
        incidents = positions[dataset]
        repeats = int(record["repeats"])
        times = record["latency_ms"]
        if len(times) != len(incidents) * repeats:
            raise AssertionError((dataset, method, len(times), len(incidents), repeats))
        if record["validation"]["top1_agreement"] != 1.0:
            raise AssertionError((dataset, method, "score parity"))
        for position, incident in enumerate(incidents):
            for repeat in range(repeats):
                rows.append({
                    "dataset": dataset, "method": method,
                    "incident_position": position, **incident,
                    "seed": record["seed"] if record["seed"] is not None else "deterministic",
                    "repeat": repeat,
                    "latency_ms": times[position * repeats + repeat],
                })
    if len(rows) != 4368:
        raise AssertionError(("local observation count", len(rows)))
    return rows


def llm_usage(positions: dict):
    decisions, calls = [], []
    accounting = json.loads((API / "llm_accounting.json").read_text())
    for dataset in DATASETS:
        incidents = positions[dataset]
        for method in LLM_METHODS:
            for run in range(3):
                audit_path = API / dataset / method / f"api-run-{run}" / "llm_audit.json"
                audit = json.loads(audit_path.read_text())
                if len(audit) != len(incidents):
                    raise AssertionError((dataset, method, run, len(audit)))
                for position, (entry, incident) in enumerate(zip(audit, incidents)):
                    if int(entry["test_position"]) != position:
                        raise AssertionError((dataset, method, run, position))
                    totals = {"input_tokens": 0, "output_tokens": 0,
                              "reasoning_tokens": 0, "total_tokens": 0, "wall_seconds": 0.0}
                    for call_index, relative_path in enumerate(entry["call_paths"]):
                        call_path = ROOT / relative_path
                        record = json.loads(call_path.read_text())
                        usage = record["provider_response"]["usage"]
                        if not usage:
                            raise AssertionError((relative_path, "missing provider usage"))
                        input_tokens = int(usage["input_tokens"])
                        output_tokens = int(usage["output_tokens"])
                        total_tokens = int(usage["total_tokens"])
                        reasoning_tokens = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
                        if input_tokens + output_tokens != total_tokens or reasoning_tokens > output_tokens:
                            raise AssertionError((relative_path, usage))
                        wall = float(record["wall_seconds"])
                        call = {
                            "dataset": dataset, "method": method, "api_run": run,
                            "incident_position": position, **incident,
                            "call_index": call_index, "call_path": relative_path,
                            "input_tokens": input_tokens, "output_tokens": output_tokens,
                            "reasoning_tokens": reasoning_tokens,
                            "total_tokens": total_tokens, "wall_seconds": wall,
                        }
                        calls.append(call)
                        for name in totals:
                            totals[name] += call[name]
                    decisions.append({
                        "dataset": dataset, "method": method, "api_run": run,
                        "incident_position": position, **incident,
                        "logical_calls": len(entry["call_paths"]), **totals,
                    })
            selected = [row for row in decisions if row["dataset"] == dataset and row["method"] == method]
            expected = accounting["datasets"][dataset]["methods"][method]["cost"]
            for key, original in (("input_tokens", "prompt_tokens"),
                                  ("output_tokens", "completion_tokens"),
                                  ("total_tokens", "total_tokens"),
                                  ("logical_calls", "logical_calls")):
                if sum(row[key] for row in selected) != int(expected[original]):
                    raise AssertionError((dataset, method, key, expected[original]))
            actual_wall = sum(row["wall_seconds"] for row in selected)
            if abs(actual_wall - float(expected["summed_call_wall_seconds"])) > 1e-6:
                raise AssertionError((dataset, method, "wall", actual_wall))
    if len(decisions) != 504 or len(calls) != 1989:
        raise AssertionError(("API counts", len(decisions), len(calls)))
    return decisions, calls


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bundles, positions = load_incidents()
    local = local_measurements(positions)
    decisions, calls = llm_usage(positions)
    write_csv(OUTPUT / "local_measurements.csv", local)
    write_csv(OUTPUT / "llm_decisions.csv", decisions)
    write_csv(OUTPUT / "llm_calls.csv", calls)
    results = {"datasets": {}, "scope": {
        "local": "prepared-tensor loaded-model CPU score time, three repeats per test incident and seed",
        "llm": "sum of successful provider HTTP call wall seconds and provider usage, three API runs",
        "grouping": "frozen test-set fault type; descriptive, not independent samples over seeds/repeats",
    }}
    for dataset in DATASETS:
        dataset_result = {}
        types = sorted({item["fault_type"] for item in positions[dataset]})
        for fault_type in types:
            type_incidents = [item for item in positions[dataset] if item["fault_type"] == fault_type]
            part = {
                "display": DISPLAY[fault_type], "unique_incidents": len(type_incidents),
                "local": {}, "llm": {},
            }
            for method in LOCAL_METHODS:
                values = [float(row["latency_ms"]) for row in local
                          if row["dataset"] == dataset and row["fault_type"] == fault_type and row["method"] == method]
                part["local"][method] = {"latency_ms": describe(values)}
            for method in LLM_METHODS:
                selected = [row for row in decisions if row["dataset"] == dataset
                            and row["fault_type"] == fault_type and row["method"] == method]
                item = {name: describe([row[name] for row in selected]) for name in
                        ("logical_calls", "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "wall_seconds")}
                part["llm"][method] = item
                if item["total_tokens"]["count"] != 3 * len(type_incidents):
                    raise AssertionError((dataset, fault_type, method, "decision count"))
            dataset_result[fault_type] = part
        results["datasets"][dataset] = dataset_result
    results["audit"] = {
        "test_incidents": {name: len(positions[name]) for name in DATASETS},
        "local_measurements": len(local), "llm_decisions": len(decisions),
        "llm_calls": len(calls),
        "source_sha256": {
            "latency_results": sha256(LATENCY / "results.json"),
            "llm_accounting": sha256(API / "llm_accounting.json"),
            **{f"{name}_incidents": sha256(bundles[name].path / "incidents.json") for name in DATASETS},
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    (OUTPUT / "REPORT.md").write_text(render_report(results))
    paper = ROOT / "Paper/fse2027_revision_20260926"
    (paper / "supplement_fault_type_efficiency.tex").write_text(render_supplement(results))
    print(json.dumps(results["audit"], ensure_ascii=False, indent=2))
    for dataset, types in results["datasets"].items():
        print(f"\n{dataset}")
        for item in types.values():
            tokens = {method: round(item["llm"][method]["total_tokens"]["mean"])
                      for method in LLM_METHODS}
            print(f"{item['display']:22s} n={item['unique_incidents']:2d} "
                  f"CARE={item['local']['CARE']['latency_ms']['mean']:.2f} ms "
                  f"tokens={tokens}")


if __name__ == "__main__":
    main()
