"""Aggregate frozen per-service scores into exploratory FSE result tables."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fse2027_baseline_common import OUTPUT, ROOT, METRICS, atomic_json, load_dataset, ranks_from_scores, sha256


PROTOCOL = OUTPUT / "BASELINE_PROTOCOL_FROZEN.json"
EXPECTED_PROTOCOL = "81d0a737e261cddfda665d4a59fb81d567e371e846893dee30f58631e3d676da"
DETERMINISTIC = ["MicroScope", "MicroRCA", "SBLD", "LogFaultFlagger", "MicroRank", "Nezha"]
STOCHASTIC = ["TraceAnomaly", "Eadro", "MULAN"]
API = ["ReAct", "mABC", "CARE+GPT"]


def contributions(scores: list[np.ndarray], labels: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    ranks = np.stack([ranks_from_scores(matrix, labels) for matrix in scores])
    per_run: dict[str, np.ndarray] = {}
    for k in (1, 3, 5):
        per_run[f"PR@{k}"] = (ranks <= k).astype(float)
    for k in (3, 5):
        per_run[f"MAP@{k}"] = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return (
        {metric: values.mean(axis=0) for metric, values in per_run.items()},
        {metric: values.mean(axis=1).tolist() for metric, values in per_run.items()},
    )


def bootstrap_interval(values: np.ndarray, draw: np.ndarray) -> dict[str, float]:
    samples = values[draw].mean(axis=1)
    return {"low": float(np.quantile(samples, 0.025)), "high": float(np.quantile(samples, 0.975))}


def summarize(scores: list[np.ndarray], labels: np.ndarray, rng_seed: int = 2027) -> dict[str, Any]:
    incident, runs = contributions(scores, labels)
    rng = np.random.default_rng(rng_seed)
    draw = rng.integers(0, len(labels), size=(10_000, len(labels)))
    summary: dict[str, Any] = {
        "run_count": len(scores),
        "incident_count": len(labels),
        "bootstrap": "10,000 percentile draws over unique incidents; conditional on retained runs",
        "per_incident_contribution": {metric: value.tolist() for metric, value in incident.items()},
        "per_run": runs,
    }
    for metric in METRICS:
        run_values = np.asarray(runs[metric])
        summary[metric] = {
            "mean": float(incident[metric].mean()),
            "std": float(run_values.std(ddof=1)) if len(run_values) > 1 else None,
            "ci95": bootstrap_interval(incident[metric], draw),
        }
    return summary


def paired(care: dict[str, Any], baseline: dict[str, Any], seed: int = 2027) -> dict[str, Any]:
    n = care["incident_count"]
    if baseline["incident_count"] != n:
        raise ValueError("paired comparison incident count mismatch")
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(10_000, n))
    output = {}
    for metric in METRICS:
        difference = np.asarray(care["per_incident_contribution"][metric]) - np.asarray(
            baseline["per_incident_contribution"][metric]
        )
        interval = bootstrap_interval(difference, draw)
        delta = float(difference.mean())
        output[metric] = {
            "care_minus_baseline": delta,
            "ci95": interval,
            "practically_notable_absolute_5pp": abs(delta) >= 0.05,
            "ordering_uncertain": interval["low"] <= 0 <= interval["high"],
        }
    return output


def load_saved_scores(dataset: str, method: str, runs: list[str], source: Path = OUTPUT) -> list[np.ndarray]:
    if method in API and source.resolve() != OUTPUT.resolve():
        status_file = source / 'status.json'
        if not status_file.is_file():
            return []
        status = json.loads(status_file.read_text())
        if status.get('status') != 'complete':
            return []
        completed = {tuple(item) for item in status.get('completed_run_sets', [])}
        if any((dataset, method, int(run.rsplit('-', 1)[1])) not in completed for run in runs):
            return []
        for run in runs:
            cache_dir = source / 'llm-cache' / method / dataset / run
            cache_files = list(cache_dir.rglob('*.json')) if cache_dir.is_dir() else []
            if not cache_files:
                return []
            for path in cache_files:
                record = json.loads(path.read_text())
                if record.get('error') or record.get('response') is None:
                    return []
    matrices = []
    for run in runs:
        path = source / dataset / method / run / "scores.npz"
        if not path.exists():
            return []
        archive = np.load(path)
        matrices.append(np.asarray(archive["test_scores"], dtype=np.float64))
    return matrices


def load_care_scores(dataset: str, test_indices: np.ndarray) -> list[np.ndarray]:
    matrices = []
    for seed in (42, 43, 44, 45, 46):
        path = ROOT / f"artifacts/care/review-response-v1/{dataset}/seed-{seed}/predictions.npz"
        archive = np.load(path)
        np.testing.assert_array_equal(archive["indices"], test_indices)
        matrices.append(np.asarray(archive["final"], dtype=np.float64))
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-output', type=Path, default=OUTPUT)
    parser.add_argument('--report-output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    api_output = args.api_output.resolve()
    report_output = args.report_output.resolve()
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL:
        raise SystemExit("frozen protocol hash mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    report: dict[str, Any] = {
        "status": "exploratory_post_hoc",
        "protocol_sha256": EXPECTED_PROTOCOL,
        "datasets": {},
        "missing_methods": {},
    }
    if api_output != OUTPUT:
        manifest = api_output / 'API_CAMPAIGN_FROZEN.json'
        if not manifest.is_file():
            raise SystemExit('A new API source requires its frozen campaign manifest')
        report['api_campaign'] = {'source': str(api_output.relative_to(ROOT)),
                                  'manifest_sha256': sha256(manifest)}
    table_rows = []
    for dataset_name in ("trainticket", "aiops2022"):
        bundle = load_dataset(dataset_name)
        labels = bundle.labels[bundle.test]
        scores_by_method: dict[str, list[np.ndarray]] = {}
        for method in DETERMINISTIC:
            scores_by_method[method] = load_saved_scores(dataset_name, method, ["run-0"])
        for method in STOCHASTIC:
            scores_by_method[method] = load_saved_scores(
                dataset_name, method, [f"seed-{seed}" for seed in (42, 43, 44, 45, 46)]
            )
        for method in API:
            scores_by_method[method] = load_saved_scores(
                dataset_name, method, [f"api-run-{run}" for run in (0, 1, 2)], api_output
            )
        scores_by_method["CARE"] = load_care_scores(dataset_name, bundle.test)
        summaries = {
            method: summarize(scores, labels)
            for method, scores in scores_by_method.items()
            if scores
        }
        missing = [method for method, scores in scores_by_method.items() if not scores]
        report["missing_methods"][dataset_name] = missing
        comparisons = {
            method: paired(summaries["CARE"], summary)
            for method, summary in summaries.items()
            if method != "CARE"
        }
        report["datasets"][dataset_name] = {
            "test_incidents": len(labels),
            "methods": summaries,
            "paired_care_comparisons": comparisons,
        }
        for method, summary in summaries.items():
            entry = protocol["methods"][method]
            row = {
                "dataset": dataset_name,
                "method": method,
                "implementation_class": entry["implementation_class"],
                "material_adapter": entry["material_adapter"],
                "runs": summary["run_count"],
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = summary[metric]["mean"]
                row[f"{metric}_std"] = summary[metric]["std"]
                row[f"{metric}_ci_low"] = summary[metric]["ci95"]["low"]
                row[f"{metric}_ci_high"] = summary[metric]["ci95"]["high"]
            table_rows.append(row)
    atomic_json(report_output / "aggregate.json", report)
    output_csv = report_output / "main_results.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps({
        "aggregate": str((report_output / "aggregate.json").relative_to(ROOT)),
        "aggregate_sha256": sha256(report_output / "aggregate.json"),
        "table": str(output_csv.relative_to(ROOT)),
        "rows": len(table_rows),
        "missing_methods": report["missing_methods"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
