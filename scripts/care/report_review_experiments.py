"""Aggregate the frozen CARE reviewer-response run into auditable tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5")


def load(path: Path):
    return json.loads(path.read_text())


def rank_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.asarray(
        [int(np.flatnonzero(row == label)[0]) for row, label in zip(order, labels)]
    )
    result = {}
    for k in (1, 3, 5):
        hit = ranks < min(k, scores.shape[1])
        result[f"PR@{k}"] = float(hit.mean())
        if k > 1:
            result[f"MAP@{k}"] = float(
                np.where(hit, 1.0 / (ranks + 1), 0.0).mean()
            )
    return result


def summarize(records: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    records = list(records)
    if not records:
        raise ValueError("Cannot summarize an empty record list")
    result = {}
    for metric in METRICS:
        values = np.asarray([record[metric] for record in records], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "n": int(len(values)),
        }
    return result


def format_metric(summary: dict, metric: str) -> str:
    item = summary[metric]
    return f"{100 * item['mean']:.2f} ± {100 * item['std']:.2f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(prefix: dict, summary: dict) -> dict:
    row = dict(prefix)
    for metric, values in summary.items():
        for statistic, value in values.items():
            row[f"{metric}_{statistic}"] = value
    return row


def seed_average(records: list[dict], group_keys: tuple[str, ...], metric_key: str):
    """Average repeated masks within seed, then summarize across model seeds."""
    grouped = defaultdict(lambda: defaultdict(list))
    for seed, seed_records in records:
        for record in seed_records:
            key = tuple(record[name] for name in group_keys)
            grouped[key][seed].append(record)
    output = {}
    for key, by_seed in grouped.items():
        seed_metrics = []
        auxiliaries = []
        for values in by_seed.values():
            seed_metrics.append(
                {
                    metric: float(np.mean([item[metric_key][metric] for item in values]))
                    for metric in METRICS
                }
            )
            auxiliaries.extend(values)
        output[key] = {
            "metrics": summarize(seed_metrics),
            "records": len(auxiliaries),
            "model_seeds": len(by_seed),
            "raw": auxiliaries,
        }
    return output


def aggregate(output: Path, config: dict) -> dict:
    expected_seeds = config["seeds"]
    aggregate_result = {
        "protocol": load(output / "protocol.json"),
        "status": load(output / "status.json"),
        "datasets": {},
        "llm_costs": load(output / "llm_costs.json"),
    }
    if aggregate_result["status"]["state"] != "complete":
        raise RuntimeError("Refusing to report an incomplete experiment")
    supplementary_path = (
        output / "supplementary-isolated-missing-ablation" / "summary.json"
    )
    supplementary = load(supplementary_path) if supplementary_path.exists() else None

    for dataset in config["datasets"]:
        dataset_path = ROOT / config["datasets"][dataset]["data"]
        dataset_metadata = load(dataset_path / "metadata.json")
        dataset_rows = load(dataset_path / "incidents.json")
        seed_results = [
            load(output / dataset / f"seed-{seed}" / "result.json")
            for seed in expected_seeds
        ]
        if [record["seed"] for record in seed_results] != expected_seeds:
            raise ValueError(f"Unexpected seed coverage for {dataset}")

        main = {
            method: summarize(
                record["main"][method]["metrics"] for record in seed_results
            )
            for method in seed_results[0]["main"]
            if isinstance(seed_results[0]["main"][method], dict)
            and "metrics" in seed_results[0]["main"][method]
        }
        modules = {
            method: summarize(
                record["module_ablations"][method]["metrics"]
                for record in seed_results
            )
            for method in seed_results[0]["module_ablations"]
        }
        supplementary_missing = None
        if supplementary is not None:
            isolated_records = supplementary["datasets"][dataset]
            if [record["seed"] for record in isolated_records] != expected_seeds:
                raise ValueError(
                    f"Unexpected isolated-ablation seed coverage for {dataset}"
                )
            modules["no_missing_augmentation_isolated"] = summarize(
                record["full"] for record in isolated_records
            )
            supplementary_missing = {"complete": {}, "local": {}}
            for modality in isolated_records[0]["complete_missing"]:
                supplementary_missing["complete"][modality] = {
                    variant: {
                        scope: summarize(
                            record["complete_missing"][modality][variant][scope]
                            for record in isolated_records
                        )
                        for scope in ("correction_only", "method")
                    }
                    for variant in ("with_flags", "without_flags")
                }
            for modality in ("metric", "log", "trace", "mixed"):
                for rate in config["missingness"]["rates"]:
                    key = f"{modality}|{rate}"
                    selected_by_seed = []
                    for record in isolated_records:
                        selected = [
                            item for item in record["local_missing"]
                            if item["modality"] == modality
                            and item["missing_rate"] == rate
                        ]
                        selected_by_seed.append(selected)
                    entry = {
                        "realized_missing_rate": float(np.mean([
                            item["realized_missing_rate"]
                            for selected in selected_by_seed
                            for item in selected
                        ])),
                        "variants": {},
                    }
                    for variant in ("with_flags", "without_flags"):
                        entry["variants"][variant] = {}
                        for scope in ("correction_only", "method"):
                            per_seed = [
                                {
                                    metric: float(np.mean([
                                        item[variant][scope][metric]
                                        for item in selected
                                    ]))
                                    for metric in METRICS
                                }
                                for selected in selected_by_seed
                            ]
                            entry["variants"][variant][scope] = summarize(per_seed)
                    supplementary_missing["local"][key] = entry
        complete_missing = {
            modality: {
                method: summarize(
                    record["complete_missing"][modality][method]
                    for record in seed_results
                )
                for method in seed_results[0]["complete_missing"][modality]
            }
            for modality in seed_results[0]["complete_missing"]
        }

        local_missing_by_method = {
            method: seed_average(
                [
                    (seed, record["local_missing"])
                    for seed, record in zip(expected_seeds, seed_results)
                ],
                ("modality", "missing_rate"),
                method,
            )
            for method in ("standard_evidence", "robust_correction", "robust_method")
        }
        local_missing = {}
        for key in local_missing_by_method["robust_method"]:
            raw = local_missing_by_method["robust_method"][key]["raw"]
            local_missing[key] = {
                "methods": {
                    method: values[key]["metrics"]
                    for method, values in local_missing_by_method.items()
                },
                "realized_missing_rate": float(
                    np.mean([record["realized_missing_rate"] for record in raw])
                ),
                "records": len(raw),
                "model_seeds": len(expected_seeds),
            }

        evidence_modalities = {
            combination: summarize(
                record["modality_models"]["evidence_only"][combination]
                for record in seed_results
            )
            for combination in seed_results[0]["modality_models"]["evidence_only"]
        }
        end_to_end_modalities = {
            missing: summarize(
                record["modality_models"]["end_to_end_leave_one_out"][missing]
                for record in seed_results
            )
            for missing in seed_results[0]["modality_models"]["end_to_end_leave_one_out"]
        }

        unseen_evidence = summarize(
            record["unseen_fault"]["aggregate"] for record in seed_results
        )
        strict_file = np.load(output / dataset / "strict-unseen-predictions.npz")
        strict_labels = strict_file["labels"]
        strict = {}
        for method in ("base", "without_generic", "robust_method"):
            strict[method] = summarize(
                rank_metrics(strict_file[f"{method}_seed_{seed}"], strict_labels)
                for seed in config["strict_unseen_seeds"]
            )
        strict_records = load(output / dataset / "strict-unseen.json")["records"]
        strict_by_fault = {}
        for fault_type in sorted({record["fault_type"] for record in strict_records}):
            selected = [
                record for record in strict_records if record["fault_type"] == fault_type
            ]
            strict_by_fault[fault_type] = {
                "count": selected[0]["test_count"],
                "overlap_purged": max(
                    record["overlap_purged_count"] for record in selected
                ),
                "base": summarize(record["base"] for record in selected),
                "without_generic": summarize(
                    record["without_generic"] for record in selected
                ),
                "robust_method": summarize(
                    record["robust_method"] for record in selected
                ),
            }

        by_fault = {}
        for fault_type in seed_results[0]["main"]["robust_method"]["by_fault_type"]:
            by_fault[fault_type] = {
                "count": seed_results[0]["main"]["robust_method"]["by_fault_type"][fault_type]["count"],
                "base": summarize(
                    record["main"]["base"]["by_fault_type"][fault_type]["metrics"]
                    for record in seed_results
                ),
                "robust_method": summarize(
                    record["main"]["robust_method"]["by_fault_type"][fault_type]["metrics"]
                    for record in seed_results
                ),
            }

        sensitivity_records = defaultdict(list)
        for seed, record in zip(expected_seeds, seed_results):
            for item in record["sensitivity"]:
                if item["parameter"] == "candidate_and_fusion":
                    key = (item["parameter"], str(item["top_k"]), item["evidence_weight"])
                elif item["parameter"] == "tree_count":
                    key = (item["parameter"], item["tree_count"])
                else:
                    key = (item["parameter"], item["min_samples_leaf"])
                sensitivity_records[key].append(item["metrics"])
        sensitivity = {
            "|".join(map(str, key)): summarize(values)
            for key, values in sensitivity_records.items()
        }
        legacy_rule_records = defaultdict(list)
        llm_simplex_records = defaultdict(list)
        for record in seed_results:
            legacy = record.get("legacy_rule_sensitivity")
            if legacy:
                for item in legacy["weights"]:
                    key = ("rule_weights", item["w_b"], item["w_c"], item["w_r"])
                    legacy_rule_records[key].append(item["metrics"])
                for item in legacy["health_thresholds"]:
                    key = ("health_threshold", item["anomaly_z_threshold"])
                    legacy_rule_records[key].append(item["metrics"])
                for item in legacy["retrieval_hops"]:
                    key = ("retrieval_hops", item["hops"])
                    legacy_rule_records[key].append(item["metrics"])
            for item in record.get("simplex") or []:
                key = ("llm_simplex", item["w_b"], item["w_c"], item["w_r"])
                llm_simplex_records[key].append(item["metrics"])
        legacy_rule_sensitivity = {
            "|".join(map(str, key)): summarize(values)
            for key, values in legacy_rule_records.items()
        }
        llm_simplex = {
            "|".join(map(str, key)): summarize(values)
            for key, values in llm_simplex_records.items()
        }
        graph = seed_average(
            [(seed, record["graph_retention"]) for seed, record in zip(expected_seeds, seed_results)],
            ("retention",),
            "metrics",
        )
        for item in graph.values():
            del item["raw"]
        perturbations = {
            name: summarize(record["perturbations"][name] for record in seed_results)
            for name in seed_results[0]["perturbations"]
        }

        llm = {}
        for method in (
            "llm_only", "robust_no_llm", "robust_plus_llm_25", "deployment_metrics"
        ):
            if method in seed_results[0]["llm"]:
                llm[method] = summarize(record["llm"][method] for record in seed_results)
        llm["deployment_selected_weight"] = seed_results[0]["llm"].get(
            "deployment_selected_weight"
        )
        llm["deployment_llm_calls_per_seed"] = seed_results[0]["llm"].get(
            "deployment_llm_calls"
        )

        paired = [record["main"]["paired_vs_base"] for record in seed_results]
        timing = {
            "robust_ranker_fit_seconds": {
                "mean": float(np.mean([record["main"]["fit_seconds"] for record in seed_results])),
                "std": float(np.std([record["main"]["fit_seconds"] for record in seed_results], ddof=1)),
            },
            "main_evidence_suite_fit_seconds": {
                "mean": float(np.mean([record["main"]["evidence_suite_fit_seconds"] for record in seed_results])),
                "std": float(np.std([record["main"]["evidence_suite_fit_seconds"] for record in seed_results], ddof=1)),
            },
            "loaded_single_incident_latency_ms": {
                stage: {
                    statistic: float(np.mean([
                        record["main"]["latency"][stage][statistic]
                        for record in seed_results
                    ]))
                    for statistic in ("mean_ms", "p95_ms")
                }
                for stage in ("base", "evidence_and_fusion", "total")
            },
            "torch_cpu_threads": seed_results[0]["main"]["latency"]["torch_cpu_threads"],
        }
        base_evaluations = [
            load(output / "base" / dataset / f"seed-{seed}" / "evaluation.json")
            for seed in expected_seeds
        ]
        timing["base_training_seconds"] = {
            "mean": float(np.mean([item["training_seconds"] for item in base_evaluations])),
            "std": float(np.std([item["training_seconds"] for item in base_evaluations], ddof=1)),
        }
        gpu_values = [
            item["peak_gpu_memory_bytes"]
            for item in base_evaluations
            if item["peak_gpu_memory_bytes"] is not None
        ]
        timing["base_peak_gpu_memory_bytes"] = max(gpu_values) if gpu_values else None

        aggregate_result["datasets"][dataset] = {
            "data_scope": {
                "counts": dataset_metadata.get("counts", {}),
                "candidate_services": len(dataset_metadata.get("services", [])),
                "observed_root_services": len(
                    {row["root_service"] for row in dataset_rows}
                ),
                "fault_types": len({row["fault_type"] for row in dataset_rows}),
                "limitations": dataset_metadata.get("limitations", []),
                "split_protocol": dataset_metadata.get("split_protocol"),
            },
            "seeds": expected_seeds,
            "main": main,
            "candidate_recall": {
                key: {
                    "mean": float(np.mean([
                        record["main"]["candidate_recall"][key]
                        for record in seed_results
                    ])),
                    "std": float(np.std([
                        record["main"]["candidate_recall"][key]
                        for record in seed_results
                    ], ddof=1)),
                }
                for key in seed_results[0]["main"]["candidate_recall"]
            },
            "module_ablations": modules,
            "complete_missing": complete_missing,
            "local_missing": {"|".join(map(str, key)): value for key, value in local_missing.items()},
            "evidence_modalities": evidence_modalities,
            "end_to_end_modality_leave_one_out": end_to_end_modalities,
            "unseen_type_evidence_only": unseen_evidence,
            "history_type_holdout": strict,
            "history_type_by_fault": strict_by_fault,
            "by_fault_type": by_fault,
            "sensitivity": sensitivity,
            "legacy_rule_sensitivity": legacy_rule_sensitivity,
            "llm_simplex": llm_simplex,
            "graph_retention": {str(key[0]): value for key, value in graph.items()},
            "perturbations": perturbations,
            "llm": llm,
            "paired_vs_base": paired,
            "timing": timing,
            "auditability": load(output / dataset / "auditability.json"),
            "supplementary_isolated_missing_ablation": (
                {
                    "parent_protocol_sha256": supplementary[
                        "parent_protocol_sha256"
                    ],
                    "source_sha256": supplementary["source_sha256"],
                    "results": supplementary_missing,
                }
                if supplementary is not None
                else None
            ),
        }
    return aggregate_result


def export_tables(result: dict, tables: Path) -> None:
    main_rows = []
    module_rows = []
    fault_rows = []
    modality_rows = []
    missing_rows = []
    strict_rows = []
    sensitivity_rows = []
    module_labels = {
        "no_missing_augmentation": (
            "no augmentation + no availability flags (legacy two-factor cell)"
        ),
        "no_missing_augmentation_isolated": (
            "no augmentation, availability flags retained (post-protocol supplement)"
        ),
    }
    missing_method_labels = {
        "standard_evidence": (
            "unaugmented specific+type-agnostic evidence correction"
        ),
    }
    for dataset, data in result["datasets"].items():
        for method, summary in data["main"].items():
            main_rows.append(flatten_summary({"dataset": dataset, "method": method}, summary))
        full = data["module_ablations"]["full"]
        for method, summary in data["module_ablations"].items():
            row = flatten_summary(
                {
                    "dataset": dataset,
                    "ablation": method,
                    "display_label": module_labels.get(method, method),
                },
                summary,
            )
            for metric in METRICS:
                row[f"{metric}_delta_vs_full"] = summary[metric]["mean"] - full[metric]["mean"]
            module_rows.append(row)
        for fault, item in data["by_fault_type"].items():
            for method in ("base", "robust_method"):
                fault_rows.append(flatten_summary({"dataset": dataset, "fault_type": fault, "count": item["count"], "method": method}, item[method]))
        for combination, summary in data["evidence_modalities"].items():
            modality_rows.append(flatten_summary({"dataset": dataset, "scope": "evidence_only", "configuration": combination}, summary))
        for missing, summary in data["end_to_end_modality_leave_one_out"].items():
            modality_rows.append(flatten_summary({"dataset": dataset, "scope": "end_to_end_leave_one_out", "configuration": missing}, summary))
        for modality, methods in data["complete_missing"].items():
            for method, summary in methods.items():
                missing_rows.append(flatten_summary({"dataset": dataset, "condition": "complete", "modality": modality, "method": method, "method_label": missing_method_labels.get(method, method)}, summary))
        for key, item in data["local_missing"].items():
            modality, rate = key.split("|")
            for method, summary in item["methods"].items():
                missing_rows.append(flatten_summary({"dataset": dataset, "condition": "local", "modality": modality, "requested_rate": rate, "realized_rate": item["realized_missing_rate"], "method": method, "method_label": missing_method_labels.get(method, method)}, summary))
        supplementary_missing = data["supplementary_isolated_missing_ablation"]
        if supplementary_missing is not None:
            supplementary_missing = supplementary_missing["results"]
            for modality, variants in supplementary_missing["complete"].items():
                for variant, scopes in variants.items():
                    for scope, summary in scopes.items():
                        method = f"no_augmentation_{variant}_{scope}"
                        missing_rows.append(flatten_summary({"dataset": dataset, "condition": "complete", "modality": modality, "method": method, "method_label": method}, summary))
            for key, item in supplementary_missing["local"].items():
                modality, rate = key.split("|")
                for variant, scopes in item["variants"].items():
                    for scope, summary in scopes.items():
                        method = f"no_augmentation_{variant}_{scope}"
                        missing_rows.append(flatten_summary({"dataset": dataset, "condition": "local", "modality": modality, "requested_rate": rate, "realized_rate": item["realized_missing_rate"], "method": method, "method_label": method}, summary))
        for method, summary in data["history_type_holdout"].items():
            strict_rows.append(flatten_summary({"dataset": dataset, "fault_type": "ALL", "method": method}, summary))
        for fault, item in data["history_type_by_fault"].items():
            for method in ("base", "without_generic", "robust_method"):
                strict_rows.append(flatten_summary({"dataset": dataset, "fault_type": fault, "count": item["count"], "overlap_purged": item["overlap_purged"], "method": method}, item[method]))
        for key, summary in data["sensitivity"].items():
            sensitivity_rows.append(flatten_summary({"dataset": dataset, "setting": key}, summary))
        for key, summary in data["legacy_rule_sensitivity"].items():
            sensitivity_rows.append(flatten_summary({"dataset": dataset, "setting": key}, summary))
        for key, summary in data["llm_simplex"].items():
            sensitivity_rows.append(flatten_summary({"dataset": dataset, "setting": key}, summary))
        for retention, item in data["graph_retention"].items():
            sensitivity_rows.append(flatten_summary({"dataset": dataset, "setting": f"graph_retention|{retention}"}, item["metrics"]))
        for setting, summary in data["perturbations"].items():
            sensitivity_rows.append(flatten_summary({"dataset": dataset, "setting": f"perturbation|{setting}"}, summary))
    for name, rows in (
        ("main_results.csv", main_rows),
        ("module_ablations.csv", module_rows),
        ("fault_types.csv", fault_rows),
        ("modalities.csv", modality_rows),
        ("missingness.csv", missing_rows),
        ("history_type_holdout.csv", strict_rows),
        ("sensitivity.csv", sensitivity_rows),
    ):
        write_csv(tables / name, rows)


def render_report(result: dict) -> str:
    lines = [
        "# CARE reviewer-response experiment report",
        "",
        f"Protocol SHA-256: `{result['status']['protocol_sha256']}`. Metrics are stored as proportions; tables display percentage mean ± sample standard deviation across independently trained seeds.",
        "",
        "Main, module, modality, fault-stratum, and paired results use five independently trained seeds (42--46). The stricter history-type holdout uses three seeds (42--44) because it retrains a separate base model for every held fault type. Missingness and graph stress tests average deterministic masks within each model seed before reporting variation across the same five model seeds.",
        "",
        "The frozen test cohorts were inspected in earlier project iterations, so these results are an exploratory controlled supplement rather than a new untouched confirmatory evaluation.",
    ]
    for dataset, data in result["datasets"].items():
        title = "TrainTicket" if dataset == "trainticket" else "AIOps2022"
        scope = data["data_scope"]
        limitation_text = " ".join(scope["limitations"])
        provenance = (
            "This is a newly collected controlled-injection cohort"
            if dataset == "trainticket"
            else "This is a reconstructed subset of the public challenge data"
        )
        lines += [
            "",
            f"## {title}",
            "",
            f"Data scope: {provenance}, with split counts `{scope['counts']}`, {scope['candidate_services']} candidate services, {scope['observed_root_services']} observed root services, and {scope['fault_types']} fault types. {limitation_text}",
            "",
            "### Main results",
            "",
        ]
        main_rows = [
            [method, *[format_metric(summary, metric) for metric in METRICS]]
            for method, summary in data["main"].items()
        ]
        lines.append(markdown_table(["Method", *METRICS], main_rows))
        full = data["main"]["robust_method"]
        base = data["main"]["base"]
        lines += [
            "",
            f"The improved method changes PR@1 by {100 * (full['PR@1']['mean'] - base['PR@1']['mean']):+.2f} points and MAP@5 by {100 * (full['MAP@5']['mean'] - base['MAP@5']['mean']):+.2f} points versus the retrained base. Incident-level paired fixed/broken counts and bootstrap intervals remain in `aggregate.json`.",
            f"Under complete observations, the unaugmented standard-evidence branch is {100 * (data['main']['standard_evidence']['PR@1']['mean'] - full['PR@1']['mean']):+.2f} PR@1 points relative to the robust main method. This clean-data difference is reported as an accuracy/robustness tradeoff rather than hidden by selecting the best test condition.",
            "",
        ]
        paired_rows = []
        for seed, paired in zip(data["seeds"], data["paired_vs_base"]):
            paired_rows.append([
                seed,
                paired["fixed"],
                paired["broken"],
                f"{100 * paired['top1_delta']:+.2f}",
                f"[{100 * paired['paired_bootstrap_95_percentile_ci'][0]:+.2f}, {100 * paired['paired_bootstrap_95_percentile_ci'][1]:+.2f}]",
                f"{paired['mcnemar_exact_two_sided_p']:.4f}",
            ])
        lines.append(markdown_table(
            ["Seed", "Fixed", "Broken", "Top-1 Δ points", "Paired bootstrap 95% CI", "Exact McNemar p"],
            paired_rows,
        ))
        lines += [
            "",
            "The same test incidents recur across seeds, so these rows are not pooled as independent samples and no multiple-seed p-value is reported.",
            "",
            "### Module ablations",
            "",
        ]
        module_rows = []
        ablation_metrics = ("PR@1", "PR@3", "PR@5", "MAP@5")
        module_full = data["module_ablations"]["full"]
        module_labels = {
            "no_missing_augmentation": "no augmentation + no flags (legacy factorial cell)",
            "no_missing_augmentation_isolated": "no augmentation, flags retained (supplement)",
        }
        for method, summary in data["module_ablations"].items():
            module_rows.append([
                module_labels.get(method, method),
                *[format_metric(summary, metric) for metric in ablation_metrics],
                f"{100 * (summary['PR@1']['mean'] - module_full['PR@1']['mean']):+.2f}",
            ])
        lines.append(markdown_table(["Variant", *ablation_metrics, "Δ PR@1 vs full"], module_rows))
        lines.append("")
        lines.append("The isolated augmentation row is a post-protocol factorial completion with its own source hash and the frozen parent protocol hash; it was added after detecting that the legacy cell also removed flags.")

        lines += ["", "### Modality contribution and missing observations", ""]
        fixed_full_key = "+".join(("metric", "log", "trace"))
        fixed_full = data["evidence_modalities"][fixed_full_key]
        evidence_modality_rows = [[
            "all modalities",
            *[format_metric(fixed_full, metric) for metric in ablation_metrics],
            "reference",
        ]]
        for missing in ("metric", "log", "trace"):
            keep = "+".join(
                modality for modality in ("metric", "log", "trace")
                if modality != missing
            )
            summary = data["evidence_modalities"][keep]
            evidence_modality_rows.append([
                f"without {missing}",
                *[format_metric(summary, metric) for metric in ablation_metrics],
                f"{100 * (summary['PR@1']['mean'] - fixed_full['PR@1']['mean']):+.2f} PR@1 points",
            ])
        lines.append("Evidence-branch fixed-availability comparison (same fitting policy):")
        lines.append("")
        lines.append(markdown_table(
            ["Evidence configuration", *ablation_metrics, "Effect"],
            evidence_modality_rows,
        ))
        lines += [
            "",
            "This comparison isolates which telemetry enters the evidence branch. It does not isolate the base model's modality contribution.",
            "",
            "End-to-end retrained deployment regimes:",
            "",
        ]
        modality_rows = [["robust all-modality method", *[format_metric(full, metric) for metric in ablation_metrics], "reference regime"]]
        for missing, summary in data["end_to_end_modality_leave_one_out"].items():
            modality_rows.append([
                f"without {missing}",
                *[format_metric(summary, metric) for metric in ablation_metrics],
                f"{100 * (summary['PR@1']['mean'] - full['PR@1']['mean']):+.2f} PR@1 points",
            ])
        lines.append(markdown_table(["Configuration", *ablation_metrics, "Effect"], modality_rows))
        lines.append("")
        lines.append(
            "The end-to-end rows retrain modality-specific base and evidence models, while the reference also uses missing-modality augmentation. Their differences describe complete deployment regimes and must not be read as isolated causal effects of deleting one modality. In particular, a positive no-log difference does not establish that logs are useless or harmful."
        )
        lines += [
            "",
            "Evidence-only single/pair combinations, full-modality deletion, and service-level random missingness are provided in `tables/modalities.csv` and `tables/missingness.csv`.",
            "",
            "Complete-modality deletion compares end-to-end variants with the same base checkpoint and mask:",
            "",
        ]
        supplemental_missing = data["supplementary_isolated_missing_ablation"]["results"]
        complete_rows = []
        for missing, methods in data["complete_missing"].items():
            variants = (
                ("No augmentation/flags", supplemental_missing["complete"][missing]["without_flags"]["method"]),
                ("No augmentation, flags kept", supplemental_missing["complete"][missing]["with_flags"]["method"]),
                ("Robust augmented method", methods["robust_method"]),
            )
            for label, summary in variants:
                complete_rows.append([missing, label, *[format_metric(summary, metric) for metric in ablation_metrics]])
        lines.append(markdown_table(
            ["Deleted modality", "Variant", *ablation_metrics],
            complete_rows,
        ))
        lines += ["", "For random service-level mixed missingness:", ""]
        local_rows = []
        for requested in (0.25, 0.5, 0.75):
            item = data["local_missing"][f"mixed|{requested}"]
            supplemental_item = supplemental_missing["local"][f"mixed|{requested}"]
            variants = (
                ("No augmentation/flags", supplemental_item["variants"]["without_flags"]["method"]),
                ("No augmentation, flags kept", supplemental_item["variants"]["with_flags"]["method"]),
                ("Robust augmented method", item["methods"]["robust_method"]),
            )
            for label, summary in variants:
                local_rows.append([
                    requested, f"{100 * item['realized_missing_rate']:.1f}%", label,
                    *[format_metric(summary, metric) for metric in ablation_metrics],
                ])
        lines.append(markdown_table(
            ["Requested rate", "Realized rate", "Variant", *ablation_metrics],
            local_rows,
        ))
        lowest = data["local_missing"]["mixed|0.75"]
        lowest_supplemental = supplemental_missing["local"]["mixed|0.75"]
        robust_gain = (
            lowest["methods"]["robust_method"]["PR@1"]["mean"]
            - lowest_supplemental["variants"]["with_flags"]["method"]["PR@1"]["mean"]
        )
        lines += [
            "",
            f"At the highest mixed setting, the mask removes {100 * lowest['realized_missing_rate']:.1f}% of event-service-modality cells after enforcing at least one surviving modality per service. With the base checkpoint and availability flags held fixed, missing-modality augmentation changes end-to-end PR@1 by {100 * robust_gain:+.2f} points.",
            "Missing-modality augmentation is condition dependent: it hurts all three mixed-missingness settings on TrainTicket and the lower two settings on AIOps2022, while helping complete outages and the highest AIOps2022 mixed setting. The deterministic masks and conservative removal of dependent cached summaries are leakage-avoiding stress tests; they do not reproduce raw partial-trace collection or measure production retraining frequency.",
            "",
            "### Historical type generalization",
            "",
        ]
        strict_rows = [
            [method, *[format_metric(summary, metric) for metric in ablation_metrics]]
            for method, summary in data["history_type_holdout"].items()
        ]
        lines.append(markdown_table(["Method", *ablation_metrics], strict_rows))
        lines += [
            "",
            "This is a leave-one-fault-type-out supervised/history-type test over three model seeds. Held-type examples and any other fitting events in the same overlap groups are removed from base training, checkpoint selection, topology construction, evidence fitting, and inner selection. Shared pre-window numeric calibration is retained and may include telemetry overlapping neighboring incidents, and logs are unavailable in this reconstructed holdout path. The result therefore does not establish end-to-end raw-data cold-start performance or log-based transfer to unseen fault types.",
            f"Relative to the matched no-generic branch, the type-agnostic expert changes PR@1 by {100 * (data['history_type_holdout']['robust_method']['PR@1']['mean'] - data['history_type_holdout']['without_generic']['PR@1']['mean']):+.2f} points; its benefit is therefore dataset dependent.",
            "",
            "### Performance by fault type",
            "",
        ]
        fault_rows = []
        for fault, item in data["by_fault_type"].items():
            fault_rows.append([
                fault,
                item["count"],
                format_metric(item["base"], "PR@1"),
                format_metric(item["robust_method"], "PR@1"),
                format_metric(item["robust_method"], "MAP@5"),
            ])
        lines.append(markdown_table(["Fault type", "n", "Base PR@1", "CARE PR@1", "CARE MAP@5"], fault_rows))
        lines.append("")
        lines.append("Fault-type rows with very small n are descriptive and should not be treated as stable subgroup estimates.")

        lines += ["", "### Candidate bottleneck and parameter sensitivity", ""]
        all_key = max(data["candidate_recall"], key=lambda key: int(key))
        candidate_rows = []
        for top_k in (1, 3, 5, "all"):
            recall_key = all_key if top_k == "all" else str(top_k)
            sensitivity_key = f"candidate_and_fusion|{top_k}|0.75"
            candidate_rows.append([
                top_k,
                f"{100 * data['candidate_recall'][recall_key]['mean']:.2f} ± {100 * data['candidate_recall'][recall_key]['std']:.2f}",
                format_metric(data["sensitivity"][sensitivity_key], "PR@1"),
                format_metric(data["sensitivity"][sensitivity_key], "MAP@5"),
            ])
        lines.append(markdown_table(
            ["Candidate k", "Base candidate recall", "Final PR@1 at evidence weight .75", "Final MAP@5"],
            candidate_rows,
        ))
        all_service_weights = {
            weight: data["sensitivity"][f"candidate_and_fusion|all|{weight}"]
            for weight in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        }
        best_all_weight = max(
            all_service_weights,
            key=lambda weight: all_service_weights[weight]["PR@1"]["mean"],
        )
        lines += [
            "",
            f"With k=1, correction cannot change the base Top-1. All-service scoring removes this hard ceiling and is the frozen main setting. The descriptive all-service sweep peaks at evidence weight {best_all_weight} with PR@1 {format_metric(all_service_weights[best_all_weight], 'PR@1')}; the frozen main weight remains 0.75.",
            "",
        ]
        parameter_rows = []
        for value in (64, 128, 256, 512):
            summary = data["sensitivity"][f"tree_count|{value}"]
            parameter_rows.append(["trees", value, format_metric(summary, "PR@1"), format_metric(summary, "MAP@5")])
        for value in (1, 3, 5):
            summary = data["sensitivity"][f"min_samples_leaf|{value}"]
            parameter_rows.append(["min leaf", value, format_metric(summary, "PR@1"), format_metric(summary, "MAP@5")])
        for value in (0.0, 0.25, 0.5, 0.75, 1.0):
            summary = data["graph_retention"][str(value)]["metrics"]
            parameter_rows.append(["graph retention", value, format_metric(summary, "PR@1"), format_metric(summary, "MAP@5")])
        lines.append(markdown_table(["Parameter", "Value", "PR@1", "MAP@5"], parameter_rows))

        perturb_best = max(
            data["perturbations"].items(), key=lambda item: item[1]["PR@1"]["mean"]
        )
        perturb_worst = min(
            data["perturbations"].items(), key=lambda item: item[1]["PR@1"]["mean"]
        )
        health = {
            key: value for key, value in data["legacy_rule_sensitivity"].items()
            if key.startswith("health_threshold|")
        }
        hops = {
            key: value for key, value in data["legacy_rule_sensitivity"].items()
            if key.startswith("retrieval_hops|")
        }
        rule_weights = {
            key: value for key, value in data["legacy_rule_sensitivity"].items()
            if key.startswith("rule_weights|")
        }
        best_rule = max(rule_weights.items(), key=lambda item: item[1]["PR@1"]["mean"])
        worst_rule = min(rule_weights.items(), key=lambda item: item[1]["PR@1"]["mean"])
        lines += [
            "",
            f"Timestamp/granularity perturbations range from {format_metric(perturb_worst[1], 'PR@1')} ({perturb_worst[0]}) to {format_metric(perturb_best[1], 'PR@1')} ({perturb_best[0]}). Trace-shift conditions also remove cached RPC and current-call summaries, so their change cannot be attributed to clock shift alone. Legacy frozen-proposal replay spans PR@1 {100 * worst_rule[1]['PR@1']['mean']:.2f}%–{100 * best_rule[1]['PR@1']['mean']:.2f}% over rule weights (best grid key `{best_rule[0]}`); health thresholds span {100 * min(v['PR@1']['mean'] for v in health.values()):.2f}%–{100 * max(v['PR@1']['mean'] for v in health.values()):.2f}%, and retrieval hops span {100 * min(v['PR@1']['mean'] for v in hops.values()):.2f}%–{100 * max(v['PR@1']['mean'] for v in hops.values()):.2f}%. These are downstream replays of fixed proposals.",
        ]

        lines += ["", "### Cost and latency", ""]
        timing = data["timing"]
        latency = timing["loaded_single_incident_latency_ms"]
        cost_rows = [
            ["base training / seed (s)", f"{timing['base_training_seconds']['mean']:.3f} ± {timing['base_training_seconds']['std']:.3f}"],
            ["main evidence suite fit / seed (s)", f"{timing['main_evidence_suite_fit_seconds']['mean']:.3f} ± {timing['main_evidence_suite_fit_seconds']['std']:.3f}"],
            ["base inference mean / incident (ms)", f"{latency['base']['mean_ms']:.3f}"],
            ["evidence + fusion mean / incident (ms)", f"{latency['evidence_and_fusion']['mean_ms']:.3f}"],
            ["total loaded mean / incident (ms)", f"{latency['total']['mean_ms']:.3f}"],
        ]
        lines.append(markdown_table(["Measurement", "Value"], cost_rows))
        lines.append("")
        lines.append(
            "Loaded latency times model forward passes in memory: the evidence term includes the specific and type-agnostic predictors plus their inner fusion; extraction, disk I/O, and the final scalar base/evidence fusion are excluded."
        )
        audit = data["auditability"]
        lines += ["", "### Structural auditability checks", ""]
        if dataset == "trainticket":
            lines.append(
                f"All {audit['responses']} responses parsed as JSON and obeyed the candidate constraint. All {audit['exact_citation_valid_edges']}/{audit['proposed_edges']} proposed edges cited an exact retrieved propagation fact; the observation gate retained {audit['validly_cited_proposal_gate_pass']} and rejected {audit['validly_cited_proposal_gate_rejected']} ({100 * audit['validly_cited_proposal_gate_rejection_rate']:.2f}%). All {audit['generated_explanations']} recorded root-evidence objects matched the prompt input exactly. These checks establish provenance and faithfulness, while independent path/edge ground truth is unavailable."
            )
        else:
            lines.append(
                f"FPG support is training-only ({audit['fpg_support_incidents']} supporting incidents; {audit['test_incidents_in_fpg_support']} test incidents). The AIOps LLM emits one constrained service token and no path or explanation, so path faithfulness is not applicable. Independent path/edge ground truth is unavailable."
            )

    costs = result["llm_costs"]
    lines += ["", "## LLM cost and deployment decision", ""]
    llm_rows = []
    for variant in costs["variants"]:
        if variant["dataset"] == "trainticket" and variant["variant"] == "compact_json":
            split = variant["splits"].get("test", {})
            llm_rows.append([
                "TrainTicket compact JSON",
                f"{split.get('input_tokens', {}).get('mean', float('nan')):.1f}",
                f"{split.get('output_tokens', {}).get('mean', float('nan')):.1f}",
                f"{split.get('seconds', {}).get('mean', float('nan')):.3f}",
                variant.get("peak_gpu_memory_bytes"),
            ])
        elif "inference_input_tokens" in variant:
            llm_rows.append([
                "TrainTicket archived free form",
                f"{variant['inference_input_tokens']['mean']:.1f}",
                f"{variant['inference_output_tokens']['mean']:.1f}",
                f"{variant['inference_seconds']['mean']:.3f}",
                "not recorded",
            ])
        elif variant["dataset"] == "aiops2022":
            first = next(iter(variant["inference_order_seeds"].values()))
            llm_rows.append([
                "AIOps2022 constrained token",
                f"{first['input_tokens']['mean']:.1f}",
                f"{first['output_tokens']['mean']:.1f}",
                f"{first['seconds']['mean']:.3f}",
                "not recorded",
            ])
    lines.append(markdown_table(["LLM variant", "Input tokens", "Visible output tokens", "Seconds/incident", "Peak GPU bytes"], llm_rows))
    accuracy_rows = []
    for dataset, data in result["datasets"].items():
        title = "TrainTicket" if dataset == "trainticket" else "AIOps2022"
        accuracy_rows.append([
            title,
            format_metric(data["llm"]["llm_only"], "PR@1"),
            format_metric(data["llm"]["robust_no_llm"], "PR@1"),
            format_metric(data["llm"]["robust_plus_llm_25"], "PR@1"),
            format_metric(data["llm"]["deployment_metrics"], "PR@1"),
            data["llm"]["deployment_selected_weight"],
        ])
    lines += ["", markdown_table(
        ["Dataset", "LLM only PR@1", "CARE without LLM", "CARE + 25% LLM", "Frozen-policy PR@1", "Frozen LLM weight"],
        accuracy_rows,
    )]
    training_rows = []
    for variant in costs["variants"]:
        if "training_seconds" not in variant:
            continue
        training_rows.append([
            variant["dataset"],
            f"{variant['training_seconds']:.1f}",
            variant["adapter_parameter_summary"]["parameters"],
            variant["deployment_adapter_bytes"],
            variant["complete_training_artifact_bytes"],
        ])
    lines += ["", markdown_table(
        ["Dataset", "LoRA training seconds", "Adapter parameters", "Deployment adapter bytes", "Complete training artifact bytes"],
        training_rows,
    )]
    lines += [
        "",
        f"The base checkpoint contains {costs['base_model_parameter_summary']['parameters']:,} parameters and occupies {costs['base_model_bytes']:,} bytes on disk. An archived TrainTicket compact-JSON experiment with the older base Top-5 pipeline selected LLM fusion weight 0 on validation. That pre-test operational policy was frozen and transferred to the current robust fusion, the archived free-form TrainTicket branch, and AIOps2022; it is not matched validation tuning of those current branches. The reported main method therefore makes zero LLM calls and gains no test accuracy from the LLM. Each archived LLM score vector is reused across the five independently trained base-model seeds, so seed-wise LLM accuracy variation reflects the base/evidence branch rather than five LLM generations. The nonzero 0.25 fusion is retained only as a descriptive diagnostic. Local execution incurred $0 API charge; electricity and cloud-equivalent prices were not instrumented. Archived decoded outputs omit terminal special tokens that cannot be recovered.",
        "",
        "## Sensitivity, auditability, and raw artifacts",
        "",
        "Candidate/fusion, tree count, minimum leaf size, graph-feature retention, legacy rule weights/threshold/hops, timestamp shifts, and temporal coarsening are test-set descriptive sweeps. Graph retention deletes cached graph/RPC evidence and the base model's historical adjacency; it does not delete the separate historical FPG used by the legacy LLM replay. The `no_graph` module removes graph-derived evidence but retains the base GAT and its adjacency. Their complete values are in `tables/sensitivity.csv`; they do not replace the frozen main configuration.",
        "",
        "The structural audit checks candidate constraints, exact cited facts, observation gating, and evidence-field faithfulness. It does not measure causal path correctness because neither dataset contains independent propagation-edge ground truth.",
        "",
        "Machine-readable aggregates are stored in `aggregate.json`. CSV tables retain means, sample standard deviations, ranges, counts, and deltas. Saved score matrices cover the main, fallback, LLM, perturbation, strict-history, and supplemental end-to-end branches. Other module, primary-missingness, graph, sensitivity/rule, and supplemental correction-only branches retain per-seed metrics and model artifacts but not every score matrix; those rows can be re-aggregated from JSON but cannot all be independently rank-replayed from archived arrays.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/care/review-response-v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/care/review-response-v1")
    )
    args = parser.parse_args()
    config = load(ROOT / args.config)
    output = ROOT / args.output
    result = aggregate(output, config)
    report_dir = output / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "aggregate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    export_tables(result, report_dir / "tables")
    report = render_report(result)
    (report_dir / "REPORT.md").write_text(report)
    (ROOT / "refine-logs/EXPERIMENT_RESULTS.md").write_text(report)
    print(report_dir / "REPORT.md")


if __name__ == "__main__":
    main()
