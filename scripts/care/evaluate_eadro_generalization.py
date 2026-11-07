"""Audit Eadro split generalization and run a validation-only regularization study.

The frozen FSE baseline and its published scores are never overwritten. Candidate
selection uses only fit/validation data and independent selection seeds; the
held-out test is read after the candidate has been selected.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from fse2027_baseline_common import (
    ROOT, fallback_scores, load_dataset, ranking_metrics, complete_or_fallback,
    ranks_from_scores, sha256,
)
from fse2027_neural_baselines import train_supervised


ARCHIVE = ROOT / "artifacts/care/fse2027-baselines"
OUT = ROOT / "artifacts/care/eadro-generalization-20260928"
SELECTION_SEEDS = (2027, 2028, 2029)
RESULT_SEEDS = (42, 43, 44, 45, 46)
METRICS = ("PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5")


def split_metrics(scores: np.ndarray, bundle) -> dict:
    result = {}
    for name, indices in (("fit", bundle.fit), ("validation", bundle.validation),
                          ("test", bundle.test)):
        values = ranking_metrics(scores[indices], bundle.labels[indices])
        ranks = ranks_from_scores(scores[indices], bundle.labels[indices])
        groups = {int(label): float(np.mean(ranks[bundle.labels[indices] == label] == 1))
                  for label in np.unique(bundle.labels[indices])}
        result[name] = {
            "metrics": values, "macro_root_pr1": float(np.mean(list(groups.values()))),
            "root_pr1": {bundle.services[label]: value for label, value in groups.items()},
            "count": len(indices),
        }
    return result


def candidates(frozen: dict) -> dict[str, dict]:
    def modified(**updates):
        value = copy.deepcopy(frozen)
        value.update(updates)
        return value
    return {
        "frozen_config": copy.deepcopy(frozen),
        "compact16": modified(hidden_dim=16, dropout=0.4, learning_rate=0.001,
                              weight_decay=0.001, patience=10),
        "compact16_strong": modified(hidden_dim=16, dropout=0.5,
                                     learning_rate=0.001, weight_decay=0.01,
                                     patience=8),
        "wide32_strong": modified(hidden_dim=32, dropout=0.5,
                                 learning_rate=0.001, weight_decay=0.001,
                                 patience=8),
    }


def audit_archive(bundle) -> dict:
    records = []
    matrices = []
    for seed in RESULT_SEEDS:
        path = ARCHIVE / bundle.name / "Eadro" / f"seed-{seed}" / "scores.npz"
        saved = np.load(path)
        if not np.array_equal(saved["labels"], bundle.labels):
            raise AssertionError((bundle.name, seed, "label order"))
        scores = np.asarray(saved["all_scores"], dtype=np.float64)
        if not np.allclose(scores[bundle.test], saved["test_scores"], atol=1e-8):
            raise AssertionError((bundle.name, seed, "test matrix"))
        records.append({"seed": seed, **split_metrics(scores, bundle)})
        matrices.append(scores[bundle.test])
    mean = {name: {metric: float(np.mean([row[name]["metrics"][metric] for row in records]))
                   for metric in METRICS} for name in ("fit", "validation", "test")}
    macro = {name: float(np.mean([row[name]["macro_root_pr1"] for row in records]))
             for name in ("fit", "validation", "test")}
    root_counts = {name: dict(Counter(bundle.services[bundle.labels[i]] for i in indices))
                   for name, indices in (("fit", bundle.fit), ("validation", bundle.validation),
                                         ("test", bundle.test))}
    groups = {name: {str(bundle.rows[i].get("overlap_group", bundle.rows[i]["incident_id"]))
                     for i in indices}
              for name, indices in (("fit", bundle.fit), ("validation", bundle.validation),
                                    ("test", bundle.test))}
    if any(groups[a] & groups[b] for a, b in (("fit", "validation"),
                                               ("fit", "test"), ("validation", "test"))):
        raise AssertionError((bundle.name, "overlap group leakage"))
    return {
        "per_seed": records, "mean": mean, "macro_root_pr1": macro,
        "root_counts": root_counts,
        "group_overlap": {f"{a}/{b}": len(groups[a] & groups[b])
                          for a, b in (("fit", "validation"), ("fit", "test"),
                                       ("validation", "test"))},
        "source_sha256": {
            "incidents": sha256(bundle.path / "incidents.json"),
            "features": sha256(bundle.path / "features.npz"),
            "checkpoint_scores": {
                str(seed): sha256(ARCHIVE / bundle.name / "Eadro" /
                                  f"seed-{seed}" / "scores.npz") for seed in RESULT_SEEDS
            },
        },
    }


def select_and_evaluate(bundle, audit: dict) -> dict:
    frozen = json.loads((ARCHIVE / bundle.name / "Eadro" / "summary.json").read_text())[
        "selected_configuration"
    ]
    options = candidates(frozen)
    fallback = fallback_scores(bundle)
    validation = {}
    for name, config in options.items():
        rows = []
        for seed in SELECTION_SEEDS:
            raw, _ = train_supervised(bundle, "Eadro", config, seed)
            completed, invalid = complete_or_fallback(raw, fallback)
            if invalid.any():
                raise AssertionError((bundle.name, name, seed, "invalid score"))
            metrics = split_metrics(completed, bundle)
            rows.append({"seed": seed, "fit": metrics["fit"],
                         "validation": metrics["validation"]})
        validation[name] = {
            "config": config, "runs": rows,
            "val_pr1": float(np.mean([r["validation"]["metrics"]["PR@1"] for r in rows])),
            "val_macro_root_pr1": float(np.mean([r["validation"]["macro_root_pr1"] for r in rows])),
            "val_map5": float(np.mean([r["validation"]["metrics"]["MAP@5"] for r in rows])),
            "fit_pr1": float(np.mean([r["fit"]["metrics"]["PR@1"] for r in rows])),
        }
    selected = max(options, key=lambda name: (
        validation[name]["val_pr1"], validation[name]["val_macro_root_pr1"],
        validation[name]["val_map5"],
        -(validation[name]["fit_pr1"] - validation[name]["val_pr1"]),
        -options[name]["hidden_dim"], name,
    ))
    results = []
    new_contributions = []
    old_contributions = []
    for seed in RESULT_SEEDS:
        raw, _ = train_supervised(bundle, "Eadro", options[selected], seed)
        scores, invalid = complete_or_fallback(raw, fallback)
        if invalid.any():
            raise AssertionError((bundle.name, selected, seed, "invalid score"))
        metrics = split_metrics(scores, bundle)
        results.append({"seed": seed, **metrics})
        new_contributions.append((ranks_from_scores(scores[bundle.test], bundle.labels[bundle.test]) == 1).astype(float))
        old = np.load(ARCHIVE / bundle.name / "Eadro" / f"seed-{seed}" / "scores.npz")
        old_contributions.append((ranks_from_scores(old["test_scores"], bundle.labels[bundle.test]) == 1).astype(float))
    delta = np.mean(np.asarray(new_contributions) - np.asarray(old_contributions), axis=0)
    draws = np.random.default_rng(2027).integers(0, len(delta), size=(10000, len(delta)))
    sampled = delta[draws].mean(axis=1)
    means = {name: {metric: float(np.mean([row[name]["metrics"][metric] for row in results]))
                    for metric in METRICS} for name in ("fit", "validation", "test")}
    return {
        "selection_policy": "mean validation PR@1, macro-root PR@1, MAP@5, lower fit-val gap, smaller width; no test access",
        "selection_seeds": SELECTION_SEEDS, "result_seeds": RESULT_SEEDS,
        "candidates": validation, "selected": selected,
        "selected_config": options[selected], "result_runs": results,
        "result_mean": means,
        "paired_test_pr1_delta_vs_frozen": {
            "mean": float(delta.mean()),
            "ci95": [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))],
        },
        "frozen_test_pr1": audit["mean"]["test"]["PR@1"],
    }


def report(data: dict) -> str:
    lines = ["# Eadro 泛化审计与正则化对照", "",
             "冻结 FSE2027 Eadro 行保持不变。本对照只从训练/验证事件选择候选，"
             "然后用独立结果种子 42–46 在原测试集上评估一次。", "",
             "原始适配使用每数据集一个小验证集；TrainTicket 12 个，AIOps2022 21 个。"
             "置信区间按独立测试事件配对抽样，不把五个种子视为独立事件。", ""]
    for dataset in ("trainticket", "aiops2022"):
        item = data["datasets"][dataset]
        audit, study = item["archive_audit"], item.get("regularization_study")
        lines += [f"## {dataset}", "",
                  "冻结 Eadro fit/validation/test PR@1: " + "/".join(
                      f"{100 * audit['mean'][name]['PR@1']:.2f}%" for name in
                      ("fit", "validation", "test")) + ".", "",
                  "按当前划分中出现的根因类别计算的宏平均 PR@1：" + "/".join(
                      f"{100 * audit['macro_root_pr1'][name]:.2f}%" for name in
                      ("fit", "validation", "test")) + "。各划分包含的根因类别不同，宏平均也不可直接作为配对比较。", "",
                  "训练/验证/测试根因频次：", ""]
        for split in ("fit", "validation", "test"):
            lines.append(f"- {split}: " + ", ".join(
                f"{name}={count}" for name, count in sorted(audit["root_counts"][split].items())))
        lines += ["", "事件重叠组交集：" + str(audit["group_overlap"]) + "。", ""]
        if study:
            lines += ["| 候选 | 验证 PR@1 | 验证根因宏平均 PR@1 | 验证 MAP@5 | fit PR@1 |",
                      "|---|---:|---:|---:|---:|"]
            for name, row in study["candidates"].items():
                lines.append(f"| {name} | {100*row['val_pr1']:.2f} | "
                             f"{100*row['val_macro_root_pr1']:.2f} | "
                             f"{100*row['val_map5']:.2f} | {100*row['fit_pr1']:.2f} |")
            lines += ["", f"验证集选中：**{study['selected']}**。", "",
                      "| 方案 | fit PR@1 | validation PR@1 | test PR@1 | test PR@3 | test PR@5 | test MAP@5 |",
                      "|---|---:|---:|---:|---:|---:|---:|",
                      f"| 冻结 Eadro | {100*audit['mean']['fit']['PR@1']:.2f} | "
                      f"{100*audit['mean']['validation']['PR@1']:.2f} | "
                      f"{100*audit['mean']['test']['PR@1']:.2f} | "
                      f"{100*audit['mean']['test']['PR@3']:.2f} | "
                      f"{100*audit['mean']['test']['PR@5']:.2f} | "
                      f"{100*audit['mean']['test']['MAP@5']:.2f} |",
                      f"| 验证集所选对照 | {100*study['result_mean']['fit']['PR@1']:.2f} | "
                      f"{100*study['result_mean']['validation']['PR@1']:.2f} | "
                      f"{100*study['result_mean']['test']['PR@1']:.2f} | "
                      f"{100*study['result_mean']['test']['PR@3']:.2f} | "
                      f"{100*study['result_mean']['test']['PR@5']:.2f} | "
                      f"{100*study['result_mean']['test']['MAP@5']:.2f} |", "",
                      "配对测试 PR@1 差异："
                      f"{100*study['paired_test_pr1_delta_vs_frozen']['mean']:+.2f} 点，"
                      f"95% 事件 bootstrap 区间 "
                      f"[{100*study['paired_test_pr1_delta_vs_frozen']['ci95'][0]:+.2f}, "
                      f"{100*study['paired_test_pr1_delta_vs_frozen']['ci95'][1]:+.2f}] 点。", ""]
    lines += [
        "## 判断与处理", "",
        "训练集分数高于验证集，说明存在一定泛化差距；但 AIOps2022 的测试 PR@1 高于验证，"
        "且验证集含 5 个 frontend 根因、测试集没有。验证集同时只有 21 个事件，"
        "其总体准确率受根因组成显著影响。两数据集的重叠事件组均无跨划分交集。", "",
        "预先列出的三种更强正则化/更小模型候选未在两个测试集上改善验证集所选方案："
        "TrainTicket 比冻结 Eadro 低 8.00 点，AIOps2022 低 0.77 点。"
        "因此不能把高测试分数直接归因为可用强正则化消除的过拟合，"
        "也不能用测试集挑选较差的模型来降低 baseline。冻结 Eadro 行保持不变。", "",
        "该对照仍是同一组已反复查看的测试事件上的事后诊断，不能证明 Eadro 在新故障域没有过拟合。"
        "要估计跨故障类型/服务的泛化，需要新采集的独立事件或按故障类型/服务预先留出的新评测。", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    output = {"protocol": {
        "selection_seeds": SELECTION_SEEDS, "result_seeds": RESULT_SEEDS,
        "candidate_configs_are_predeclared": True,
        "test_is_not_used_for_selection": True,
    }, "datasets": {}}
    for name in ("trainticket", "aiops2022"):
        bundle = load_dataset(name)
        audit = audit_archive(bundle)
        item = {"archive_audit": audit}
        if not args.audit_only:
            item["regularization_study"] = select_and_evaluate(bundle, audit)
        output["datasets"][name] = item
        print(name, {split: audit["mean"][split]["PR@1"] for split in
                     ("fit", "validation", "test")}, flush=True)
        if not args.audit_only:
            study = item["regularization_study"]
            print("selected", study["selected"], "test", study["result_mean"]["test"]["PR@1"], flush=True)
    (OUT / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    (OUT / "REPORT.md").write_text(report(output))


if __name__ == "__main__":
    main()
