"""Build a source-linked digest of the finalized public-dataset CARE studies."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/care/CURRENT_EXPERIMENT_SUMMARY_20260928.md"
MANIFEST = ROOT / "artifacts/care/CURRENT_EXPERIMENT_SUMMARY_SOURCES.json"
SOURCES = {
    "matched_main": "artifacts/care/fse2027-opencode/opencode-go-20260927/main_results.csv",
    "review_aggregate": "artifacts/care/review-response-v1/report/aggregate.json",
    "latency": "artifacts/care/fse2027-latency-20260927/summary.json",
    "fault_efficiency": "artifacts/care/fse2027-fault-type-efficiency-20260928/summary.json",
    "api_accounting": "artifacts/care/fse2027-opencode/opencode-go-20260927/llm_accounting.json",
    "api_cost_estimate": "artifacts/care/fse2027-opencode/opencode-go-20260927/COST_ESTIMATE.json",
    "api_experiment_audit": "artifacts/care/fse2027-opencode/opencode-go-20260927/EXPERIMENT_AUDIT.json",
    "paper_table": "Paper/fse2027_revision_20260926/figures/table_main_results.tex",
    "ablation_table": "Paper/fse2027_revision_20260926/figures/table_ablation.tex",
    "eadro_generalization": "artifacts/care/eadro-generalization-20260928/results.json",
    "historical_lora": "artifacts/care/lora-aiops-paper128-v2/REPORT.md",
}
METRICS = ("PR@1", "PR@3", "PR@5", "MAP@5")
MAIN_METRICS = ("PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5")
DATASETS = (("trainticket", "TrainTicket"), ("aiops2022", "AIOps2022"))
METHODS = (
    "MicroScope", "MicroRCA", "SBLD", "LogFaultFlagger", "TraceAnomaly",
    "MicroRank", "Nezha", "Eadro", "MULAN", "ReAct", "mABC", "CARE+GPT", "CARE",
)
MODULES = (
    ("full", "CARE"), ("top5_restriction", "Top-5 update, tail kept"),
    ("unaugmented_with_generic_top5", "Unaugmented, generic Top-5"),
    ("original_unaugmented_top5", "Original unaugmented Top-5"),
    ("type_agnostic_only", "Aggregated expert only"),
    ("no_type_agnostic_expert", "No aggregated expert"),
    ("no_availability_flags", "No availability flags"),
    ("no_rpc", "No RPC residuals"),
    ("no_call_graph", "No call-neighborhood features"),
    ("no_graph", "No correction graph"),
    ("base_no_cl", "Base: no contrastive learning"),
    ("base_standard_cl", "Base: standard contrastive learning"),
    ("base_no_gat", "Base: no GAT"),
    ("base_no_modality_dropout", "Base: no modality dropout"),
    ("no_missing_augmentation_isolated", "No augmentation, flags retained"),
    ("no_missing_augmentation", "No augmentation and no flags"),
)
MODALITIES = (
    ("metric+log+trace", "All three"), ("log+trace", "Without metric"),
    ("metric+trace", "Without log"), ("metric+log", "Without trace"),
    ("metric", "Metric only"), ("log", "Log only"), ("trace", "Trace only"),
)
LOCAL_METHODS = METHODS[:9] + ("CARE",)
API_METHODS = ("CARE+GPT", "ReAct", "mABC")


def load(name: str):
    return json.loads((ROOT / SOURCES[name]).read_text())


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def value(summary: dict, metric: str, *, sd: bool = False) -> str:
    item = summary[metric]
    main = pct(item["mean"])
    return f"{main} ± {pct(item['std'])}" if sd else main


def table(headers, rows) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
        "",
    ]


def main() -> None:
    with (ROOT / SOURCES["matched_main"]).open(newline="") as handle:
        matched = list(csv.DictReader(handle))
    assert len(matched) == 2 * len(METHODS)
    by_main = {(row["dataset"], row["method"]): row for row in matched}
    assert len(by_main) == len(matched)
    review = load("review_aggregate")["datasets"]
    latency = load("latency")["methods"]
    fault_eff = load("fault_efficiency")["datasets"]
    accounting = load("api_accounting")["datasets"]
    costs = load("api_cost_estimate")["by_dataset_method"]
    api_audit = load("api_experiment_audit")["campaign_counts"]
    eadro_audit = load("eadro_generalization")["datasets"]
    for ds, _ in DATASETS:
        for metric in MAIN_METRICS:
            assert abs(float(by_main[(ds, "CARE")][f"{metric}_mean"]) -
                       review[ds]["main"]["robust_method"][metric]["mean"]) < 1e-10

    lines = [
        "# CARE 当前实验结果汇总（2026-09-28）", "",
        "本汇总只把最终冻结的 TrainTicket/AIOps2022 公开数据实验作为论文当前结果。"
        "同一测试事件在多个模型种子或 API 运行中重复出现，不能当作独立样本。"
        "表内排名指标均为百分数；主表是最终匹配评测，消融表为各自冻结实验的种子均值。"
        "涉密工业 PM 未复现，也不计入任何结果。", "",
        "## 实验口径", "",
    ]
    rows = []
    for ds, title in DATASETS:
        scope = review[ds]["data_scope"]
        c = scope["counts"]
        rows.append([title, c["train"], c.get("validation", "—"), c["test"],
                     scope["candidate_services"], scope["observed_root_services"],
                     scope["fault_types"]])
    lines += table(["数据集", "训练事件", "验证事件", "测试事件", "候选服务", "出现过的根因服务", "故障类型"], rows)
    lines += [
        "TrainTicket 的内存增长和磁盘 I/O 是明确实现的注入代理；当前 41 个候选应用服务不等于原论文未能核实的部署规模。"
        "AIOps2022 的 102/26 是按论文规模重建的划分，原始 128 个事件 ID 与划分种子未公开，不能声称逐事件同源。", "",
        "## 主实验：同一测试集上的根因排序", "",
        "单元格为均值；括号中为运行/种子标准差（单次运行不适用）。"
        "13 个方法使用同一服务顺序和评价器，但输入信息和实现适配程度不同。", "",
    ]
    for ds, title in DATASETS:
        lines += [f"### {title}", ""]
        rows = []
        for method in METHODS:
            row = by_main[(ds, method)]
            cells = []
            for metric in MAIN_METRICS:
                mean = pct(float(row[f"{metric}_mean"]))
                std = row[f"{metric}_std"]
                cells.append(f"{mean} ({pct(float(std))})" if std else mean)
            rows.append([method, row["runs"], *cells])
        lines += table(["方法", "运行数", *MAIN_METRICS], rows)
    lines += [
        "CARE 与最接近的零 LLM 外部适配 Eadro 的 PR@1 差异为 TrainTicket +4.67、AIOps2022 +10.77 个百分点；"
        "对应配对区间分别为 [-5.33, 14.00] 与 [-3.08, 26.15]，都跨过 0。"
        "主表是小规模固定测试集上的点估计，不能据此断言稳定胜出。", "",
        "CARE+GPT 的实际输入是五模型均分集成，其 PR@1 参考值为 93.33/92.31；"
        "加入 GPT 后变为 91.11/92.31。因此它不是相对单个 CARE 种子提升的证据。", "",
        "### Eadro 泛化审计", "",
        "保存的完整预测可直接复算训练、验证和测试表现。更强正则化候选仅按验证集选择；"
        "以下测试结果是事后诊断，不替换冻结主表。", "",
    ]
    audit_rows = []
    for ds, title in DATASETS:
        item = eadro_audit[ds]
        frozen = item["archive_audit"]["mean"]
        study = item["regularization_study"]
        after = study["result_mean"]
        audit_rows.append([
            title, "/".join(pct(frozen[s]["PR@1"]) for s in
                            ("fit", "validation", "test")),
            study["selected"],
            "/".join(pct(after[s]["PR@1"]) for s in
                     ("fit", "validation", "test")),
        ])
    lines += table(["数据集", "冻结 Eadro fit/val/test PR@1", "验证所选正则化候选",
                    "候选 fit/val/test PR@1"], audit_rows)
    lines += [
        "TrainTicket 训练/验证/测试为 96.11/91.67/90.00%；AIOps2022 为 85.93/74.29/81.54%。"
        "AIOps 验证集含 5 个 frontend 根因而测试集没有，测试总体分数高于验证并不能证明或排除新域过拟合。"
        "验证所选正则化候选在测试上反而下降 8.00/0.77 点，故保留冻结 Eadro 行。"
        "事件重叠组跨划分交集为 0；完整逐种子、根因频次与配对区间见"
        " `artifacts/care/eadro-generalization-20260928/REPORT.md`。", "",
        "## CARE 阶段与模块消融", "",
    ]
    for ds, title in DATASETS:
        lines += [f"### {title}：阶段", ""]
        stages = (("base", "基础定位"), ("standard_evidence", "KPI 专属、无增强"),
                  ("robust_evidence", "KPI 专属、增强"), ("generic_evidence", "聚合证据"),
                  ("robust_correction", "双专家修正"), ("robust_method", "CARE 冻结融合"))
        rows = [[label, *[value(review[ds]["main"][key], m, sd=True) for m in METRICS]]
                for key, label in stages]
        lines += table(["阶段", *METRICS], rows)
        lines += [f"### {title}：模块（均值 ± 种子标准差）", ""]
        full = review[ds]["module_ablations"]["full"]["PR@1"]["mean"]
        rows = []
        for key, label in MODULES:
            summary = review[ds]["module_ablations"][key]
            rows.append([label, *[value(summary, m, sd=True) for m in METRICS],
                         f"{100 * (summary['PR@1']['mean'] - full):+.2f}"])
        lines += table(["干预", *METRICS, "ΔPR@1 vs CARE"], rows)
    lines += [
        "模块干预与固定可用性拟合回答不同问题。TrainTicket 移除 RPC 残差/修正图时 PR@1 分别下降 24.00/29.33 点，"
        "PR@5 只下降 6.00/7.33 点，主要是首位排序变差。AIOps2022 对应 PR@1 下降 5.38/4.62 点。"
        "完整观测下未增强的 KPI 专属分支在 TrainTicket 达到 98.00% PR@1，高于 CARE 94.67%，说明鲁棒设计不是所有条件下的准确率最优。", "",
        "## 模态作用与部分可观测", "",
    ]
    for ds, title in DATASETS:
        lines += [f"### {title}：单独拟合的固定可用性证据分支", ""]
        rows = [[label, *[value(review[ds]["evidence_modalities"][key], m, sd=True)
                          for m in METRICS]] for key, label in MODALITIES]
        lines += table(["可用模态", *METRICS], rows)
        lines += [f"### {title}：端到端重新训练的缺模态部署方案", ""]
        items = [("robust_method", "全部模态 CARE", "main"),
                 ("metric", "无 metric", "end_to_end_modality_leave_one_out"),
                 ("log", "无 log", "end_to_end_modality_leave_one_out"),
                 ("trace", "无 trace", "end_to_end_modality_leave_one_out")]
        rows = [[label, *[value(review[ds][category][key], m, sd=True) for m in METRICS]]
                for key, label, category in items]
        lines += table(["部署方案", *METRICS], rows)
        lines += [f"### {title}：整模态缺失，固定基座和掩码", ""]
        rows = []
        for missing in ("metric", "log", "trace"):
            for label, summary in (
                ("无增强、无标志", review[ds]["supplementary_isolated_missing_ablation"]["results"]["complete"][missing]["without_flags"]["method"]),
                ("无增强、有标志", review[ds]["supplementary_isolated_missing_ablation"]["results"]["complete"][missing]["with_flags"]["method"]),
                ("有增强、有标志", review[ds]["complete_missing"][missing]["robust_method"]),
            ):
                rows.append([missing, label, *[value(summary, m) for m in METRICS]])
        lines += table(["缺失模态", "方案", *METRICS], rows)
        lines += [f"### {title}：混合服务级缺失", ""]
        rows = []
        for requested in (0.25, 0.5, 0.75):
            key = f"mixed|{requested}"
            item = review[ds]["local_missing"][key]
            supplemental = review[ds]["supplementary_isolated_missing_ablation"]["results"]["local"][key]
            for label, summary in (
                ("无增强、无标志", supplemental["variants"]["without_flags"]["method"]),
                ("无增强、有标志", supplemental["variants"]["with_flags"]["method"]),
                ("有增强、有标志", item["methods"]["robust_method"]),
            ):
                rows.append([f"{requested:.0%}", f"{item['realized_missing_rate']:.1%}", label,
                             *[value(summary, m) for m in METRICS]])
        lines += table(["请求删除", "实际删除", "方案", *METRICS], rows)
    lines += [
        "另外还完成了 metric/log/trace 各自 25/50/75% 的服务级删除；"
        "全部 12 种掩码设置及五项排名指标见 `review_aggregate` 的 `local_missing` 和 `tables/missingness.csv`。"
        "混合删除下增强并非普遍受益：TrainTicket 三档均降低 PR@1；AIOps2022 仅最高删除档提高 PR@1。"
        "整模态消融、端到端重新训练和混合掩码不能互换解释。", "",
        "## 历史未出现的故障类型", "",
    ]
    for ds, title in DATASETS:
        rows = [[label, *[value(review[ds]["history_type_holdout"][key], m, sd=True)
                          for m in METRICS]] for key, label in
                (("base", "留出类型基础模型"), ("without_generic", "无聚合专家"),
                 ("robust_method", "CARE"))]
        lines += [f"### {title}", ""] + table(["方法", *METRICS], rows)
    lines += [
        "按故障类型留出并清除重叠组，使用种子 42–44。CARE 相对留出类型基础模型的 PR@1 为"
        " TrainTicket 61.11% 对 44.44%，AIOps2022 73.08% 对 53.85%。"
        "该路径保留共享数值校准且未重建 fold 专属 Drain 日志，属于有监督的留出类型迁移，不是原始遥测冷启动。", "",
        "## 故障类型分层：准确率、延迟与 token", "",
        "下面的延迟是 CARE 本地打分均值；API 时间和 token 是每次 incident-run 的均值。"
        "其余 9 种本地方法及 3 种 API 方法逐类型的均值/p95 和输入/输出 token 拆分见 `fault_efficiency` 对应报告。", "",
    ]
    for ds, title in DATASETS:
        rows = []
        for fault, item in review[ds]["by_fault_type"].items():
            eff = fault_eff[ds][fault]
            assert item["count"] == eff["unique_incidents"]
            rows.append([
                eff["display"], item["count"], value(item["base"], "PR@1"),
                value(item["robust_method"], "PR@1"),
                value(item["robust_method"], "MAP@5"),
                f"{eff['local']['CARE']['latency_ms']['mean']:.2f}",
                *[f"{eff['llm'][method]['total_tokens']['mean']:.0f}" for method in API_METHODS],
            ])
        lines += [f"### {title}", ""] + table(
            ["类型", "n", "基础 PR@1", "CARE PR@1", "CARE MAP@5", "CARE ms",
             "CARE+GPT token", "ReAct token", "mABC token"], rows)
    lines += [
        "AIOps2022 的 CPU load、Read I/O、Process abort 各仅 1 个独立测试事件，"
        "其分类准确率和 p95 都不能作为稳定的类型效应。重复模型种子/API 调用不会增大独立事件数。", "",
        "## 参数敏感性、候选瓶颈与输入扰动", "",
    ]
    sens_rows = []
    for group, label, values in (
        ("candidate_and_fusion|all|", "证据权重", ("0.0", "0.1", "0.25", "0.5", "0.75", "1.0")),
        ("tree_count|", "树数", ("64", "128", "256", "512")),
        ("min_samples_leaf|", "最小叶样本", ("1", "3", "5")),
    ):
        for setting in values:
            pair = []
            for ds, _ in DATASETS:
                summary = review[ds]["sensitivity"][group + setting]
                pair.extend((value(summary, "PR@1"), value(summary, "MAP@5")))
            sens_rows.append([label, setting, *pair])
    for setting in ("0.0", "0.25", "0.5", "0.75", "1.0"):
        pair = []
        for ds, _ in DATASETS:
            summary = review[ds]["graph_retention"][setting]["metrics"]
            pair.extend((value(summary, "PR@1"), value(summary, "MAP@5")))
        sens_rows.append(["图证据与基座邻接保留比例", setting, *pair])
    lines += table(["参数", "值", "TrainTicket PR@1", "TrainTicket MAP@5",
                    "AIOps PR@1", "AIOps MAP@5"], sens_rows)
    candidate_rows = []
    for k in ("1", "3", "5", "all"):
        cells = []
        for ds, _ in DATASETS:
            data = review[ds]
            source_k = k if k != "all" else max(data["candidate_recall"], key=int)
            strict = data["sensitivity"][f"candidate_and_fusion|{k}|0.75"]
            cells.extend((pct(data["candidate_recall"][source_k]["mean"]),
                          value(strict, "PR@1")))
        candidate_rows.append([k, *cells])
    lines += table(["候选 k", "TrainTicket recall", "TrainTicket strict PR@1",
                    "AIOps recall", "AIOps strict PR@1"], candidate_rows)
    perturb_rows = []
    for name in review["trainticket"]["perturbations"]:
        cells = []
        for ds, _ in DATASETS:
            data = review[ds]["perturbations"][name]
            cells.extend((value(data, "PR@1"), value(data, "MAP@5")))
        perturb_rows.append([name, *cells])
    lines += table(["时间扰动", "TrainTicket PR@1", "TrainTicket MAP@5",
                    "AIOps PR@1", "AIOps MAP@5"], perturb_rows)
    lines += [
        "还完成 25 组旧规则参数与 15 组 LLM 单纯形权重扫描；逐设置五项指标保存在"
        " `review_aggregate` 的 `legacy_rule_sensitivity`、`llm_simplex` 和 `tables/sensitivity.csv`。"
        "这些是测试集上的描述性扫描，不能按最佳测试格重新宣称冻结方法性能。", "",
        "## 诊断时间与 LLM 开销", "",
        "本地时间以单 CPU 线程、已加载模型、准备好的数值输入开始计时；API 时间仅合计成功 HTTP 请求。"
        "两者边界不同，也都不是原始遥测采集到结论的端到端耗时。均值/p95 如下。", "",
    ]
    rows = []
    for method in LOCAL_METHODS + API_METHODS:
        cells = []
        for ds, _ in DATASETS:
            item = latency[ds][method]
            if method in API_METHODS:
                cells.append(f"{item['mean_ms'] / 1000:.2f}/{item['p95_ms'] / 1000:.2f} s")
            else:
                cells.append(f"{item['mean_ms']:.2f}/{item['p95_ms']:.2f} ms")
        rows.append([method, *cells])
    lines += table(["方法", "TrainTicket 均值/p95", "AIOps 均值/p95"], rows)
    cost_rows = []
    for ds, title in DATASETS:
        for method in API_METHODS:
            item = accounting[ds]["methods"][method]
            cost = item["cost"]
            cost_rows.append([
                title, method, item["incident_run_decisions"],
                f"{item['calls_per_incident']:.2f}",
                f"{cost['prompt_tokens']:,}", f"{cost['completion_tokens']:,}",
                f"{cost['total_tokens']:,}", f"{item['tokens_per_incident']:,.0f}",
                f"{item['successful_call_sum_seconds_per_incident']:.2f}",
                f"{costs[f'{ds}/{method}']['usage_equivalent_usd']:.6f}",
            ])
    lines += table(["数据集", "方法", "决策数", "调用/决策", "输入 token 总计",
                    "输出 token 总计", "总 token", "token/决策", "HTTP s/决策",
                    "使用量折算 USD"], cost_rows)
    assert api_audit["logical_calls"] == 1989
    assert api_audit["provider_completed"] + api_audit["provider_incomplete_max_output_tokens"] == 1989
    outcome_rows = []
    for ds, title in DATASETS:
        for method in API_METHODS:
            item = accounting[ds]["methods"][method]
            outcome_rows.append([
                title, method, item["incident_run_decisions"],
                item.get("invalid_incident_outputs", "—"),
                item.get("accepted_corrections") if item.get("accepted_corrections") is not None else "—",
                ", ".join(f"{k}:{v}" for k, v in item["reason_counts"].items()),
            ])
    lines += table(["数据集", "API 方法", "诊断决策", "无效最终输出", "采纳修正",
                    "终止/回退状态计数"], outcome_rows)
    lines += [
        "当前 API 账本为 1,989 次成功调用、7,070,635 token（输入 6,462,318，输出 608,317），"
        "HTTP 时间合计 8,316.20 秒；766 次 provider 生成完成，1,223 次触及 `max_output_tokens`，"
        "因此部分基线是短预算适配结果。"
        "按 2026-09-27 价格表换算的使用量价值合计约 $1.555，订阅增量收费未知。"
        "CARE 本身每次诊断 0 次 LLM 调用、0 token。", "",
        "## 结构约束与可审计性", "",
        "TrainTicket 的训练历史图支持集有 108 个训练事件、0 个测试事件。"
        "30/30 个历史 JSON 响应可解析且满足候选约束；435/435 条提出的边具有精确引用，"
        "其中 318 条通过门控；30/30 条文字解释复述的输入证据可在记录中核对。"
        "AIOps2022 使用受约束的单服务 token，因此不产生路径或解释。"
        "两数据集都没有独立的传播边或因果路径真值；这些检查只证明结构和来源一致，"
        "不能证明路径因果正确。详细字段在 `review_aggregate` 的 `auditability`。", "",
        "## 独立于当前 FSE 主表的历史探索", "",
        "早期本地 Qwen3-8B 方案（与当前 GPT API 基线不同）：", "",
    ]
    qwen_rows = []
    for ds, title in DATASETS:
        for key, label in (("llm_only", "本地 Qwen 单独诊断"),
                           ("robust_no_llm", "CARE 无 LLM"),
                           ("robust_plus_llm_25", "CARE + 0.25 Qwen 融合")):
            summary = review[ds]["llm"][key]
            qwen_rows.append([title, label, *[value(summary, m) for m in METRICS]])
    lines += table(["数据集", "方法", *METRICS], qwen_rows)
    lines += [
        "该探索的部署融合权重最终选为 0，当前论文主结果不调用本地 Qwen。"
        "此表与下面单独训练的 AIOps LoRA 不是同一次协议。"
        "早期训练记录为 TrainTicket 1,580.74 秒、AIOps2022 1,052.57 秒，"
        "各自部署适配器约 174.66 MB；训练时使用 A100 80GB。", "",
        "AIOps paper128 专用 Qwen3-8B LoRA：固定排列 seed 42 的独立定位 PR@1 为 84.62%（22/26），"
        "未微调基座为 26.92%，同输入直接证据模型为 92.31%；证据模型与 LoRA 固定融合仍为 92.31%。"
        "该实验仅一个训练种子，使用受约束的单 token 分类；不能和当前 GPT API 表作为同一推理协议比较。"
        "完整配对区间、排列敏感性及训练预算见 `historical_lora`。", "",
        "## 读数时的关键限制", "",
        "1. 主实验测试事件只有 30/26 个；AIOps 部分故障类型仅 1 个事件。"
        "置信区间和种子标准差反映这些固定事件及模型/调用重复，不代表新部署域。",
        "2. 外部 baseline 中有源码适配与按论文重建的实质性 adapter，输入语义不完全相同；"
        "不能把数值差异直接解释为原作者完整系统差异。",
        "3. 模态缺失包含单独拟合、端到端重训、固定基座扰动三种设计；"
        "只在相同设计内比较。图保留率同时改动图/RPC 描述符和基座邻接。",
        "4. API 分析只使用最终完整的 `opencode-go-20260927` 活动；"
        "此前中断或代理失败的运行不进入主表。", "",
        "## 可复核源文件", "",
    ]
    for name, path in SOURCES.items():
        lines.append(f"- `{name}`：`{path}`")
    lines += [
        "- 逐故障类型的全部 13 种方法延迟与 LLM 输入/输出 token："
        "`artifacts/care/fse2027-fault-type-efficiency-20260928/REPORT.md`。",
        "- 消融、模态、缺失、故障类型、敏感性的逐配置 CSV："
        "`artifacts/care/review-response-v1/report/tables/`。", "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    hashes = {name: {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
              for name, path in SOURCES.items()}
    MANIFEST.write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n")
    print(OUT)
    print(f"{len(lines)} lines; {len(matched)} matched rows; {len(MODULES)} module variants")


if __name__ == "__main__":
    main()
