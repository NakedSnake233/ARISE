"""Export measured base-model results and precise paper-revision language."""
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=ROOT/'artifacts/care/base-aiops2022-paper128'
    summary=json.loads((folder/'summary.json').read_text())
    data=ROOT/summary['protocol']['config']['data']
    rows=json.loads((data/'incidents.json').read_text())
    primary=summary['primary'];metrics=list(summary['mean'])
    lines=['# AIOps2022 基础模型：按论文规模重建的实验', '',
           '原论文仅给出 128 个事件与 8:2 划分，没有事件清单或随机种子。此次采用冻结的 128 事件队列，102 训练 / 26 测试。该实验对齐已明确条件，不能声称恢复了原始划分。', '',
           '## 实测结果', '',
           '主结果预先指定初始化 seed 42；补充结果采用固定的 42、43、44、45、46 五个初始化种子，在同一划分上重复训练。均值的标准差是初始化间的样本标准差，不代表五个独立测试集。', '',
           '| 指标 | 论文 Base-only | 本次 seed 42 | 五种子均值 ± 标准差 |',
           '|---|---:|---:|---:|']
    for metric in metrics:
        lines.append(f"| {metric} | {summary['paper_base_only'][metric]:.4f} | {primary[metric]:.4f} | {summary['mean'][metric]:.4f} ± {summary['sample_std'][metric]:.4f} |")
    lines += ['', '以上为基础定位模型，不是完整 CARE。论文数值仅供描述性参照；原事件 ID 缺失，不能作配对复现比较。测试集只有 26 个事件，PR@k 每多命中一个事件变化约 3.85 个百分点。', '',
              '## 对齐条件与补充设置', '',
              '- 事件级 8:2（整数取整为 102/26）；同事件和重叠观测窗口不跨集合。',
              '- Drain3、PF-IDF、指标/trace 标准化与 SDG 均从本次 102 个训练事件重新拟合，未复用旧划分的拟合产物。',
              '- 8-head GAT、MLP、学习率 0.01、监督定位损失与 metric-anchored InfoNCE / residual diversity；调用边对称化并增加自环。',
              '- 补齐论文描述的重叠指标 patches；patch 长度 3、stride 1、平均聚合为明确记录的重建选择。',
              '- 论文未指定基础模型训练轮数：本次预先固定 50 epochs，使用最后一轮权重；完整训练 80% 数据，不另外扣除验证集。测试结果不用于 checkpoint、超参数或种子选择。',
              '- 其余未指定的批量大小、网络宽度、窗口长度、损失权重等见 configs/care/aiops2022-paper-base.json。本次使用 CPU，不报告与论文硬件可比的推理时延。', '',
              '## 论文实验描述建议', '',
              '> We selected 128 eligible service-level fault incidents from the public AIOps2022 March fault data using a fixed, label-agnostic sampling procedure (seed 42). We split the cohort into 102 training and 26 test incidents, keeping overlapping observation windows in the same split. All preprocessing statistics, log templates, and service-dependency edges were fitted using training incidents only. The base localizer used eight GAT heads and a learning rate of 0.01. We trained for a prespecified 50 epochs and evaluated the final checkpoint, without test-driven model selection. We report seed 42 as the primary run and the mean and sample standard deviation over five fixed initialization seeds (42–46) on the same split. The incident manifest and reconstruction settings are provided with the implementation.', '',
              '上述文字只适用于本次基础模型实验。完整 CARE、其他基线和消融需在相同新划分上重新训练并评价后，才能更新相应论文表格。', '',
              '## 与旧实验的隔离', '']
    old=json.loads((ROOT/'data/care/aiops2022-prepared/incidents.json').read_text())
    old_train={r['incident_id'] for r in old if r['split']=='train'}
    test={r['incident_id'] for r in rows if r['split']=='test'}
    overlap=len(old_train & test)
    lines += [f'新测试集中有 {overlap} 个事件属于旧实验的训练集。因此旧基础模型、旧 Qwen LoRA adapter、旧 FPG 和旧预处理不能用于声称新划分的测试成绩。本次基础模型全部重新初始化训练，旧实验保留在原目录。', '',
              '## 划分分布', '', '| 根因服务 | 训练事件 | 测试事件 |', '|---|---:|---:|']
    counts={s:collections.Counter(r['root_service'] for r in rows if r['split']==s) for s in ['train','test']}
    for service in sorted(counts['train']|counts['test']):
        lines.append(f"| {service} | {counts['train'][service]} | {counts['test'][service]} |")
    lines += ['', '## 产物', '',
              '- 划分清单：configs/care/aiops2022-paper-split.json',
              '- 预处理：data/care/aiops2022-paper128-prepared/',
              '- 主模型与逐事件排名：artifacts/care/base-aiops2022-paper128/seed-42/',
              '- 所有重复实验、均值和标准差：artifacts/care/base-aiops2022-paper128/summary.json', '']
    destination=ROOT/'AIOPS_PAPER_BASE_RESULTS.md'
    destination.write_text('\n'.join(lines))
    sources=[ROOT/'scripts/care'/name for name in ['paper_split.py','prepare_aiops.py','train_base.py','run_paper_base.py']]
    sources += [ROOT/'Paper/ICSE_care_refs40/conference_101719_revised.tex']
    (folder/'source_checksums.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(destination)


if __name__=='__main__':main()
