"""Generate complete tables, paired statistics and figures from measured outputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def moments(values):
    return {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else 0.}


def summarize(reports,key):
    return {metric:moments([r[key][metric] for r in reports]) for metric in reports[0][key]}


def cell(summary,key='PR@1'):
    r=summary[key];return f"{r['mean']:.4f} ± {r['std']:.4f}"


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('artifacts/care/full-v3'));args=p.parse_args();out=args.run
    protocol=json.loads((out/'protocol.json').read_text());cfg=protocol['config'];state=json.loads((out/'status.json').read_text())
    if state['state']!='complete':raise ValueError('Refuse to publish an incomplete experiment matrix')
    summary={};raw=[];report=['# RPC 证据重排：完整实验结果','',
      '所有数字来自冻结实验配置的实际产物。五个种子均重新训练基础模型和证据重排器；满足相同配置及输入哈希的已有基础训练被复用。主配置固定证据权重 0.25、候选 Top-5，不根据本轮测试成绩选择种子或参数。',
      '', '这是对已有队列的完整重复和消融，不是独立新测试集上的确认；主方法不调用 LLM，也不代表原论文所有外部基线或工业 PM 已复现。', '',
      '## 主要结果', '', '| 数据集 | 方法 | PR@1 | PR@3 | PR@5 | MAP@3 | MAP@5 |','|---|---|---:|---:|---:|---:|---:|']
    for dataset in cfg['datasets']:
        methods={}
        for variant in cfg['base_variants']:
            reports=[]
            for seed in cfg['seeds']:
                r=json.loads((out/dataset/variant/f'seed-{seed}/correction/result.json').read_text())
                assert r['protocol_sha256']==hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()
                reports.append(r)
                for phase in ['base','corrected']:raw.append({'dataset':dataset,'variant':variant,'seed':seed,'phase':phase,**r[phase]})
            methods[variant]={'base':summarize(reports,'base'),'corrected':summarize(reports,'corrected'),'per_seed':reports}
        ablations={}
        for name in cfg['evidence_ablations']:
            rs=[json.loads((out/dataset/'full'/f'seed-{s}/correction/evidence_ablations.json').read_text())[name] for s in cfg['seeds']]
            ablations[name]={'corrected':summarize(rs,'corrected'),'per_seed':rs}
        grids=[json.loads((out/dataset/'full'/f'seed-{s}/correction/sensitivity.json').read_text())['grid'] for s in cfg['seeds']]
        grid=[]
        for i,entry in enumerate(grids[0]):
            grid.append({'top_k':entry['top_k'],'weight':entry['weight'],
                         'metrics':{k:moments([g[i]['metrics'][k] for g in grids]) for k in entry['metrics']}})
        deltas=[]
        for s in cfg['seeds']:
            pred=np.load(out/dataset/'full'/f'seed-{s}/correction/predictions.npz')
            y=np.load(Path(cfg['datasets'][dataset]['data'])/'features.npz')['labels'][pred['indices']]
            deltas.append((pred['final'].argmax(1)==y).astype(float)-(pred['base'].argmax(1)==y).astype(float))
        incident_delta=np.mean(deltas,axis=0);rng=np.random.default_rng(42)
        boot=incident_delta[rng.integers(0,len(incident_delta),size=(10000,len(incident_delta)))].mean(1)
        paired={'mean_top1_delta':float(incident_delta.mean()),'incident_cluster_bootstrap_95_percentile_ci':np.quantile(boot,[.025,.975]).tolist(),
                'note':'Resample incidents, retain all five seeds together. Repeated-test exploratory interval; not a new holdout confirmation.'}
        faults={}
        full=methods['full']['per_seed']
        for fault in full[0]['by_fault_type']:
            rs=[r['by_fault_type'][fault] for r in full]
            faults[fault]={'count':rs[0]['count'],'base':summarize(rs,'base'),'corrected':summarize(rs,'corrected')}
        summary[dataset]={'methods':methods,'evidence_ablations':ablations,'sensitivity':grid,'paired_across_seeds':paired,'fault_types':faults,
                          'predict_seconds_per_event':moments([r['evidence_predict_seconds_per_event'] for r in full])}
        for phase in ['base','corrected']:
            label='基础模型' if phase=='base' else 'RPC 证据重排'
            report.append('| '+dataset+' | '+label+' | '+' | '.join(cell(methods['full'][phase],k) for k in ['PR@1','PR@3','PR@5','MAP@3','MAP@5'])+' |')
        report.append('| '+dataset+' | 独立证据模型（预设对照） | '+' | '.join(cell(ablations['evidence_only']['corrected'],k) for k in ['PR@1','PR@3','PR@5','MAP@3','MAP@5'])+' |')
    report+=['','数值为五个种子的均值 ± 样本标准差。这里的种子重复包含基础训练变化，与前次固定基础模型、只改变重排树种子的结果不同。']
    report+=['','独立证据模型在两套数据上的 Top-1 都高于当前保守融合主配置。这是预设对照暴露的主方法局限：固定权重与基础候选集合仍可能限制证据发挥作用。不能据此宣称融合一定优于独立证据模型，也未根据该测试结果更换主方法或融合权重。下一轮若研究置信度校准或候选扩展，应只在训练内交叉验证中选择，再用独立新事件确认。']
    for dataset,s in summary.items():
        report+=['',f'## {dataset}：配对结果','', '| 种子 | 基础 PR@1 | 修正 PR@1 | 纠对 | 改错 | 双侧精确 McNemar p |','|---|---:|---:|---:|---:|---:|']
        for r in s['methods']['full']['per_seed']:
            p=r['paired_statistics'];report.append(f"| {r['seed']} | {r['base']['PR@1']:.4f} | {r['corrected']['PR@1']:.4f} | {p['fixed']} | {p['broken']} | {p['mcnemar_exact_two_sided_p']:.4f} |")
        p=s['paired_across_seeds'];report+=['',f"跨种子平均 Top-1 差值：{p['mean_top1_delta']:.6f}；以事件为簇、保留种子相关性的 bootstrap 95% 区间：{p['incident_cluster_bootstrap_95_percentile_ci']}。未把五个种子的相同事件视作独立样本。多项探索性检验未做多重比较校正，不据此宣称普遍显著提升。",'',
          '### 基础组件与同步模态消融','', '| 配置 | 基础 PR@1 | 修正 PR@1 | 修正 MAP@5 |','|---|---:|---:|---:|']
        for name,m in s['methods'].items():report.append(f"| {name} | {cell(m['base'])} | {cell(m['corrected'])} | {cell(m['corrected'],'MAP@5')} |")
        report+=['','no_metrics / no_logs 同步移除基础模型与证据模型的该模态。no_traces 更准确地表示“去 trace 数值”：移除 trace 编码器观测和 RPC 数值证据；基础图仍来自训练事件，证据邻域仍使用当前事件观测的调用拓扑，因此不是完全移除所有 trace 来源的信息。standard_cl 是三模态两两对称 InfoNCE。','',
          '### 修正组件消融','', '| 配置 | PR@1 | MAP@5 |','|---|---:|---:|']
        for name,m in s['evidence_ablations'].items():report.append(f"| {name} | {cell(m['corrected'])} | {cell(m['corrected'],'MAP@5')} |")
        report+=['','evidence_only 是独立证据模型在所有服务上的排名，其余消融仍使用固定 Top-5、0.25 融合。所有结果完整报告，没有据此替换主模型。','',
          '### 故障类型','', '| 故障类型 | 测试事件 | 基础 PR@1 | 修正 PR@1 | 修正 MAP@5 |','|---|---:|---:|---:|---:|']
        for name,m in s['fault_types'].items():report.append(f"| {name} | {m['count']} | {cell(m['base'])} | {cell(m['corrected'])} | {cell(m['corrected'],'MAP@5')} |")
        ms=s['predict_seconds_per_event'];report+=['',f"证据评分耗时（CPU，按测试批量摊到每事件）：{ms['mean']*1000:.3f} ± {ms['std']*1000:.3f} ms。此处不包含遥测提取、基础推理和 I/O，不能作为端到端诊断耗时。"]
    if (out/'timing.json').exists():
        timing=json.loads((out/'timing.json').read_text());summary['timing']=timing
        report+=['','## 单事件推理计时','', '| 数据集 | 设备 | 基础均值(ms) | 证据与融合均值(ms) | 总均值(ms) | 总 P95(ms) |','|---|---|---:|---:|---:|---:|']
        for d,r in timing.items():
            report.append(f"| {d} | {r['device']} | {r['base']['mean_seconds']*1000:.3f} | {r['evidence_and_fusion']['mean_seconds']*1000:.3f} | {r['total_cached_feature_inference']['mean_seconds']*1000:.3f} | {r['total_cached_feature_inference']['p95_seconds']*1000:.3f} |")
        report+=['','计时使用已加载模型和预计算数值特征，每个测试事件重复三次，包含基础 forward、概率传回 CPU、证据打分及融合。排除原始遥测处理、模型加载和 I/O；计时期间完整实验中的 CPU 消融仍在运行，不能解释为独占机器的延迟上限或端到端诊断耗时。TrainTicket 另采样三个事件的原始 RPC 文件提取耗时，详见 timing.json。']
    report+=['','## 可复现性与范围','',
      '- 固定协议和源代码校验和：`artifacts/care/full-v3/protocol.json`。',
      '- 70 个配置的原始训练日志、模型、逐事件概率与纠错结果均保留；完整模型含全部敏感性和修正消融。',
      '- AIOps 使用新 paper128 划分，未使用旧队列 LoRA/FPG。RPC 无原生角色标签，按相同操作父子链路推断；覆盖率见 `aiops2022/rpc/coverage.json`。',
      '- TrainTicket 与 AIOps 分别沿用各自确定的基础训练协议，不把跨数据集差异归因于单个模块。',
      '- 未运行工业 PM、外部方法完整复现、新故障事件确认集；不据本轮结果支持这些未完成的比较。',
      '- 全部敏感性结果见 `summary.json`；图中的网格只描述固定模型行为，不用于重新选参。',
      '', '正式论文应报告本轮多种子均值及其局限，并将 RPC 证据重排写成新的方法扩展；不能将旧论文中的 LLM/FPG 贡献自动套用于这组结果。']
    report+=['','## 审计补充与数据限制','',
      '独立上下文、同系列模型的只读审计给出 WARN：实测数值可复算，未发现虚构标注或当前训练使用测试标签；主要限制来自重复测试、小范围工作负载、混合采集版本及原有溯源清单不完整。审计属于 provisional assurance，详见项目根目录 EXPERIMENT_AUDIT.md。',
      '',
      '- 指标定义：PR@k 是真根因进入前 k 名的事件比例；单根因条件下 MAP@k 是每事件 `1/rank`（rank≤k，否则 0）的均值。PR@k 不是除以 k 的检索精确率。',
      '- TrainTicket 的 41 个应用服务中，11 个有非零 trace 特征，12 个有非零 log 特征，29 个在全部事件中两者都为零。故障实际注入 10 个目标服务，主要负载是两类车次搜索请求。测试中的服务/故障组合都曾在训练出现，不构成未见服务或未见故障组合评价。',
      '- TrainTicket 150 条采集记录中，140 条标记 collector_version=2；10 条较早记录缺少该版本字段和新版 baseline trace gate，包含 4 条测试事件。本轮保持原队列不变，未将其冒充统一版本采集。',
      '- AIOps 测试覆盖 8/10 个根因服务，与训练共享七个 cloud-day。窗口根据已标注的故障起点截取，不验证自动故障检测；缺少恢复时刻，排除其他故障起点不能证明不存在前次故障延续。',
      '- 原始数据、标注、缓存数组与有序特征名、复用/新模型的运行后哈希封存见 data_model_seal.json；运行 scripts/care/resume_full_suite.py 会先验证封存清单再决定是否恢复。运行后封存不冒充运行前时间证明，也不消除科学评价范围限制。',
      '', '独立审计还按原设备与原批次划分重放了全部 70 个基础模型，概率逐位一致；70 个证据模型的重放误差不超过约 2.2e-16。共核对 278 条真实标注、60 项修正消融、240 个敏感性单元与 140 行指标 CSV。']
    summary['protocol_sha256']=hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    text='\n'.join(report)+'\n';(out/'REPORT.md').write_text(text);Path('CARE_FULL_EXPERIMENT_RESULTS.md').write_text(text)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/care-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure=out/'figures';figure.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for ax,dataset in zip(axes,cfg['datasets']):
        s=summary[dataset];names=cfg['base_variants'];x=np.arange(len(names))
        for offset,key,color,label in [(-.18,'base','#8192a3','Base'),(.18,'corrected','#1769aa','With RPC evidence correction')]:
            mean=[s['methods'][n][key]['PR@1']['mean'] for n in names];std=[s['methods'][n][key]['PR@1']['std'] for n in names]
            ax.bar(x+offset,mean,.36,yerr=std,capsize=3,color=color,label=label)
        ax.set_xticks(x,['Full','No CL','Std CL','No GAT','No metric','No trace','No log'],rotation=30,ha='right');ax.set_ylim(0,1.04)
        ax.set_ylabel('PR@1');ax.set_title(dataset);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    for ext in ['pdf','svg']:fig.savefig(figure/('components.'+ext))
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,dataset in zip(axes,cfg['datasets']):
        for k in cfg['sensitivity']['top_k']:
            entries=[g for g in summary[dataset]['sensitivity'] if g['top_k']==k]
            ax.errorbar([e['weight'] for e in entries],[e['metrics']['PR@1']['mean'] for e in entries],
                        yerr=[e['metrics']['PR@1']['std'] for e in entries],marker='o',capsize=2,label=f'k={k}')
        ax.set(xlabel='Evidence fusion weight',ylabel='PR@1',title=dataset,ylim=(0,1.03));ax.grid(alpha=.2);ax.legend()
    for ext in ['pdf','svg']:fig.savefig(figure/('sensitivity.'+ext))
    plt.close(fig)
    print('Complete report:',out/'REPORT.md')


if __name__=='__main__':main()
