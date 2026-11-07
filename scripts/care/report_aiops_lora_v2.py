"""Independently recompute saved ranks and produce an honest completed-run report."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    from safetensors import safe_open
    cfg=json.loads(Path('configs/care/aiops-paper128-lora-v2.json').read_text());out=Path(cfg['output'])
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    result=json.loads((out/'result.json').read_text());pred=np.load(out/'comparison.npz')
    rows=json.loads((Path(cfg['data'])/'incidents.json').read_text());meta=json.loads((Path(cfg['data'])/'metadata.json').read_text())
    services=meta['services'];te=pred['indices'];labels=pred['labels']
    assert len(te)==26 and np.array_equal(te,[i for i,r in enumerate(rows) if r['split']=='test'])
    assert np.array_equal(labels,[services.index(rows[i]['root_service']) for i in te])
    def check(p,reported):
        assert p.shape==(26,10) and np.isfinite(p).all() and (p>=0).all()
        assert np.allclose(p.sum(1),1,atol=1e-6)
        positions=np.asarray([list(np.argsort(-r,kind='stable')).index(int(y))+1 for r,y in zip(p,labels)])
        assert reported['correct']==int(sum(positions==1))
        for k in [1,3,5]:assert np.isclose(reported[f'PR@{k}'],sum(positions<=k)/26)
        for k in [3,5]:assert np.isclose(reported[f'MAP@{k}'],sum(1/int(r) if r<=k else 0 for r in positions)/26)
    check(pred['base'],result['base']);check(pred['evidence'],result['evidence_only'])
    for seed in cfg['inference_order_seeds']:
        for method in ['unadapted','lora','base_plus_lora','evidence_plus_lora']:
            check(pred[f'{method}_{seed}'],result['orders'][str(seed)][method])
    state=json.loads((out/'adapter/trainer_state.json').read_text())
    assert state['global_step']==39 and state['epoch']==3
    with safe_open(str(out/'adapter/adapter_model.safetensors'),framework='np') as f:
        weights={k:float(np.linalg.norm(f.get_tensor(k).astype(np.float64))) for k in f.keys() if 'lora_B' in k}
    assert weights and all(np.isfinite(v) and v>0 for v in weights.values())
    seal=json.loads((out/'protocol.json').read_text())
    for p,digest in seal['sha256'].items():assert sha(Path(p))==digest,p
    supplement=json.loads((out/'supplemental-seal.json').read_text())
    assert supplement['original_protocol_sha256']==sha(out/'protocol.json')
    for p,digest in supplement['sha256'].items():assert sha(Path(p))==digest,p
    verification={'status':'passed','checks':['26 original incident labels and fixed indices match','all 14 method/order rank metric records independently recomputed',
                     'frozen inputs/configuration/sources unchanged','supplemental pre-evaluation seal verified, including original Qwen weight/tokenizer hashes','final checkpoint epoch 3, step 39','all LoRA B tensors finite and nonzero'],
                  'lora_B_tensors':len(weights),'lora_B_l2_range':[min(weights.values()),max(weights.values())],
                  'sha256':{str(p):sha(p) for p in [out/'result.json',out/'comparison.npz',out/'adapter/adapter_model.safetensors',out/'adapter/adapter_config.json',out/'adapter/trainer_state.json']}}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    primary=result['orders'][str(cfg['primary_order_seed'])]
    lines=['# AIOps paper128 专用 LoRA 实验结果','',
           '本轮独立训练已完成。使用 102 个训练事件，从官方 Qwen3-8B 新建 LoRA，固定训练 3 轮 / 39 步，使用最终权重；26 个测试事件不参与梯度更新或 checkpoint 选择。',
           '', '## 主结果：预先固定的排列种子 42', '',
           '| 方法 | Top-1 正确数 | PR@1 | PR@3 | PR@5 | MAP@5 |',
           '|---|---:|---:|---:|---:|---:|']
    methods=[('基础定位模型 seed 42',result['base']),('直接证据模型 seed 42',result['evidence_only']),
             ('同输入未微调 Qwen3-8B',primary['unadapted']),('新 LoRA 独立定位',primary['lora']),
             ('基础模型 + 0.25 LoRA（top-5）',primary['base_plus_lora']),('直接证据模型 + 0.25 LoRA（全部候选）',primary['evidence_plus_lora'])]
    for name,r in methods:lines.append(f"| {name} | {r['correct']}/26 | {r['PR@1']:.2%} | {r['PR@3']:.2%} | {r['PR@5']:.2%} | {r['MAP@5']:.4f} |")
    lines+=['','## 配对变化','','| 比较 | 修复 | 误改 | PR@1 变化（百分点） | 95% 配对 bootstrap 区间（百分点） | 精确 McNemar p |','|---|---:|---:|---:|---|---:|']
    for name,p in primary['paired'].items():
        lo,hi=p['paired_bootstrap_95_percentile_ci'];lines.append(f"| {name} | {p['fixed']} | {p['broken']} | {100*p['top1_delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {p['mcnemar_exact_two_sided_p']:.4f} |")
    lines+=['','## 排列敏感性：全部预注册排列','','| 排列种子 | 未微调 Qwen PR@1 | LoRA PR@1 | 基础 + LoRA PR@1 | 证据 + LoRA PR@1 |','|---|---:|---:|---:|---:|']
    for seed,r in result['orders'].items():lines.append('| '+seed+' | '+' | '.join(f"{r[m]['PR@1']:.2%}" for m in ['unadapted','lora','base_plus_lora','evidence_plus_lora'])+' |')
    lines+=['','以上是同一套 LoRA 的三种候选排列，不能当作三个训练种子。主结果仍为预先固定的种子 42，未选择测试效果最好的排列。',
            '', '## 评测方式与限制','',
            '- 输入展示全部 10 个服务，不包含基础模型排名或分数；监督目标是原始标注根因对应的字母。历史图只使用训练事件，当前训练事件检索排除自身。',
            '- 这里评测的是受约束的下一 token 诊断分类：在 A–J 的单 token logits 中排序。不是自由生成的解释质量，也不验证因果传播图。',
            f"- 主排列下，10 个合法字母在完整词表中的平均概率质量：未微调 {primary['valid_letter_mass']['unadapted']:.4f}，LoRA {primary['valid_letter_mass']['lora']:.4f}。合法字母内 softmax 用于服务排序，性能指标的分母始终为 26 个真实测试事件。",
            '- 仅一个训练种子；测试集小且已经多次用于前期探索，统计区间和 p 值未修正研究迭代/多重比较，也不代表新的独立测试集。',
            '- paper128 按论文规模重建：论文未提供原始 128 个事件 ID 或划分种子，不能声称逐事件与论文划分一致。事件窗口锚定已标注的故障起点，本实验衡量已知事件窗口内的定位，不是故障检测。',
            '- 独立 LoRA 与独立证据模型都看到全部 10 个服务；基础模型前 5 候选仅覆盖 23/26 个根因，固定 top-5 融合受此候选范围限制。',
            '- 本轮没有根据测试结果修改提示词、超参数、checkpoint 或融合权重。固定融合可能不能利用独立定位模型的所有增益。',
            '', '## 复核与产物','',
            '- `verification.json`：原始事件标签映射、全部方法指标独立重算、训练预算、LoRA 更新和冻结协议哈希检查。',
            '- `adapter/`：新 LoRA 权重、配置、训练状态和日志。',
            '- `result.json`、`comparison.npz`、`cases.csv`：指标和逐事件结果。',
            '- `unadapted-inference.json`、`lora-inference.json`：每次推理的原始字母 logits、合法输出概率质量和未约束首 token。',
            '- `protocol.json`、`training.yaml`：训练前固定的配置和输入/源码哈希。`supplemental-seal.json` 在训练期间、评测开始前补记共享指标源码、参照预测和实际 Qwen 模型/tokenizer 文件；原协议未改写。',
            '', '原始产物目录：`artifacts/care/lora-aiops-paper128-v2/`；训练数据：`data/care/sft-aiops-paper128-v2/`。']
    bc=pred['base'].argmax(1)==labels;lc=pred['lora_42'].argmax(1)==labels;fc=pred['base_plus_lora_42'].argmax(1)==labels
    covered=np.array([y in np.argsort(-p,kind='stable')[:5] for p,y in zip(pred['base'],labels)])
    diagnostics={'base_errors':int((~bc).sum()),'lora_fixes_base_errors':int((~bc&lc).sum()),
                 'lora_fixes_outside_base_top5':int((~bc&lc&~covered).sum()),
                 'lora_fixes_not_recovered_by_fixed_fusion':int((~bc&lc&~fc).sum()),
                 'lora_breaks_base_correct':int((bc&~lc).sum()),
                 'lora_order_top1_disagreements_vs_primary':{str(s):int((pred[f'lora_{s}'].argmax(1)!=pred['lora_42'].argmax(1)).sum()) for s in [43,44]}}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    lines+=['','## 修正行为核对','',
            f"基础模型的 {diagnostics['base_errors']} 个错误中，独立 LoRA 修复 {diagnostics['lora_fixes_base_errors']} 个；其中 {diagnostics['lora_fixes_outside_base_top5']} 个真实根因位于基础模型前 5 候选之外。独立 LoRA 的这些修复中，{diagnostics['lora_fixes_not_recovered_by_fixed_fusion']} 个未转化为固定融合的正确预测。独立 LoRA 同时误改基础模型原先正确的 {diagnostics['lora_breaks_base_correct']} 个事件。",'',
            f"相对主排列，排列 43 和 44 分别改变 {diagnostics['lora_order_top1_disagreements_vs_primary']['43']}、{diagnostics['lora_order_top1_disagreements_vs_primary']['44']} 个事件的 LoRA Top-1 输出。这是排列敏感性观测，不是重新选模型的依据。"]
    report='\n'.join(lines)+'\n';(out/'REPORT.md').write_text(report);Path('AIOPS_LORA_V2_RESULTS.md').write_text(report)
    with (out/'cases.csv').open('w') as f:
        names=['base','evidence','unadapted_42','lora_42','base_plus_lora_42','evidence_plus_lora_42']
        writer=csv.writer(f);writer.writerow(['index','incident_id','root_service','fault_type']+names)
        for j,i in enumerate(te):writer.writerow([int(i),rows[i]['incident_id'],rows[i]['root_service'],rows[i]['fault_type']]+[services[int(pred[n][j].argmax())] for n in names])
    print(json.dumps(verification,indent=2));print(report)


if __name__=='__main__':main()
