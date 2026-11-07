"""Render only observed S3v2 results, separating full coverage and pilots."""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import pandas as pd
from .common import ROOT, json_dump, sha256


def table(frame):
    frame = frame.fillna('—').astype(str)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in frame.itertuples(index=False, name=None)])


def render(folder):
    metrics = pd.read_csv(folder / 'metrics.csv', keep_default_na=False)
    coverage = pd.read_csv(folder / 'job_coverage.csv')
    receipt = json.loads((folder / 'receipt.json').read_text())
    main = metrics[(metrics.stratum == 'all') & ~metrics.invalid_by_design.astype(bool)]
    full = main[main.coverage == 'full_5fold_5seed'].copy()
    full['column'] = full.scenario_param.map({'none': 'S0', '0.10': 'q=0.10', '0.25': 'q=0.25', '0.50': 'q=0.50'})
    full['Top1 (%)'] = pd.to_numeric(full['PR@1']) * 100
    lines = ['# S3v2 冻结模型与基线重算', '',
             f'本报告对应 {folder.relative_to(ROOT)} 的不可变汇总快照。', '',
             f'已纳入预测 {receipt["prediction_rows"]:,} 行；快照任务状态：{json.dumps(receipt["queue_counts"], ensure_ascii=False)}。', '',
             '计划矩阵为 125 个冻结模型任务和 250 个可运行官方基线任务，共 375 个。任务数不是方法数：每个冻结模型任务包含多个方法。未完成或受限项不能视为完整验证。', '',
             '协议随机删除合格服务的全部观测，与根因身份无关；固定共享掩码、五折和 seeds42–46。q 是请求的服务比例，实际比例受最少删除两个服务、取整及原始可用性影响。所有主表均包含全体事件；失败事件保留在分母中，排名指标贡献为零。', '',
             '## 实际删除率', '']
    rates = full[(full.method == 'ARISE') & (full.scenario == 'S3v2_all')]
    out = rates[['dataset', 'scenario_param', 'actual_cell_deletion_rate', 'actual_service_deletion_fraction']].copy()
    for col in out.columns[2:]:out[col] = (pd.to_numeric(out[col]) * 100).map(lambda x: f'{x:.2f}%')
    lines += [table(out), '', '## 全覆盖主表：Top1（%）', '',
              '只列达到 5 折×5 种子的完整方法。S0 为同一冻结模型的完整输入对照。V-a 是有意保留的泄漏诊断，不进入有效主表。']
    for ds, group in full.groupby('dataset'):
        values = group.pivot(index='method', columns='column', values='Top1 (%)').reindex(columns=['S0', 'q=0.10', 'q=0.25', 'q=0.50']).round(2).reset_index()
        lines += ['', f'### {ds}', '', table(values)]
    lines += ['', '## 覆盖不足及失败记录', '']
    partial = main[(main.coverage != 'full_5fold_5seed') & (main.scenario_param == '0.10')]
    lines += [table(partial[['dataset', 'method', 'coverage', 'events', 'event_seed_rows']]), ''] if len(partial) else ['本快照中已产出的方法均达到完整覆盖；以下研发试运行和外部受限项仍需单列。', '']
    failures = main[pd.to_numeric(main.failures) > 0]
    lines += [table(failures[['dataset', 'method', 'scenario_param', 'failures', 'event_seed_rows', 'failure_rate']]), ''] if len(failures) else ['当前已汇总产物没有算法失败事件。', '']
    statuses = coverage.groupby(['dataset', 'component', 'status']).size().rename('jobs').reset_index()
    lines += [table(statuses), '',
              '原版 CIRCA 的既有前置实验失败记录保持受限；CIRCA+rank-prune 是明确披露的去冗余变体，不能冒充原版。AIOps 的该变体仍受此前失败门槛限制。RE2-OB/TT 的 RCD 既有超时限制没有用其他算法替代。', '',
              'ReAct/mABC 尚未完成新场景评估。2026-10-02 10:24 UTC 的单次无实验数据健康检查仍返回 HTTP 429 / Go usage limit exceeded；保留原授权，不切换模型、不循环重试。原 S0 结果不能替代新缺失场景结果。', '',
              'RE2-SS/RCD 只有历史 fold0/seed42 单事件可比参考，其他 S0 是本轮新计算，不能宣称全折历史复现。', '',
              '## 改进版的同事件试运行比较', '',
              '仅 TT/AIOps 的 fold0、seed42；各方法使用相同事件。不能与五折主表跨范围比较，也不能以这一试运行支持完整改进有效性。']
    pilot = pd.read_csv(folder / 'paired_pilot_metrics.csv', keep_default_na=False)
    methods = ['ARISE', 'ARISE[V-b]', 'ARISE[V-d]', 'ET-spec', 'ARISE-R1', 'ARISE-R2', 'ARISE-R3', 'ARISE-R3-no-block', 'ARISE-R3-small-cal']
    pilot = pilot[pilot.method.isin(methods)].copy()
    pilot['Top1 (%)'] = pd.to_numeric(pilot['PR@1']) * 100
    for ds, group in pilot.groupby('dataset'):
        values = group.pivot(index='method', columns='scenario_param', values='Top1 (%)').reindex(columns=['none', '0.10', '0.25', '0.50']).round(2).reset_index()
        lines += ['', f'### {ds}', '', table(values)]
    lines += ['', '## 解释范围与可追溯性', '',
              '这些是描述性性能结果。相同事件的五个种子不构成五个独立事件，本报告不给出伪独立显著性结论。根因新增缺失分层仍含条件性的缺失位置提示，不能单独证明鲁棒性；完整分层及显式零样本计数见 metrics.csv。', '',
              '新推理先遮掩输入并重算派生证据；所有存在完整历史对照的方法均经过固定容差 S0 一致性检查。未新增训练、调测试集参数或选择种子。预测及分数来源校验见 receipt.json，逐事件分母见 predictions.parquet。', '',
              f'本轮任务累计 CPU {receipt.get("cpu_hours", 0):.3f} 核小时、GPU {receipt.get("gpu_hours", 0):.3f} 任务小时（快照值；包含保留的失败尝试）。总预算上限 CPU300 / GPU56，旧任务与待执行承诺由队列一并计入。', '']
    (folder / 'REPORT.md').write_text('\n'.join(lines))
    json_dump(folder / 'report_receipt.json', dict(source_sha256=sha256(Path(__file__)), metrics_sha256=sha256(folder / 'metrics.csv'), report_sha256=sha256(folder / 'REPORT.md')))
    return folder / 'REPORT.md'


def finalize():
    from .s3v2_queue import FOLDER
    from .s3v2_collect import collect
    from .disk_guard_fast import DiskGuard
    with (FOLDER / '.finalize.lock').open('a') as lock:
        try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:return {'status': 'finalizer_already_running'}
        pointer = FOLDER / 'final_evaluation.json'
        if pointer.exists():return json.loads(pointer.read_text())
        state = json.loads((FOLDER / 'queue_state.json').read_text())
        if len(state['jobs']) != 375 or any(r['status'] not in ('completed', 'scientific_blocked') for r in state['jobs'].values()):
            raise ValueError('matrix not complete or scoped-limited; use an explicitly interim collection')
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        folder = ROOT / 'results_v2' / ('S3v2_evaluation_final_' + stamp)
        guard = DiskGuard(ROOT); key = 's3v2_final_collection'
        guard.reserve(key, folder, ROOT / 'logs/s3v2_final_collection.log')
        try:
            collect(folder); report = render(folder)
            result = dict(status='completed_available_matrix', report=str(report.relative_to(ROOT)), queue_counts=state['counts'])
            json_dump(pointer, result)
            return result
        finally:guard.release(key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--folder', type=Path); parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    print(json.dumps(finalize()) if args.finalize else render(args.folder.resolve()))
