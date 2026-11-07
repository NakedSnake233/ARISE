"""Score the frozen chronological TT test; labels enter only this collector."""
import argparse
from collections import Counter
import datetime as dt
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, dataset_dir, json_dump, sha256
from .remaining_queue import verified
from .scenario_v2 import load_cases, mask_path, OUTPUT as MASK_ROOT
from .s3v2_infer import SCENES
from .s3v2_report import table
from .train_evidence import fuse_all
from .metrics import ranking_metrics

FOLDER = ROOT / 'results_v2/TT_forward_correction'
SPLIT = ROOT / 'splits/tt_forward_90_60.json'
SEEDS = set(range(42, 47))
EXPERTS = ['ET-spec', 'ET-spec+gate+aug', 'RootFreqPrior', 'Random']
ADAPTED = {'Eadro': 'Eadro-adapted', 'Eadro+MD': 'Eadro+MD-adapted', 'MULAN': 'MULAN-adapted'}
EXPECTED_METHODS = {'ARISE', 'context-only', *EXPERTS, *ADAPTED.values()}
FULL = 'full_forward60_5seed'
PREDICTION_COLUMNS = ['dataset', 'method', 'fold', 'seed', 'incident_id', 'scenario',
    'scenario_param', 'rank_of_root', 'stratum', 'deleted', 'original_available',
    'deleted_services', 'eligible_services', 'source_job']


def expected_jobs():
    jobs = {'forward_prepare': ('train_only_preprocessing', None, None)}
    for seed in sorted(SEEDS):
        for suffix, component, method in [('context', 'context', None), ('experts', 'experts', None),
                ('Eadro', 'baseline', 'Eadro'), ('EadroMD', 'baseline', 'Eadro+MD'), ('MULAN', 'baseline', 'MULAN')]:
            jobs[f'forward_{suffix}_s{seed}'] = (component, method, seed)
    return jobs


def validate_spec(name, row, ids, split):
    component, method, seed = expected_jobs()[name]
    cfg = row['spec']['expected_config']
    expected = dict(task='TFWD', dataset='tt', fold=0, smoke=False,
                    split=split, component=component)
    if seed is not None:
        expected.update(seed=seed, scenarios=list(SCENES))
    if method is not None:
        expected['method'] = method
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('unexpected forward job configuration: ' + name)
    shape = ([1, 60, 41] if seed is None else [4, 4, 60, 41] if component == 'experts' else [4, 60, 41])
    if row['spec']['test_ids'] != ids or list(row['spec']['score_shape']) != shape:
        raise ValueError('unexpected test identity or shape: ' + name)
    return component, method, seed


def load_preparation(row, part, services, original):
    folder = ROOT / row['output']
    if not verified(folder, row['spec']):
        raise ValueError('unverified forward preprocessing')
    meta = json.loads((folder / 'run_meta.json').read_text())
    for filename, field in [('metadata.json', 'metadata_sha256'), ('features.npz', 'features_sha256')]:
        if sha256(folder / filename) != meta[field]:
            raise ValueError('preprocessing changed: ' + filename)
    info = json.loads((folder / 'metadata.json').read_text())
    if info['train_ids'] != part['train'] or info['test_ids'] != part['test'] or info['services'] != services:
        raise ValueError('preprocessing differs from frozen chronological split/catalogue')
    with np.load(folder / 'features.npz', allow_pickle=False) as z:
        availability = z['availability'].copy()
    if availability.shape != (150, 41, 3) or not np.isin(availability, (0, 1)).all():
        raise ValueError('invalid preprocessing availability')
    for key, observed in zip(part['test'], availability[90:]):
        if not np.array_equal(observed, original[key]):
            raise ValueError('shared S3v2 original availability differs: ' + key)
    return folder


def collect(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    state_path = FOLDER / 'queue_state.json'
    state = json.loads(state_path.read_text())
    parts = json.loads(SPLIT.read_text())
    if set(parts) != {'fold_0'}:
        raise ValueError('forward split must have exactly fold_0')
    part = parts['fold_0']; ids = part['test']
    if (len(part['train']) != 90 or len(ids) != 60 or len(set(part['train'] + ids)) != 150
            or part['split'] != 'chronological_90_60'):
        raise ValueError('invalid fixed 90/60 cohort')
    jobs = state['jobs']; expected = expected_jobs()
    if set(jobs) - set(expected):
        raise ValueError('unexpected jobs in forward queue')
    inputs = {str(SPLIT.relative_to(ROOT)): sha256(SPLIT),
              str(state_path.relative_to(ROOT)): sha256(state_path)}
    manifest_path = FOLDER / 'queue_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = manifest['source_sha256']
    if not sources:
        raise ValueError('empty frozen source closure')
    for relative, digest in sources.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError('frozen source/configuration changed: ' + relative)
    manifest_jobs = manifest['jobs']
    if isinstance(manifest_jobs, list):
        if len({job['id'] for job in manifest_jobs}) != len(manifest_jobs):
            raise ValueError('duplicate manifest job')
        manifest_jobs = {job['id']: job for job in manifest_jobs}
    if set(manifest_jobs) != set(expected) or any(row['spec'] != manifest_jobs[name] for name, row in jobs.items()):
        raise ValueError('queue specifications differ from frozen manifest')
    inputs[str(manifest_path.relative_to(ROOT))] = sha256(manifest_path)
    coverage = []; blocks = {}
    services, cases, _ = load_cases('tt')
    if len(services) != 41 or len(set(services)) != 41:
        raise ValueError('unexpected TT service catalogue')
    original = {key: a for _, key, a in cases}
    for name, identity in expected.items():
        row = jobs.get(name)
        if row is None:
            coverage.append(dict(job=name, status='absent', component=identity[0], seed=identity[2]))
            continue
        component, method, seed = validate_spec(name, row, ids, part['split'])
        coverage.append(dict(job=name, status=row['status'], **row['spec']['expected_config']))
        if row['status'] != 'completed':
            continue
        folder = ROOT / row['output']
        if not verified(folder, row['spec']):
            raise ValueError('unverified completed artifact: ' + name)
        receipt_path = folder / 'run_meta.json'
        inputs[str(receipt_path.relative_to(ROOT))] = sha256(receipt_path)
        if component == 'train_only_preprocessing':
            load_preparation(row, part, services, original)
            continue
        if jobs.get('forward_prepare', {}).get('status') != 'completed':
            raise ValueError('completed model lacks completed preprocessing')
        meta = json.loads(receipt_path.read_text())
        if meta.get('input_sha256', {}).get(str(SPLIT.relative_to(ROOT))) != sha256(SPLIT):
            raise ValueError('model receipt missing matching frozen split: ' + name)
        with np.load(folder / 'scores.npz', allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        if arrays['incident_ids'].tolist() != ids or arrays['scenario_keys'].tolist() != SCENES:
            raise ValueError('artifact identity/scenes changed: ' + name)
        if component == 'experts':
            if arrays['method_names'].tolist() != EXPERTS:
                raise ValueError('unexpected expert methods')
            if arrays['evidence'].shape != (4, 60, 41) or not np.isfinite(arrays['evidence']).all():
                raise ValueError('invalid evidence branch')
        blocks.setdefault(seed, {})[method or component] = (arrays, name, meta)

    # Labels are read after model receipts and identifiers have been accepted.
    label_path = dataset_dir('tt') / 'incidents.json'
    records = json.loads(label_path.read_text())
    labels = {r['incident_id']: services.index(r['root_service']) for r in records}
    chronological = sorted(records, key=lambda r: (r['start'], r['end'], r['incident_id']))
    if ([r['incident_id'] for r in chronological[:90]] != part['train']
            or [r['incident_id'] for r in chronological[90:]] != ids
            or max(r['end'] for r in chronological[:90]) > min(r['start'] for r in chronological[90:])):
        raise ValueError('cohort is not the frozen nonoverlapping chronological 90/60 split')
    inputs[str(label_path.relative_to(ROOT))] = sha256(label_path)
    pair = lambda record: (record['root_service'], record['fault_type'])
    train_pairs = set(map(pair, chronological[:90])); test_pairs = set(map(pair, chronological[90:]))
    seen_pair_events = sum(pair(record) in train_pairs for record in chronological[90:])
    mask_manifest_path = MASK_ROOT / 'masks_manifest.json'
    mask_manifest = json.loads(mask_manifest_path.read_text())['datasets']['tt']
    if mask_manifest['services'] != services:
        raise ValueError('mask service catalogue changed')
    inputs[str(mask_manifest_path.relative_to(ROOT))] = sha256(mask_manifest_path)
    rows = []
    for seed, components in sorted(blocks.items()):
        masks = [np.ones_like(np.stack([original[key] for key in ids]))]; hashes = {}
        for scene in SCENES[1:]:
            path = mask_path(MASK_ROOT, 'tt', float(scene.rsplit('_', 1)[1]), seed)
            digest = mask_manifest['files'][str(path.relative_to(MASK_ROOT))]
            if sha256(path) != digest:
                raise ValueError('frozen S3v2 mask changed')
            with np.load(path, allow_pickle=False) as z:
                mask = np.stack([z[key] for key in ids])
            if mask.shape != (60, 41, 3) or not np.isin(mask, (0, 1)).all():
                raise ValueError('invalid shared S3v2 mask')
            masks.append(mask); hashes[str(path.relative_to(ROOT))] = digest
        inputs.update(hashes)
        methods = {}; ctx = components.get('context')
        for kind, (z, source, meta) in components.items():
            if meta['mask_sha256'] != hashes:
                raise ValueError('model did not use the exact shared masks: ' + source)
            if kind == 'context':
                methods['context-only'] = z['scores'], source
            elif kind == 'experts':
                methods.update({method: (scores, source) for method, scores in zip(EXPERTS, z['scores'])})
                if ctx is not None:
                    methods['ARISE'] = (np.stack([fuse_all((.25, c), (.75, e))
                        for c, e in zip(ctx[0]['scores'], z['evidence'])]), ctx[1] + '|' + source)
            else:
                methods[ADAPTED[kind]] = z['scores'], source
        for method, (scores, source) in methods.items():
            if scores.shape != (4, 60, 41) or not np.isfinite(scores).all():
                raise ValueError('invalid final scores: ' + method)
            for scene_index, scene in enumerate(SCENES):
                ranked = np.argsort(-scores[scene_index], axis=-1, kind='stable')
                for i, key in enumerate(ids):
                    a0 = original[key]; mask = masks[scene_index][i]; a = a0 * mask; target = labels[key]
                    rows.append(dict(dataset='tt', method=method, fold=0, seed=seed, incident_id=key,
                        scenario='S0' if scene_index == 0 else 'S3v2_all',
                        scenario_param='none' if scene_index == 0 else scene.rsplit('_', 1)[1],
                        rank_of_root=int(np.flatnonzero(ranked[i] == target)[0] + 1),
                        stratum='root_originally_unobserved' if not a0[target].any() else
                            'root_newly_missing' if not a[target].any() else 'root_still_observed',
                        deleted=int((a0 * (1 - mask)).sum()), original_available=int(a0.sum()),
                        deleted_services=int((a0.any(1) & ~a.any(1)).sum()),
                        eligible_services=int(a0.any(1).sum()), source_job=source))
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    if frame.duplicated(['method', 'seed', 'incident_id', 'scenario_param']).any():
        raise ValueError('duplicate prediction key')
    summaries = []; per_seed = []
    for (method, param), group in frame.groupby(['method', 'scenario_param']):
        complete = (len(group) == 300 and set(group.seed) == SEEDS
                    and all(set(g.incident_id) == set(ids) and len(g) == 60 for _, g in group.groupby('seed')))
        for stratum in ('all', 'root_newly_missing', 'root_still_observed', 'root_originally_unobserved'):
            subset = group if stratum == 'all' else group[group.stratum == stratum]
            metric = ranking_metrics(subset.rank_of_root.to_numpy()) if len(subset) else {
                key: None for key in ('PR@1', 'PR@3', 'PR@5', 'MAP@3', 'MAP@5', 'MRR')}
            summaries.append(dict(method=method, scenario_param=param, stratum=stratum,
                coverage=FULL if complete else 'partial', event_seed_rows=len(subset), events=subset.incident_id.nunique(),
                actual_cell_deletion_rate=subset.deleted.sum() / max(1, subset.original_available.sum()),
                actual_service_deletion_fraction=subset.deleted_services.sum() / max(1, subset.eligible_services.sum()), **metric))
        for seed, subset in group.groupby('seed'):
            per_seed.append(dict(method=method, scenario_param=param, seed=seed,
                Top1=float(subset.rank_of_root.eq(1).mean()), events=len(subset)))
    summary = pd.DataFrame(summaries)
    valid = summary[summary.stratum == 'all'].copy() if len(summary) else pd.DataFrame()
    complete = (set(jobs) == set(expected) and all(row['status'] == 'completed' for row in jobs.values())
                and len(valid) == len(EXPECTED_METHODS) * 4 and set(valid.method) == EXPECTED_METHODS
                and set(valid.coverage) == {FULL})
    if state.get('status') == 'completed' and not complete:
        raise ValueError('terminal queue has incomplete jobs/methods/scenes; no final receipt published')
    output.mkdir(parents=True)
    frame.to_parquet(output / 'predictions.parquet', index=False)
    summary.to_csv(output / 'metrics.csv', index=False)
    seed_frame = pd.DataFrame(per_seed)
    seed_frame.to_csv(output / 'seed_metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'job_coverage.csv', index=False)
    counts = dict(Counter(row['status'] for row in jobs.values()))
    lines = ['# TT 前向时间划分补充验证', '', f'任务状态：{counts}；完整结果要求26任务、9方法×4场景×60事件×5种子。', '']
    if len(valid):
        descriptive = seed_frame.groupby(['method', 'scenario_param']).Top1.agg(['mean', 'std', 'count']).reset_index()
        descriptive.to_csv(output / 'seed_top1_descriptive.csv', index=False)
        valid['Top1 (%)'] = valid['PR@1'] * 100
        display = valid.pivot(index=['method', 'coverage'], columns='scenario_param', values='Top1 (%)')
        lines += [table(display.reindex(columns=['none', '0.10', '0.25', '0.50']).round(2).reset_index()), '']
    else:
        lines += ['尚无完成的模型预测。', '']
    lines += ['冻结时间最早90事件训练、最后60事件测试；这是同次采集、已查看150事件上的事后前向敏感性分析，不是新独立数据集。', '',
        f'训练含{len(train_pairs)}个服务×故障类型组合，测试含{len(test_pairs)}个；{seen_pair_events}/60个测试事件（{seen_pair_events / 60:.0%}）的组合在训练中出现。本协议不隔离组合。', '',
        '预处理只用外层训练事件；context和神经适配器的内部训练/验证按组合分组，专家保留原故障类型留出beta选择。ARISE融合为0.25 context+0.75 evidence。Eadro/MULAN均为适配实现；本表不宣称复现其官方基准。', '',
        'S0表示无额外删除；S3v2使用原有与根因标签无关的共享掩码。实际删除率以删除数/原可用数汇总，根因观测分层见metrics.csv。', '',
        '五种子重复评估同60事件，不是300个独立故障；种子间标准差见seed_top1_descriptive.csv。所有分数保留，没有目标精度门槛。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    json_dump(output / 'receipt.json', dict(status='complete_forward_evaluation' if complete else 'partial_forward_evaluation',
        queue_counts=counts, rows=len(frame), independent_test_events=60, input_receipts=inputs,
        test_seen_pair_events=seen_pair_events, test_seen_pair_fraction=seen_pair_events / 60,
        predictions_sha256=sha256(output / 'predictions.parquet'), source_sha256=sha256(Path(__file__))))
    if complete and state.get('status') == 'completed':
        json_dump(FOLDER / 'final_evaluation.json', dict(report=str((output / 'REPORT.md').relative_to(ROOT)),
            rows=len(frame), methods=9, scenes=4, test_events=60, seeds=sorted(SEEDS)))
    return str(output / 'REPORT.md')


if __name__ == '__main__':
    from .disk_guard_fast import DiskGuard
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path); args = parser.parse_args()
    with (FOLDER / '.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (FOLDER / 'final_evaluation.json').exists() and args.output is None:
            print((FOLDER / 'final_evaluation.json').read_text())
        else:
            output = args.output or ROOT / 'results_v2' / ('TT_forward_evaluation_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
            guard = DiskGuard(ROOT); guard.reserve('tt_forward_collection', output, ROOT / 'logs/tt_forward_collection.log')
            try:
                print(collect(output))
            finally:
                guard.release('tt_forward_collection')
