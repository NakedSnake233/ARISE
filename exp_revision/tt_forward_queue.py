"""Fixed chronological TT validation, bounded by inherited compute/disk limits."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess
import time

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q
from . import tt_pair_queue as prior
from .disk_guard_fast import DiskGuard, InsufficientSpace, guarded_command

FOLDER = ROOT / 'results_v2/TT_forward_correction'
SPLIT = ROOT / 'splits/tt_forward_90_60.json'
SCENES = ['S0_none', 'S3v2_all_0.10', 'S3v2_all_0.25', 'S3v2_all_0.50']
COMPONENTS = [('context', None), ('experts', None), ('baseline', 'Eadro'),
              ('baseline', 'Eadro+MD'), ('baseline', 'MULAN')]
SOURCES = ['exp_revision/tt_forward_queue.py', 'exp_revision/tt_forward_data.py',
           'exp_revision/tt_forward_models.py', 'exp_revision/tt_forward_collect.py',
           'exp_revision/metrics.py', 'exp_revision/generalization_data.py',
           'exp_revision/generalization_models.py', 'exp_revision/remaining_queue.py',
           'exp_revision/s3v2_infer.py', 'splits/tt_forward_90_60.json',
           'splits/tt_forward_training_labels/fold_0.json',
           'exp_revision/baseline_settings.json', 'configs/care/review-response-v1.json',
           'exp_revision/baseline_fold.py', 'exp_revision/train_evidence.py',
           'exp_revision/train_variant_evidence.py', 'exp_revision/train_context_variant.py',
           'exp_revision/masked_evidence.py', 'exp_revision/raw_spans_v2.py',
           'exp_revision/preprocess_tt.py', 'exp_revision/preprocess_aiops.py',
           'exp_revision/drain_bitset_compact.py', 'exp_revision/fold_data.py',
           'exp_revision/common.py', 'exp_revision/scenario_v2.py',
           'scripts/care/fse2027_neural_baselines.py', 'scripts/care/fse2027_baseline_common.py',
           'scripts/care/evidence_reranker.py', 'scripts/care/robust_evidence.py',
           'results_v2/S3v2_protocol/masks_manifest.json']


def verify_sources():
    manifest = json.loads((FOLDER / 'queue_manifest.json').read_text())
    for relative, digest in manifest['source_sha256'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError('frozen forward input changed: ' + relative)


def identifier(component, seed=None, method=None):
    if component == 'prepare':
        return 'forward_prepare'
    label = (method or component).replace('+', '')
    return f'forward_{label}_s{seed}'


def specification(component, seed=None, method=None):
    part = json.loads(SPLIT.read_text())['fold_0']
    ids = part['test']
    cfg = dict(task='TFWD', dataset='tt', fold=0, smoke=False, split=part['split'],
               component='train_only_preprocessing' if component == 'prepare' else component)
    if component == 'prepare':
        command = ['-m', 'exp_revision.tt_forward_data', '--prepare', '--fold', '0']
        shape = [1, len(ids), 41]
        status = 'completed_generalization_preprocessing'
    else:
        cfg.update(seed=seed, scenarios=SCENES)
        command = ['-m', 'exp_revision.tt_forward_models', '--component', component,
                   '--fold', '0', '--seed', str(seed)]
        if method:
            cfg['method'] = method
            command += ['--method', method]
        shape = [4, len(ids), 41]
        if component == 'experts':
            shape = [4, *shape]
        status = f'completed_generalization_{component}'
    return dict(id=identifier(component, seed, method), command=command,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                resource='gpu' if component == 'context' else 'cpu',
                cores=4 if component in ('context', 'experts') else 1,
                timeout_seconds=1800 if component == 'context' else 1200 if component in ('prepare', 'experts') else 600,
                max_attempts=2, expected_config=cfg, test_ids=ids, score_shape=shape,
                expected_status=[status], dependencies=[] if component == 'prepare' else ['forward_prepare'])


def budget():
    base = prior.budget()
    pair = json.loads((prior.FOLDER / 'queue_state.json').read_text())
    scope = json.loads((ROOT / 'docs/TT_ONLY_SCOPE_20261003.json').read_text())
    if not base['approved'] or scope['allowed_datasets'] != ['tt']:
        raise ValueError('missing existing compute authorization or TT scope policy')
    return dict(cpu_limit=base['cpu_limit'], gpu_limit=base['gpu_limit'],
                prior_cpu=base['prior_cpu'] + prior.charged(pair, True),
                prior_gpu=base['prior_gpu'] + prior.charged(pair, True, True),
                basis='All previous actual charges plus unreleased legacy/S3v2 commitments; user requested TT protocol correction')


def maximum_remaining(state, gpu=False):
    """Full deadlines for running attempts and every allowed future attempt."""
    total = 0.
    for row in state['jobs'].values():
        spec = row['spec']
        if (gpu and spec['resource'] != 'gpu') or row['status'] == 'completed':
            continue
        future = max(0, spec['max_attempts'] - len(row['attempts']))
        active = sum('elapsed_seconds' not in a for a in row['attempts'])
        total += (future + active) * (spec['timeout_seconds'] + 10) * (1 if gpu else spec['cores']) / 3600
    return total


def pilot_ready(state):
    rows = [state['jobs'][identifier(c, 42, m)] for c, m in COMPONENTS]
    return all(r['status'] == 'completed' and q.verified(ROOT / r['output'], r['spec']) for r in rows)


def initialize():
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = FOLDER / 'queue_state.json'
    state = json.loads(path.read_text()) if path.exists() else dict(created=q.now(), backend='local', jobs={})
    manifest_path = FOLDER / 'queue_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for relative, digest in manifest['source_sha256'].items():
            if sha256(ROOT / relative) != digest:
                raise ValueError('frozen forward input changed: ' + relative)
        for job in manifest['jobs']:
            if job['id'] not in state['jobs']:
                # Recover a crash between immutable manifest publication and the
                # first state write, but never erase evidence of a prior attempt.
                occupied = (ROOT / 'results_v2/remaining_runs' / job['id']).exists()
                occupied |= any((ROOT / 'logs').glob('remaining_' + job['id'] + '_attempt*.log'))
                if occupied:
                    raise ValueError('missing state has prior attempt artifacts: ' + job['id'])
                state['jobs'][job['id']] = dict(spec=job, status='pending', attempts=[])
            if state['jobs'][job['id']]['spec'] != job:
                raise ValueError('frozen job specification changed')
    else:
        jobs = [specification('prepare')]
        jobs += [specification(c, s, m) for s in range(42, 47) for c, m in COMPONENTS]
        for job in jobs:
            state['jobs'][job['id']] = dict(spec=job, status='pending', attempts=[])
        manifest = dict(created=q.now(), protocol='chronological_90_60',
                        acceptance='receipt/coverage correctness only; no score threshold',
                        source_sha256={p: sha256(ROOT / p) for p in SOURCES}, jobs=jobs)
        json_dump(manifest_path, manifest)
    b = budget()
    bounds = dict(cpu=maximum_remaining(state), gpu=maximum_remaining(state, True))
    state.update(budget=b, maximum_remaining_hours=bounds,
                 stage='full' if pilot_ready(state) else 'pilot',
                 cpu_hours=prior.charged(state), gpu_hours=prior.charged(state, gpu=True))
    if (b['prior_cpu'] + state['cpu_hours'] + bounds['cpu'] > b['cpu_limit'] or
            b['prior_gpu'] + state['gpu_hours'] + bounds['gpu'] > b['gpu_limit']):
        state['status'] = 'budget_limited'
    else:
        state['status'] = 'planned'
    json_dump(path, state)
    return state


def summary(state):
    return {k: v for k, v in state.items() if k != 'jobs'}


def run(plan_only=False):
    FOLDER.mkdir(parents=True, exist_ok=True)
    q.FOLDER = FOLDER
    q.guarded_command = guarded_command
    with (FOLDER / '.queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='already_running')
        state = initialize()
        if plan_only or state['status'] == 'budget_limited':
            return summary(state)
        guard = DiskGuard(ROOT)
        processes = {}
        for row in state['jobs'].values():
            if row['status'] == 'running' and q.alive(row['attempts'][-1]):
                a = row['attempts'][-1]
                guard.reserve(a['key'], ROOT/a['output'], ROOT/a['log'], child_pid=a['pid'])
        while True:
            for name, row in state['jobs'].items():
                if row['status'] == 'running' and not q.alive(row['attempts'][-1]):
                    q.finish(row['spec'], row, row['attempts'][-1], processes.pop(name, None), guard)
            # Seed 42 traverses the complete pipeline before other seeds are released.
            ready = pilot_ready(state)
            state['stage'] = 'full' if ready else 'pilot'
            storage = guard.poll()
            b = budget()
            failed = any(r['status'] == 'stuck' for r in state['jobs'].values())
            limited = False
            storage_limited = not storage['safe_to_launch']
            for name, row in state['jobs'].items():
                spec = row['spec']
                if row['status'] != 'pending' or failed or time.time() < row.get('retry_after', 0):
                    continue
                if spec['expected_config'].get('seed', 42) != 42 and not ready:
                    continue
                if any(state['jobs'][d]['status'] != 'completed' for d in spec['dependencies']):
                    continue
                active = [r for r in state['jobs'].values() if r['status'] == 'running']
                if len(active) >= 6 or not storage['safe_to_launch']:
                    break
                if spec['resource'] == 'gpu':
                    if sum(r['spec']['resource'] == 'gpu' for r in active) >= 3:
                        continue
                    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
                    if int(free.strip()) < 8192:
                        continue
                wall = (spec['timeout_seconds'] + 10) / 3600
                if b['prior_cpu'] + prior.charged(state, True) + wall * spec['cores'] > b['cpu_limit']:
                    limited = True
                    continue
                if spec['resource'] == 'gpu' and b['prior_gpu'] + prior.charged(state, True, True) + wall > b['gpu_limit']:
                    limited = True
                    continue
                verify_sources()
                try:
                    processes[name] = q.start(spec, row, state, guard)
                except InsufficientSpace:
                    storage_limited = True
                    break
                storage = guard.poll()
                storage_limited = not storage['safe_to_launch']
            active = any(r['status'] == 'running' for r in state['jobs'].values())
            complete = all(r['status'] == 'completed' for r in state['jobs'].values())
            state.update(updated=q.now(), budget=b, storage=storage,
                         cpu_hours=prior.charged(state), gpu_hours=prior.charged(state, gpu=True),
                         counts=dict(Counter(r['status'] for r in state['jobs'].values())),
                         maximum_remaining_hours=dict(cpu=maximum_remaining(state), gpu=maximum_remaining(state, True)))
            state['status'] = ('running' if active else 'stuck' if failed else 'completed' if complete else
                               'budget_limited' if limited else 'storage_limited' if storage_limited else 'waiting')
            json_dump(FOLDER/'queue_state.json', state)
            if state['status'] not in ('running', 'waiting'):
                return summary(state)
            for process in processes.values():
                process.poll()
            time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.plan_only)))
