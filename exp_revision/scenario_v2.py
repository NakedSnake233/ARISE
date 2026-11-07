"""Generate frozen root-blind masks, then separately audit label-based outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .common import ROOT, SEEDS, dataset_dir, sha256
from .missingness_v2 import VERSION, RATES, make_service_mask

DATASETS = ('tt', 'aiops', 're2_ob', 're2_ss', 're2_tt')
OUTPUT = ROOT / 'results_v2/S3v2_protocol'


def immutable_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'refusing to replace different artifact: {path}')
    else:
        with path.open('x') as stream:
            stream.write(content)


def immutable_json(path, value):
    immutable_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def write_csv(path, rows):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    immutable_text(path, stream.getvalue())


def load_cases(dataset, root=ROOT):
    """Read only split identities, metadata and A0; never load a label manifest."""
    split_path = root / f'splits/{dataset}_kfold5.json'
    cohort_path = root / f'splits/{dataset}_unlabeled_events.json'
    split = json.loads(split_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    queue = None
    if dataset.startswith('re2_'):
        queue = json.loads((root / 'results_v2/remaining_queue/queue_state.json').read_text())
    cases, sources, services = [], [], None
    for fold in range(5):
        if queue is None:
            folder = root / f'cache/fold_preprocessing/{dataset}/fold{fold}'
        else:
            job = queue['jobs'][f're2_fold_{dataset}_f{fold}']
            if job['status'] != 'completed':
                raise ValueError('RE2 fold is not complete')
            folder = root / job['output']
        meta = json.loads((folder / 'metadata.json').read_text())
        part = split[f'fold_{fold}']
        if meta['train_ids'] != part['train'] or meta['test_ids'] != part['test']:
            raise ValueError('fold order differs from frozen split')
        if set(part['train']) & set(part['test']):
            raise ValueError('training/test overlap')
        if services is not None and services != meta['services']:
            raise ValueError('fold service order differs')
        services = meta['services']
        if not services or len(set(services)) != len(services):
            raise ValueError('empty or duplicate candidate services')
        with np.load(folder / 'features.npz', allow_pickle=False) as archive:
            a0 = archive['availability'].copy()
        if a0.shape != (len(part['train']) + len(part['test']), len(services), 3):
            raise ValueError('availability shape mismatch')
        if not np.isin(a0, (0, 1)).all():
            raise ValueError('nonbinary availability')
        a0 = a0.astype(np.uint8)
        for offset, key in enumerate(part['test'], start=len(part['train'])):
            cases.append((fold, key, a0[offset]))
        sources.append(dict(fold=fold, folder=str(folder.relative_to(root)),
            metadata_sha256=sha256(folder / 'metadata.json'),
            availability_sha256=hashlib.sha256(a0.tobytes()).hexdigest()))
    counts = Counter(key for _, key, _ in cases)
    if counts != Counter(row['incident_id'] for row in cohort) or any(v != 1 for v in counts.values()):
        raise ValueError('each cohort event must occur in exactly one test fold')
    return services, cases, dict(folds=sources, split_sha256=sha256(split_path),
                                cohort_sha256=sha256(cohort_path))


def mask_path(output, dataset, fraction, seed):
    return output / f'masks/{dataset}/{VERSION}_{fraction:.2f}/seed{seed}.npz'


def write_masks(path, masks):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(masks) or any(not np.array_equal(old[k], v) for k, v in masks.items()):
                raise FileExistsError(f'existing masks differ: {path}')
    else:
        with path.open('xb') as stream:
            np.savez_compressed(stream, **masks)


def generate(output=OUTPUT, datasets=DATASETS, root=ROOT):
    rows, manifests = [], {}
    for dataset in datasets:
        services, cases, sources = load_cases(dataset, root)
        files = {}
        for fraction in RATES:
            for seed in SEEDS:
                masks = {}
                for fold, key, a0 in cases:
                    k, rates = make_service_mask(dataset, key, a0, fraction, seed)
                    masks[key] = k
                    rows.append(dict(dataset=dataset, fold=fold, seed=seed, incident_id=key,
                        scenario=VERSION, scenario_param=f'{fraction:.2f}', **rates))
                path = mask_path(output, dataset, fraction, seed)
                write_masks(path, masks)
                files[str(path.relative_to(output))] = sha256(path)
        manifests[dataset] = dict(events=len(cases), services=services, sources=sources, files=files)
    write_csv(output / 'realized_rates.csv', rows)
    receipt = dict(protocol=VERSION, status='frozen_masks_not_model_results',
        fractions=list(RATES), seeds=list(SEEDS), datasets=manifests,
        generator_reads_root_labels=False,
        primary_population='all test events, no root-conditioned inclusion',
        source_code={name:sha256(ROOT / 'exp_revision' / name)
                     for name in ('missingness_v2.py', 'scenario_v2.py')})
    immutable_json(output / 'masks_manifest.json', receipt)
    return {ds: info['events'] for ds, info in manifests.items()}


def rule_scores(a0, mask, prior):
    a = a0 * mask
    missing = ~a.any(axis=1)
    new_missing = a0.any(axis=1) & missing
    restricted = np.where(new_missing, prior, -1.) if new_missing.any() else prior.copy()
    return dict(fully_missing_first=missing.astype(float),
        newly_missing_first=new_missing.astype(float),
        most_observed_first=a.sum(axis=1).astype(float),
        training_root_frequency=prior, newly_missing_training_prior=restricted)


def top1(values, root):
    ties = values == np.max(values)
    return int(np.argmax(values) == root), float(ties[root]) / int(ties.sum())


def audit(output=OUTPUT, root=ROOT):
    """Labels are permitted here only after complete immutable mask generation."""
    manifest_path = output / 'masks_manifest.json'
    receipt = json.loads(manifest_path.read_text())
    if receipt['protocol'] != VERSION:
        raise ValueError('wrong protocol')
    for info in receipt['datasets'].values():
        for name, digest in info['files'].items():
            if sha256(output / name) != digest:
                raise ValueError('frozen mask changed')
    # Verify every dataset and input before reading labels.
    loaded = {}
    for dataset, info in receipt['datasets'].items():
        services, cases, sources = load_cases(dataset, root)
        if services != info['services'] or sources != info['sources'] or len(cases) != info['events']:
            raise ValueError('audit inputs differ from mask generation')
        loaded[dataset] = (services, cases)
    rows, label_inputs = [], {}
    for dataset, (services, cases) in loaded.items():
        label_path = (root / f'splits/{dataset}_evaluation_labels.json' if dataset.startswith('re2_')
                      else root / dataset_dir(dataset).relative_to(ROOT) / 'incidents.json')
        label_inputs[str(label_path.relative_to(root))] = sha256(label_path)
        label_rows = json.loads(label_path.read_text())
        labels = {row['incident_id']:services.index(row['root_service']) for row in label_rows}
        if len(labels) != len(label_rows) or not {key for _, key, _ in cases}.issubset(labels):
            raise ValueError('duplicate/missing evaluation labels')
        priors = {}
        for fold in range(5):
            path = root / f'splits/{dataset}_training_labels/fold_{fold}.json'
            training = json.loads(path.read_text())
            split = json.loads((root / f'splits/{dataset}_kfold5.json').read_text())[f'fold_{fold}']
            if [r['incident_id'] for r in training] != split['train']:
                raise ValueError('frequency prior does not use exactly this training fold')
            label_inputs[str(path.relative_to(root))] = sha256(path)
            priors[fold] = np.bincount([services.index(r['root_service']) for r in training], minlength=len(services)).astype(float)
        for fraction in RATES:
            for seed in SEEDS:
                with np.load(mask_path(output, dataset, fraction, seed), allow_pickle=False) as masks:
                    if set(masks.files) != {key for _, key, _ in cases}:
                        raise ValueError('mask population changed')
                    for fold, key, a0 in cases:
                        target = labels[key]; k = masks[key]; a = a0 * k
                        expected, rates = make_service_mask(dataset, key, a0, fraction, seed)
                        if not np.array_equal(k, expected):
                            raise ValueError('frozen mask not reproducible')
                        stratum = ('root_originally_unobserved' if not a0[target].any() else
                            'root_newly_missing' if not a[target].any() else 'root_still_observed')
                        for method, values in rule_scores(a0, k, priors[fold]).items():
                            stable, fair = top1(values, target)
                            rows.append(dict(dataset=dataset, fold=fold, seed=seed, incident_id=key,
                                scenario=VERSION, scenario_param=f'{fraction:.2f}', stratum=stratum,
                                method=method, top1_stable=stable, top1_uniform_ties=fair,
                                uniform_random_top1=1/len(services),
                                root_deletion_probability=(rates['realized_service_fraction'] if a0[target].any() else 0.),
                                **rates))
    write_csv(output / 'availability_rule_predictions.csv', rows)
    grouped = defaultdict(list)
    for row in rows:
        for stratum in ('all', row['stratum']):
            grouped[(row['dataset'], row['scenario_param'], stratum, row['method'])].append(row)
    summaries = []
    for (dataset, fraction, stratum, method), group in sorted(grouped.items()):
        summaries.append(dict(dataset=dataset, scenario_param=fraction, stratum=stratum,
            method=method, events=len({r['incident_id'] for r in group}), event_seed_rows=len(group),
            top1_stable=float(np.mean([r['top1_stable'] for r in group])),
            top1_uniform_ties=float(np.mean([r['top1_uniform_ties'] for r in group])),
            uniform_random_top1=group[0]['uniform_random_top1'],
            actual_cell_deletion_rate=sum(r['deleted'] for r in group)/max(1,sum(r['original_available'] for r in group)),
            mean_actual_service_deletion_rate=float(np.mean([r['realized_service_fraction'] for r in group])),
            root_newly_missing_fraction=float(np.mean([r['stratum']=='root_newly_missing' for r in group])),
            mean_root_deletion_probability=float(np.mean([r['root_deletion_probability'] for r in group])),
            insufficient_eligible_event_seeds=sum(r['insufficient_eligible_services'] for r in group)))
    write_csv(output / 'availability_rule_summary.csv', summaries)
    immutable_json(output / 'audit_receipt.json', dict(status='diagnostic_only_not_ARISE_results',
        manifest_sha256=sha256(manifest_path), label_inputs=label_inputs,
        audit_source_sha256=sha256(Path(__file__)), prediction_rows=len(rows), summary_rows=len(summaries),
        primary_stratum='all', conditional_strata='diagnostic only; root-conditioned subsets retain membership cues',
        tie_rules='report stable catalogue ordering and uniform random tie expectation; no seed selection',
        uncertainty='descriptive only; five masks per event are not independent incidents'))
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('generate', 'audit'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    start = time.process_time()
    result = generate(args.output) if args.mode == 'generate' else audit(args.output)
    print(json.dumps(dict(mode=args.mode, result=result, cpu_seconds=time.process_time()-start)))


if __name__ == '__main__':
    main()
