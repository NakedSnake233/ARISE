"""Freeze a paper-sized incident cohort without consulting model predictions."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np


def overlap_groups(rows):
    groups = []
    for cloud in sorted({r['cloud'] for r in rows}):
        group, end = [], float('-inf')
        for row in sorted((r for r in rows if r['cloud'] == cloud),
                          key=lambda r: (r['start'], r['incident_id'])):
            if row['start'] > end and group:
                groups.append(group); group = []
            group.append(row); end = max(end, row['end'])
        if group:
            groups.append(group)
    return groups


def exact_groups(groups, count):
    # A deterministic subset-sum preserves entire overlap components, including
    # at the cohort boundary. It never looks at labels or measured performance.
    reachable = {0: []}
    for i, group in enumerate(groups):
        for size, indices in list(reachable.items()):
            total = size + len(group)
            if total <= count and total not in reachable:
                reachable[total] = indices + [i]
        if count in reachable:
            return [groups[i] for i in reachable[count]]
    raise ValueError(f'Cannot select exactly {count} incidents without splitting overlapping windows')


def freeze(rows, seed=42, count=128):
    if len({r['incident_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate incident identifiers')
    rng = np.random.default_rng(seed)
    groups = overlap_groups(rows)
    rng.shuffle(groups)
    selected = exact_groups(groups, count)
    rng.shuffle(selected)
    test_groups = exact_groups(selected, round(count*.2))
    test_ids = {r['incident_id'] for g in test_groups for r in g}
    entries = []
    for gid, group in enumerate(selected):
        for row in group:
            entries.append({k: row[k] for k in ['incident_id', 'cloud', 'start', 'end']})
            entries[-1].update(split='test' if row['incident_id'] in test_ids else 'train',
                               overlap_group=gid)
    return sorted(entries, key=lambda r: r['incident_id'])


def apply_manifest(rows, manifest):
    lookup = {r['incident_id']: r for r in rows}
    entries = manifest['incidents']
    if len({r['incident_id'] for r in entries}) != len(entries):
        raise ValueError('Duplicate IDs in frozen split')
    selected = []
    for entry in entries:
        row = lookup.get(entry['incident_id'])
        if row is None or any(row[k] != entry[k] for k in ['cloud', 'start', 'end']):
            raise ValueError('Frozen incident missing or observation window changed: '+entry['incident_id'])
        if entry['split'] not in ('train', 'test'):
            raise ValueError('Paper split must contain train/test only')
        selected.append({**row, 'split': entry['split'], 'overlap_group': entry['overlap_group']})
    if dict(collections.Counter(r['split'] for r in selected)) != manifest['counts']:
        raise ValueError('Frozen counts do not match manifest')
    for group in overlap_groups(selected):
        if len({r['split'] for r in group}) != 1:
            raise ValueError('Overlapping telemetry crosses the train/test boundary')
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('data/care/aiops2022-prepared/incidents.json'))
    p.add_argument('--output', type=Path, default=Path('configs/care/aiops2022-paper-split.json'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    rows = json.loads(args.source.read_text())
    entries = freeze(rows, args.seed)
    manifest = {'protocol': 'paper-sized reconstruction; original 128 IDs unavailable',
                'seed': args.seed, 'eligible_count': len(rows),
                'selection': 'Seeded random overlap components; exact size; no model scores or root labels used',
                'source_manifest_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                'counts': dict(collections.Counter(r['split'] for r in entries)), 'incidents': entries}
    serialized = json.dumps(manifest, indent=2, ensure_ascii=False)+'\n'
    if args.output.exists() and args.output.read_text() != serialized:
        raise ValueError('Refusing to replace an existing frozen split')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'incidents'}, indent=2))


if __name__ == '__main__':
    main()
