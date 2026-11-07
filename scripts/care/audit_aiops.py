"""Inventory the supplied AIOps derivative without silently treating it as raw data."""
import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('/workspace/iwqos2026/data/2022_CCF_AIOps_challenge'))
    p.add_argument('--output', type=Path, default=Path('artifacts/care/aiops2022'))
    args = p.parse_args()
    path = args.source / 'dataset/time_interval_and_label/time_interval_window_size_7.pkl'
    # This is the user's trusted local artifact, not an arbitrary downloaded pickle.
    labels = pickle.loads(path.read_bytes())
    incidents = []
    for split, windows in labels['time_interval'].items():
        for i, (date, cloudbed, start, end) in enumerate(windows):
            entity = labels['cmdb_id'][split][i]
            incidents.append({
                'incident_id': f'aiops2022:{date}:{cloudbed}:{start}:{entity}',
                'source_partition': split, 'date': date, 'cloudbed': cloudbed,
                'observation_start': int(start), 'observation_end': int(end),
                'fault_start': None, 'fault_end': None,
                'root_entity': entity, 'root_service': labels['template'][split][i],
                'entity_type': labels['entity_type'][split][i],
                'fault_type': labels['root_cause_type'][split][i],
            })
    files = sorted(args.source.rglob('*.csv'))
    report = {
        'source': str(args.source.resolve()), 'label_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'incident_count': len(incidents),
        'by_partition': dict(collections.Counter(x['source_partition'] for x in incidents)),
        'by_entity_type': dict(collections.Counter(x['entity_type'] for x in incidents)),
        'csv_files': len(files), 'csv_bytes': sum(f.stat().st_size for f in files),
        'raw_parent_child_spans_available': any('trace_jaeger-span' in f.name for f in files),
        'raw_log_messages_available': any(f.name in ('log_filebeat.csv', 'log.csv', 'logs.csv') for f in files),
        'limitations': [
            'Supplied raw_data contains aggregated trace/log feature tables; not original spans/log messages.',
            'Observation-window boundaries are not verified injection start/end timestamps.',
            'Existing global log-template fitting and scaler provenance are unknown; do not reuse for strict reproduction.',
            '541 annotations do not identify the 128-incident subset claimed in the paper.',
            'Node-level faults need a separately specified mapping; do not relabel them as a service.',
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'local_source_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    (args.output / 'local_incidents.json').write_text(json.dumps(incidents, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
