"""Explicit zero-inclusive counts supplement; does not change frozen scores."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from .common import sha256
from .scenario_v2 import OUTPUT, immutable_json, write_csv

STRATA = ('root_newly_missing', 'root_still_observed', 'root_originally_unobserved')


def counts(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['method'] == 'fully_missing_first':
            if row['stratum'] not in STRATA:
                raise ValueError('unexpected stratum')
            groups[(row['dataset'],row['scenario_param'])].append(row)
    result = []
    for (dataset, fraction), group in sorted(groups.items()):
        if len({(r['incident_id'],r['seed']) for r in group}) != len(group):
            raise ValueError('duplicate event/seed')
        for stratum in ('all', *STRATA):
            part = group if stratum == 'all' else [r for r in group if r['stratum'] == stratum]
            result.append(dict(dataset=dataset, scenario_param=fraction, stratum=stratum,
                events=len({r['incident_id'] for r in part}), event_seed_rows=len(part),
                fraction_of_event_seed_rows=len(part)/len(group)))
    return result


def supplement(output=OUTPUT):
    source = output/'availability_rule_predictions.csv'
    with source.open(newline='') as stream:
        rows = counts(csv.DictReader(stream))
    write_csv(output/'stratum_counts.csv',rows)
    immutable_json(output/'stratum_counts_receipt.json',dict(
        source_sha256=sha256(source), source_code_sha256=sha256(Path(__file__)),
        rows=len(rows), zero_sample_performance='undefined; omitted from rule_summary',
        conditional_estimand='equal qualifying event-seed weight, not equal distinct incident weight'))
    return len(rows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    print(json.dumps(dict(stratum_count_rows=supplement(args.output))))
