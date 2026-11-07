"""Pair-disjoint TT cohorts, reusing the original train-only T5 preprocessing."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from .common import ROOT, dataset_dir, json_dump
from . import generalization_data as gd

TASK='TPAIR'
SPLIT=ROOT/'splits/tt_pairholdout_kfold5.json'
FOLDER=ROOT/'results_v2/TT_pair_holdout'


def split_file(dataset, task):
    if (dataset,task)!=('tt',TASK):raise ValueError('pair adapter scope')
    return SPLIT


def training_file(dataset, task, fold):
    split_file(dataset,task)
    return ROOT/f'splits/tt_pairholdout_training_labels/fold_{fold}.json'


def make_cohorts(rows):
    roots=sorted({r['root_service'] for r in rows});faults=sorted({r['fault_type'] for r in rows})
    if len(rows)!=150 or len(roots)!=10 or len(faults)!=5:raise ValueError('unexpected TT cohort')
    pair=lambda r:(r['root_service'],r['fault_type'])
    if set(collections.Counter(map(pair,rows)).values())!={3}:raise ValueError('expected three repetitions per pair')
    rng=np.random.default_rng(2027);roots=list(rng.permutation(roots));faults=list(rng.permutation(faults))
    # Opaque pair group IDs; model feature construction never sees root/type keys.
    pair_ids={p:f'pair_{i:02d}' for i,p in enumerate(sorted(set(map(pair,rows))))}
    assignment={p:(roots.index(p[0])+faults.index(p[1]))%5 for p in pair_ids}
    parts={}
    for fold in range(5):
        train=[{**r,'overlap_group':pair_ids[pair(r)]} for r in rows if assignment[pair(r)]!=fold]
        test=[r for r in rows if assignment[pair(r)]==fold]
        trkeys={r['incident_id'] for r in train};oldtr=[r for r in rows if r['incident_id'] in trkeys]
        if {r['overlap_group'] for r in oldtr}&{r['overlap_group'] for r in test}:raise ValueError('original overlap group crosses pair fold')
        if any(a['start']<b['end'] and b['start']<a['end'] for a in oldtr for b in test):raise ValueError('observation windows cross pair fold')
        inner=gd.inner_partition(train,.1,string_groups=True)
        parts[f'fold_{fold}']=dict(train=[r['incident_id'] for r in train],test=[r['incident_id'] for r in test],
            purged=[],split=f'pairholdout_{fold}',context_inner=inner,baseline_inner=inner,
            train_pair_groups={r['incident_id']:r['overlap_group'] for r in train})
    return parts


def freeze():
    rows=json.loads((dataset_dir('tt')/'incidents.json').read_text());lookup={r['incident_id']:r for r in rows}
    parts=make_cohorts(rows);gd.immutable_json(SPLIT,parts)
    for name,part in parts.items():
        labels=[dict(incident_id=k,root_service=lookup[k]['root_service'],fault_type=lookup[k]['fault_type'],overlap_group=part['train_pair_groups'][k]) for k in part['train']]
        gd.immutable_json(training_file('tt',TASK,int(name.split('_')[1])),labels)
    FOLDER.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='frozen_pair_cohorts',folds=5,seed=2027,events=150,groups=50,
        per_fold={name:dict(train=len(p['train']),test=len(p['test']),inner_fit=len(p['context_inner']['fit']),inner_validation=len(p['context_inner']['validation'])) for name,p in parts.items()},
        interpretation='unseen service/fault combinations among seen roots and fault types; posthoc sensitivity check')
    gd.immutable_json(FOLDER/'split_receipt.json',receipt)
    return receipt


def prepared_folder(dataset,task,fold,smoke=False):
    split_file(dataset,task)
    state=json.loads((FOLDER/'queue_state.json').read_text())
    row=state['jobs'][f'pair_prepare_f{fold}']
    if row['status']!='completed':raise ValueError('pair preprocessing incomplete')
    return ROOT/row['output']


def activate():
    gd.split_file=split_file;gd.training_file=training_file;gd.prepared_folder=prepared_folder


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--fold',type=int);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.freeze:print(json.dumps(freeze()))
    else:
        activate();print(gd.prepare('tt',TASK,a.fold,a.output,smoke=False))
