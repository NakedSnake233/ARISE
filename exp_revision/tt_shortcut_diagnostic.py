"""Read-only TT composition, split and frozen ET feature diagnostics; no refit."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from .common import ROOT, dataset_dir, json_dump, sha256
from .infer_evidence_masks import completed_folder


def run(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    started=time.perf_counter()
    rows=json.loads((dataset_dir('tt')/'incidents.json').read_text())
    meta=json.loads((dataset_dir('tt')/'metadata.json').read_text())
    lookup={r['incident_id']:r for r in rows};services=meta['services']
    splits=json.loads((ROOT/'splits/tt_kfold5.json').read_text())
    pair=lambda r:(r['root_service'],r['fault_type'])
    protocols=collections.Counter();injection=collections.defaultdict(set)
    for r in rows:
        raw=json.loads((ROOT/r['source']).read_text());p=raw['protocol']
        protocols[json.dumps(p,sort_keys=True)]+=1
        spec=raw['chaos_injected_status']['spec']
        injection[r['fault_type']].add(json.dumps({k:v for k,v in spec.items() if k not in ('selector','containerNames')},sort_keys=True))
    structure=dict(events=len(rows),candidate_services=len(services),root_services=len(set(r['root_service'] for r in rows)),
        fault_counts=dict(collections.Counter(r['fault_type'] for r in rows)),root_counts=dict(collections.Counter(r['root_service'] for r in rows)),
        pair_count=len(set(map(pair,rows))),pair_repetition_counts=dict(collections.Counter(collections.Counter(map(pair,rows)).values())),
        groups=len(set(r['overlap_group'] for r in rows)),protocols=[dict(protocol=json.loads(k),count=v) for k,v in protocols.items()],
        unique_injection_specs={k:[json.loads(x) for x in sorted(v)] for k,v in injection.items()},
        time_start=min(r['start'] for r in rows),time_end=max(r['end'] for r in rows))
    overlap=[];exposures=[];dup=[];importance=[];rules=[];selected=[];inputs={}
    for fold in range(5):
        part=splits[f'fold_{fold}'];train=[lookup[k] for k in part['train']];test=[lookup[k] for k in part['test']]
        train_pairs=collections.Counter(map(pair,train));train_types=set(r['fault_type'] for r in train);train_roots=set(r['root_service'] for r in train)
        overlap.append(dict(fold=fold,id_overlap=len(set(part['train'])&set(part['test'])),
            group_overlap=len(set(r['overlap_group'] for r in train)&set(r['overlap_group'] for r in test)),
            temporal_window_overlap=sum(a['start']<b['end'] and b['start']<a['end'] for a in train for b in test)))
        for r in test:exposures.append(dict(fold=fold,incident_id=r['incident_id'],same_pair_train_repeats=train_pairs[pair(r)],seen_fault=r['fault_type'] in train_types,seen_root=r['root_service'] in train_roots))
        path=ROOT/f'cache/fold_evidence_v2/tt/fold{fold}'
        names=json.loads((path/'metadata.json').read_text())['feature_names']
        with np.load(path/'features.npz') as z:x=z['x'].copy();ids=z['incident_ids'].tolist()
        assert ids==part['train']+part['test']
        tr=len(train);labels=np.array([services.index(lookup[k]['root_service']) for k in ids])
        hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in x]
        dup.append(dict(fold=fold,identical_full_evidence_test_events=sum(h in set(hashes[:tr]) for h in hashes[tr:])))
        # Select a signed single feature on training events only. Uniform ties
        # avoid getting accidental service-order credit for constant features.
        options=[]
        for sign in (1,-1):
            v=x*sign;ties=v==v.max(axis=1,keepdims=True)
            hits=ties[np.arange(len(ids)),labels,:]/ties.sum(1)
            options.append(hits)
        training=np.concatenate([a[:tr].mean(0) for a in options]);best=int(training.argmax());sign=1 if best<len(names) else -1;j=best%len(names)
        selected.append(dict(fold=fold,feature=names[j],sign=sign,training_expected_top1=float(training[best])))
        for i,k in enumerate(ids[tr:],tr):rules.append(dict(fold=fold,incident_id=k,fault_type=lookup[k]['fault_type'],rule='train_selected_single_feature',expected_hit=float(options[0 if sign==1 else 1][i,j])))
        # Fixed broad maximum-shock rules; no fault label is supplied to a rule.
        fixed={'max_raw_metric_shock':[i for i,n in enumerate(names) if n.startswith('raw_metric:') and n.endswith(('positive_z','negative_z'))],
               'max_trace_shock':[i for i,n in enumerate(names) if n.startswith('trace:') and n.endswith(('positive_z','negative_z'))],
               'incoming_rpc_gap':[i for i,n in enumerate(names) if n=='rpc:incoming:gap_delta_ms_log']}
        for rule,indices in fixed.items():
            v=x[:,:,indices].max(-1);ties=v==v.max(1,keepdims=True);hits=ties[np.arange(len(ids)),labels]/ties.sum(1)
            for i,k in enumerate(ids[tr:],tr):rules.append(dict(fold=fold,incident_id=k,fault_type=lookup[k]['fault_type'],rule=rule,expected_hit=float(hits[i])))
        for seed in range(42,47):
            p=completed_folder('results_v2/T4_et_spec_full','tt',fold,seed)/'model.joblib'
            model=joblib.load(p);assert len(model.feature_importances_)==len(names)
            for name,value in zip(names,model.feature_importances_):importance.append(dict(fold=fold,seed=seed,feature=name,family=name.split(':')[0],importance=float(value)))
            inputs[str(p.relative_to(ROOT))]=sha256(p)
    pd.DataFrame(exposures).to_csv(output/'split_pair_exposure.csv',index=False)
    pd.DataFrame(rules).to_csv(output/'simple_rule_predictions.csv',index=False)
    r=pd.DataFrame(rules).groupby(['rule','fault_type']).expected_hit.agg(['mean','count']);r.to_csv(output/'simple_rule_by_type.csv')
    im=pd.DataFrame(importance);im.to_csv(output/'feature_importance_all.csv',index=False)
    im.groupby('feature').importance.mean().sort_values(ascending=False).to_csv(output/'feature_importance_mean.csv')
    family=im.groupby(['fold','seed','family']).importance.sum().groupby('family').mean().sort_values(ascending=False)
    # Existing out-of-fold artifacts, including already completed held-type and
    # held-root experiments. These are S0 comparisons, not S3v2 held-type runs.
    full=ROOT/'results_v2/S3v2_evaluation_interim_20261002_1100/predictions.parquet'
    pf=pd.read_parquet(full);pf=pf[(pf.dataset=='tt')&(pf.scenario=='S0')].copy();pf['fault_type']=pf.incident_id.map(lambda k:lookup[k]['fault_type']);pf['protocol']='ordinary_grouped_5fold'
    frames=[pf[['protocol','method','seed','incident_id','fault_type','rank_of_root']]]
    for stage,protocol in [('T5a','held_fault_type'),('T5b','held_root_service')]:
        path=ROOT/f'results_v2/{stage}/predictions.parquet';f=pd.read_parquet(path);f=f[(f.dataset=='tt')&(f.scenario=='S0')].copy();f['protocol']=protocol
        frames.append(f[['protocol','method','seed','incident_id','fault_type','rank_of_root']]);inputs[str(path.relative_to(ROOT))]=sha256(path)
    scores=pd.concat(frames,ignore_index=True);scores['hit']=scores.rank_of_root.eq(1)
    scores.groupby(['protocol','method','fault_type']).hit.agg(['mean','count']).to_csv(output/'scores_by_type.csv')
    scores.groupby(['protocol','method','seed']).hit.mean().groupby(['protocol','method']).agg(['mean','std','count']).to_csv(output/'scores_by_protocol.csv')
    receipt=dict(status='read_only_posthoc_diagnostic_no_refit',structure=structure,overlap_checks=overlap,exact_duplicates=dup,
        test_events_same_pair_in_training=sum(r['same_pair_train_repeats']>0 for r in exposures),
        pair_training_repeat_distribution=dict(collections.Counter(r['same_pair_train_repeats'] for r in exposures)),
        selected_single_features=selected,simple_rule_overall=pd.DataFrame(rules).groupby('rule').expected_hit.mean().to_dict(),
        feature_family_importance=family.to_dict(),elapsed_seconds=time.perf_counter()-started,
        input_sha256=inputs,source_sha256=sha256(Path(__file__)),
        limitations=['Posthoc diagnosis after viewing performance; not a new confirmatory benchmark.',
        'Tree impurity importance is associational, not a causal feature ablation.',
        'No new training, feature subset refits, or label permutation experiment.',
        'Exact event-level feature equality is checked; raw cross-event trace/span overlap and near-duplicates are not exhaustively audited.',
        'Held-type/service comparisons reuse existing S0 artifacts; they are not new S3v2 generalization results.'])
    json_dump(output/'receipt.json',receipt)
    return {k:v for k,v in receipt.items() if k not in ('input_sha256','structure')}


if __name__=='__main__':
    from .disk_guard_fast import DiskGuard
    out=ROOT/'results_v2/TT_ET_spec_diagnostic_20261002';guard=DiskGuard(ROOT)
    guard.reserve('tt_et_diagnostic',out,ROOT/'logs/tt_et_diagnostic.log')
    try:print(json.dumps(run(out)))
    finally:guard.release('tt_et_diagnostic')
