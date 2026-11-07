"""S0 parity supplement for official inference, including finite smoke overlap."""
import json
import numpy as np
from .common import ROOT,sha256,json_dump


def check(dataset,fold,seed,method,ids,scores,failed):
    state=json.loads((ROOT/'results_v2/remaining_queue/queue_state.json').read_text())
    slug=method.replace('+','plus').replace('-','')
    name=f'full_official_{dataset}_{slug}_f{fold}_s{seed}'
    row=state['jobs'].get(name);scope='full_reference'
    if not row or row['status']!='completed':
        if (dataset,method)!=('re2_ss','RCD'):raise ValueError('required full official S0 reference unavailable: '+name)
        name='smoke_official_re2_ss_RCD_f0_s42'
        row=state['jobs'][name];scope='smoke_overlap_only'
        if fold!=0 or seed!=42:
            return dict(scope='no_same_fold_seed_reference',compared_events=0,max_abs_difference=None,
                        limitation='RE2-SS/RCD previously only had fold0/seed42 one-event smoke; new full-cohort calculation')
    folder=ROOT/row['output'];meta=json.loads((folder/'run_meta.json').read_text())
    if sha256(folder/'scores.npz')!=meta['scores_sha256']:raise ValueError('official reference changed')
    with np.load(folder/'scores.npz') as z:
        reference_ids=z['incident_ids'].tolist();idx=[ids.index(key) for key in reference_ids]
        if scope=='full_reference' and reference_ids!=ids:raise ValueError('full reference identity differs')
        difference=float(np.max(np.abs(scores[0,idx]-z['scores'][0])))
        if difference>1e-5 or not np.array_equal(failed[0,idx],z['failed'][0]):raise ValueError('official S0 parity failed')
    return dict(scope=scope,compared_events=len(idx),max_abs_difference=difference,reference_job=name,
                reference_scores_sha256=sha256(folder/'scores.npz'))


def supplement_completed():
    state=json.loads((ROOT/'results_v2/S3v2_models/queue_state.json').read_text());result={}
    for name,row in state['jobs'].items():
        cfg=row['spec']['expected_config']
        if cfg['component']!='official' or row['status']!='completed':continue
        folder=ROOT/row['output']
        with np.load(folder/'scores.npz') as z:
            result[name]=check(cfg['dataset'],cfg['fold'],cfg['seed'],cfg['method'],z['incident_ids'].tolist(),z['scores'],z['failed'])
    path=ROOT/'results_v2/S3v2_models/official_pilot_parity_supplement.json'
    json_dump(path,result)
    return result
