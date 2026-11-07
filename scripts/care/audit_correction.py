"""Summarize existing frozen correction results; never tune rules on test cases."""
import collections
import json
from pathlib import Path
import re
import numpy as np


def audit(dataset):
    data=Path('data/care')/(dataset+'-prepared');out=Path('artifacts/care')/('correction-'+dataset)
    rows=json.loads((data/'incidents.json').read_text());meta=json.loads((data/'metadata.json').read_text())
    f=np.load(data/'features.npz');p=np.load(Path('artifacts/care')/('base-'+dataset)/'predictions.npz')['probabilities']
    counters=collections.Counter();reasons=collections.Counter()
    for i,row in enumerate(rows):
        if row['split']!='test':continue
        case=json.loads((out/f'case-{i:05d}.json').read_text());obs=case['input'];truth=meta['services'][f['labels'][i]];base=meta['services'][p[i].argmax()]
        counters['incidents']+=1;counters['base_correct']+=base==truth
        counters['root_in_candidates']+=truth in obs['candidates']
        counters['high_base_confidence']+=p[i].max()>=.8
        counters['high_base_confidence_correct']+=(p[i].max()>=.8 and base==truth)
        counters['changed_top1']+=bool(case['audit'].get('changed',False));reasons[case['audit']['reason']]+=1
        try:proposal=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',case['response'].strip()))
        except (ValueError,TypeError):counters['unparseable_response']+=1;continue
        root=proposal.get('root_cause');counters['llm_root_correct']+=root==truth
        counters['llm_same_as_base']+=root==base
        counters['llm_fixes_base']+=(base!=truth and root==truth)
        counters['llm_breaks_base']+=(base==truth and root!=truth)
        counters['llm_root_without_outgoing_propagation']+=not any(f['source']==root and f['relation']=='PROPAGATES_TO' for f in obs['retrieved_facts'])
    splits={}
    for split in ['train','validation']:
        ids=[i for i,r in enumerate(rows) if r['split']==split];y=f['labels'][ids]
        splits[split]={'incidents':len(ids),'base_correct':int((p[ids].argmax(1)==y).sum()),
                       'top5_contains_root':int((np.argsort(-p[ids],axis=1)[:,:5]==y[:,None]).any(1).sum())}
    return {'counts':{k:int(v) for k,v in counters.items()},'gate_reasons':dict(reasons),'split_diagnostics':splits,
            'frozen_evaluation':json.loads((out/'evaluation.json').read_text())}


if __name__=='__main__':
    report={name:audit(name) for name in ['trainticket','aiops2022']}
    path=Path('artifacts/care/correction-diagnosis.json');path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
