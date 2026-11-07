"""Fixed-setting feature ablations after primary evaluation, not model selection.

Report EVERY variant. Never replace the primary model using these test results.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import GroupKFold
from evidence_reranker import fit, predict, model_for, fuse, paired_statistics
from train_base import ranking_metrics


def main():
    run=Path('artifacts/care/correction-v3-trainticket');out=run/'ablations';out.mkdir(exist_ok=True)
    sel=json.loads((run/'selection.json').read_text());data=Path('data/care/trainticket-prepared')
    rows=json.loads((data/'incidents.json').read_text());labels=np.load(data/'features.npz')['labels']
    x=np.load(run/'evidence_features.npz')['x'];source=Path(sel['extra_features'])
    extra=np.load(source/'features.npz')['x'];names=json.loads((source/'feature_names.json').read_text())
    coverage=np.array([n.endswith(':observed') for n in names]);gap=np.array(['gap_' in n or ':hot_' in n for n in names])
    variants={'no_rpc':np.zeros(len(names),bool),'coverage_only':coverage,'rpc_without_gap':~gap,
              'rpc_without_coverage':~coverage,'rpc_gap_only':gap}
    manifest={'created_at':time.time(),'depth':sel['selected_depth'],'weight':sel['selected_weight'],
              'variants':{k:[n for n,take in zip(names,mask) if take] for k,mask in variants.items()},
              'protocol':'Controlled ablation with fixed depth, seed 42 and fusion weight .25 from primary validation; all variants reported. Post-primary exploratory mechanism check, no replacement or test-based tuning.'}
    if (out/'plan.json').exists():raise ValueError('Ablation plan already exists; do not overwrite')
    (out/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    splits={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']};tr=splits['train']
    groups=[rows[i]['overlap_group'] for i in tr]
    base=np.load('artifacts/care/base-trainticket/predictions.npz')['probabilities'];reports={}
    for name,mask in variants.items():
        xx=np.concatenate([x,extra[:,:,mask]],axis=-1);oof=np.zeros(xx[tr].shape[:2])
        for train,val in GroupKFold(5).split(tr,groups=groups):
            model=fit(model_for(sel['selected_depth']),xx[tr[train]],labels[tr[train]])
            oof[val]=predict(model,xx[tr[val]])
        model=fit(model_for(sel['selected_depth']),xx[tr],labels[tr]);report={'training_oof_evidence':ranking_metrics(oof,labels[tr])}
        for split in ['validation','test']:
            ids=splits[split];ev=predict(model,xx[ids]);final=fuse(base[ids],ev,sel['selected_weight'])
            report[split]={'evidence_only':ranking_metrics(ev,labels[ids]),'fused':ranking_metrics(final,labels[ids]),
                           'paired_statistics':paired_statistics(base[ids],final,labels[ids])}
            np.savez_compressed(out/f'{name}-{split}.npz',indices=ids,evidence=ev,final=final)
        reports[name]=report;(out/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,json.dumps(report),flush=True)
    (out/'summary.json').write_text(json.dumps({'plan_sha256':hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),'variants':reports},indent=2)+'\n')


if __name__=='__main__':main()
