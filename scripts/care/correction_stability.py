"""Repeat a frozen correction configuration; never select a seed on test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evidence_reranker import fit, predict, model_for, fuse, paired_statistics
from train_base import ranking_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('artifacts/care/correction-v3-trainticket'))
    args=p.parse_args();sel=json.loads((args.run/'selection.json').read_text())
    data=Path('data/care')/(sel['dataset']+'-prepared');rows=json.loads((data/'incidents.json').read_text())
    features=np.load(data/'features.npz');labels=features['labels'];x=np.load(args.run/'evidence_features.npz')['x']
    if sel.get('extra_features'):
        source=Path(sel['extra_features'])/'features.npz'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=sel['extra_features_sha256']:raise ValueError('Evidence changed')
        x=np.concatenate([x,np.load(source)['x']],axis=-1)
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);te=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    base=np.load(Path('artifacts/care')/('base-'+sel['dataset'])/'predictions.npz')['probabilities'][te]
    runs=[]
    for seed in [42,43,44,45,46]:
        model=model_for(sel['selected_depth']);model.set_params(random_state=seed)
        fit(model,x[tr],labels[tr]);evidence=predict(model,x[te]);final=fuse(base,evidence,sel['selected_weight'])
        runs.append({'seed':seed,'metrics':ranking_metrics(final,labels[te]),'paired_statistics':paired_statistics(base,final,labels[te])})
        np.savez_compressed(args.run/f'stability-seed{seed}.npz',indices=te,final=final)
    keys=list(runs[0]['metrics'])
    summary={k:{'mean':float(np.mean([r['metrics'][k] for r in runs])),
                'std':float(np.std([r['metrics'][k] for r in runs],ddof=1))} for k in keys}
    report={'runs':runs,'summary':summary,'configuration':{'depth':sel['selected_depth'],'weight':sel['selected_weight']},
            'scope':'Evidence-tree seed variability only; base model, data split, representation, fusion weight all fixed.',
            'selection_sha256':hashlib.sha256((args.run/'selection.json').read_bytes()).hexdigest()}
    (args.run/'stability.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
