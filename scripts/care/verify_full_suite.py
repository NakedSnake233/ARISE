"""Deterministic checks of frozen inputs, actual ranks and training membership."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_base import ranking_metrics


def main():
    out=Path('artifacts/care/full-v3');protocol=json.loads((out/'protocol.json').read_text());cfg=protocol['config']
    state=json.loads((out/'status.json').read_text())
    if state['state']!='complete':raise ValueError('Matrix not complete')
    for name,expected in protocol['source_sha256'].items():
        assert hashlib.sha256((Path('scripts/care')/name).read_bytes()).hexdigest()==expected,name
    checked=[]
    for dataset,dc in cfg['datasets'].items():
        data=Path(dc['data']);rows=json.loads((data/'incidents.json').read_text());labels=np.load(data/'features.npz')['labels']
        train_ids={r['incident_id'] for r in rows if r['split']=='train'}
        group_splits={}
        for r in rows:group_splits.setdefault(r['overlap_group'],set()).add(r['split'])
        assert all(len(s)==1 for s in group_splits.values())
        for f,expected in protocol['inputs'][dataset].items():assert hashlib.sha256((data/f).read_bytes()).hexdigest()==expected
        for variant in cfg['base_variants']:
            for seed in cfg['seeds']:
                folder=out/dataset/variant/f'seed-{seed}/correction';r=json.loads((folder/'result.json').read_text())
                base=Path(r['base_directory']);checkpoint=torch.load(base/'model.pt',map_location='cpu',weights_only=False)
                assert set(checkpoint['training_ids'])==train_ids
                pred=np.load(folder/'predictions.npz');indices=pred['indices']
                assert list(indices)==[i for i,row in enumerate(rows) if row['split']=='test']
                assert np.isfinite(pred['base']).all() and np.isfinite(pred['final']).all()
                assert np.allclose(pred['base'].sum(1),pred['final'].sum(1))
                for key,array in [('base',pred['base']),('corrected',pred['final'])]:
                    scores=ranking_metrics(array,labels[indices]);assert scores==r[key],(dataset,variant,seed,key)
                b=pred['base'].argmax(1)==labels[indices];c=pred['final'].argmax(1)==labels[indices]
                assert int((~b&c).sum())==r['paired_statistics']['fixed']
                assert int((b&~c).sum())==r['paired_statistics']['broken']
                if variant=='full':
                    ab=json.loads((folder/'evidence_ablations.json').read_text())
                    for name in cfg['evidence_ablations']:
                        ap=np.load(folder/('ablation-'+name+'.npz'))
                        assert ranking_metrics(ap['final'],labels[indices])==ab[name]['corrected']
                checked.append(f'{dataset}/{variant}/seed-{seed}')
        # No excluded modality may enter the cached correction feature names.
        evdir=out/dataset/'evidence'
        for variant,prefix in [('no_metrics',('metric:','raw_metric:')),('no_traces',('trace:','rpc:')),('no_logs',('log:',))]:
            names=json.loads((evdir/(variant+'-names.json')).read_text())
            assert not any(n.startswith(prefix) for n in names)
    assert len(checked)==state['total']==len(state['completed'])
    result={'status':'pass','checked_configurations':len(checked),'checks':['Frozen source/input hashes','Incident-group split isolation','Exact checkpoint training membership','Raw prediction rank recomputation','Paired fixed/broken counts','Evidence ablation rank recomputation','Excluded modality feature names'],'configurations':checked}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
