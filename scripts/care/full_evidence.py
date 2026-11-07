"""Build modality-consistent evidence caches for the frozen complete suite."""
import json
from pathlib import Path
import numpy as np
from evidence_reranker import extract, raw_tt_metrics


def prepare(dataset,config,output):
    data=Path(config['data']);rows=json.loads((data/'incidents.json').read_text());meta=json.loads((data/'metadata.json').read_text())
    out=Path(output)/dataset/'evidence';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return out
    f=np.load(data/'features.npz');raw=None
    if dataset=='trainticket':raw=raw_tt_metrics(rows,meta['services'],meta['metric_names'],meta['step'],f['metrics'].shape[2])
    rpc=np.load(Path(config['rpc'])/'features.npz')['x'];rpc_names=json.loads((Path(config['rpc'])/'feature_names.json').read_text())
    manifest={}
    for variant in ['full','no_metrics','no_traces','no_logs']:
        features={k:f[k] for k in ['metrics','traces','logs']};rm=raw
        if variant!='full':
            modality=variant[3:];features[modality]=np.zeros_like(features[modality])
            if modality=='metrics' and raw is not None:rm=np.zeros_like(raw)
        x,names=extract(features,meta,rows,rm)
        if variant=='no_metrics':keep=[i for i,n in enumerate(names) if not n.startswith(('metric:','raw_metric:'))]
        elif variant=='no_traces':keep=[i for i,n in enumerate(names) if not n.startswith('trace:')]
        elif variant=='no_logs':keep=[i for i,n in enumerate(names) if not n.startswith('log:')]
        else:keep=list(range(len(names)))
        x=x[:,:,keep];names=[names[i] for i in keep]
        if variant!='no_traces':x=np.concatenate([x,rpc],axis=-1);names+=rpc_names
        if x.shape[:2]!=(len(rows),len(meta['services'])):raise ValueError('Evidence ordering/shape mismatch')
        np.savez_compressed(out/(variant+'.npz'),x=x)
        (out/(variant+'-names.json')).write_text(json.dumps(names,indent=2)+'\n')
        manifest[variant]={'shape':list(x.shape),'names':names}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
