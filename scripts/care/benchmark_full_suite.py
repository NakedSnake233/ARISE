"""Single-event warm inference timing; keep input preparation scope explicit."""
import fcntl
import json
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from train_base import CAREBase
from evidence_reranker import predict, fuse
from trace_residual import extract_one


def summary(values):
    return {'count':len(values),'mean_seconds':float(np.mean(values)),'p50_seconds':float(np.quantile(values,.5)),
            'p95_seconds':float(np.quantile(values,.95))}


def main():
    torch.set_num_threads(4);out=Path('artifacts/care/full-v3');cfg=json.loads((out/'protocol.json').read_text())['config'];results={}
    lock=Path('artifacts/care/finetuning-gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    for dataset,dc in cfg['datasets'].items():
        folder=out/dataset/'full'/'seed-42'/'correction';r=json.loads((folder/'result.json').read_text())
        device=torch.device(dc['device']);data=Path(dc['data']);rows=json.loads((data/'incidents.json').read_text());meta=json.loads((data/'metadata.json').read_text())
        f=np.load(data/'features.npz');ids=[i for i,r in enumerate(rows) if r['split']=='test']
        tensors=[torch.from_numpy(f[k]).float().to(device) for k in ['metrics','traces','logs']]
        state=torch.load(Path(r['base_directory'])/'model.pt',map_location=device,weights_only=False)
        model=CAREBase(**state['model_args']).to(device);model.load_state_dict(state['state_dict']);model.eval();adj=torch.tensor(state['adjacency'],device=device)
        x=np.load(out/dataset/'evidence/full.npz')['x']
        with (folder/'evidence_model.pkl').open('rb') as stream:ranker=pickle.load(stream)
        def sync():
            if device.type=='cuda':torch.cuda.synchronize()
        def call(i):
            sync();start=time.perf_counter()
            with torch.inference_mode():prob=model(*(t[i:i+1] for t in tensors),adj)[0].softmax(-1).cpu().numpy()
            sync();middle=time.perf_counter();ev=predict(ranker,x[i:i+1]);fuse(prob,ev,.25,5)
            end=time.perf_counter();return middle-start,end-middle,end-start
        for _ in range(5):call(ids[0])
        timings=np.array([call(i) for _ in range(3) for i in ids])
        result={'device':str(device),'seed':42,'batch_size':1,'base':summary(timings[:,0]),'evidence_and_fusion':summary(timings[:,1]),
                'total_cached_feature_inference':summary(timings[:,2]),
                'scope':'Warm models and precomputed numerical telemetry/RPC features; includes base forward, device transfer of probabilities, CPU evidence scoring and fusion; excludes raw telemetry preparation, model loading and I/O. Three repetitions per test event; filesystem cache state unspecified.'}
        if dataset=='trainticket':
            times=[]
            for i in ids[:3]:
                start=time.perf_counter();extract_one(rows[i],meta['services'],meta['before']);times.append(time.perf_counter()-start)
            result['rpc_raw_file_to_features_sample']=summary(times)
            result['rpc_timing_scope']='First three test indices in manifest order, one pass each; reads gzip spans and extracts RPC features only, excludes metrics/log preprocessing. Not total end-to-end latency.'
        results[dataset]=result
        del model,tensors
        if device.type=='cuda':torch.cuda.empty_cache()
    (out/'timing.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))


if __name__=='__main__':main()
