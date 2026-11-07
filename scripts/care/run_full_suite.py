"""Resumable, frozen multi-seed evaluation of the RPC evidence extension.

All results, including failures and regressions, are reported. A completion
marker is written only after outputs have been verified. Uses the existing GPU
lock and never treats a failed prerequisite as ready.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np
from evidence_reranker import fit, predict, model_for, fuse, paired_statistics
from full_evidence import prepare
from train_base import ranking_metrics

ROOT=Path(__file__).resolve().parents[2]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def measure(base,final,labels,rows):
    by_type={}
    for fault in sorted({r['fault_type'] for r in rows}):
        ids=np.array([i for i,r in enumerate(rows) if r['fault_type']==fault])
        by_type[fault]={'count':len(ids),'base':ranking_metrics(base[ids],labels[ids]),'corrected':ranking_metrics(final[ids],labels[ids])}
    return {'base':ranking_metrics(base,labels),'corrected':ranking_metrics(final,labels),
            'paired_statistics':paired_statistics(base,final,labels),'by_fault_type':by_type}


def run_evidence(dataset,variant,seed,cfg,base,out,evidence_dir,protocol):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        report=json.loads((out/'result.json').read_text())
        if report['protocol_sha256']!=protocol:raise ValueError('Mismatched completed correction')
        return report
    data=Path(cfg['datasets'][dataset]['data']);rows=json.loads((data/'incidents.json').read_text());meta=json.loads((data/'metadata.json').read_text())
    labels=np.load(data/'features.npz')['labels'];tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);te=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    key=variant if variant in ['no_metrics','no_logs','no_traces'] else 'full'
    x=np.load(evidence_dir/(key+'.npz'))['x'];names=json.loads((evidence_dir/(key+'-names.json')).read_text())
    probability=np.load(base/'predictions.npz')['probabilities'];b=probability[te]
    start=time.perf_counter();model=model_for(cfg['evidence']['depth']);model.set_params(random_state=seed,n_estimators=cfg['evidence']['trees'],min_samples_leaf=cfg['evidence']['min_samples_leaf'])
    fit(model,x[tr],labels[tr]);fit_seconds=time.perf_counter()-start
    with (out/'evidence_model.pkl').open('wb') as f:pickle.dump(model,f)
    start=time.perf_counter();ev=predict(model,x[te]);prediction_seconds=time.perf_counter()-start
    final=fuse(b,ev,cfg['evidence']['weight'],cfg['evidence']['top_k']);test_rows=[rows[i] for i in te]
    report={'dataset':dataset,'variant':variant,'seed':seed,'protocol_sha256':protocol,'base_directory':str(base),
            'evidence_fit_seconds':fit_seconds,'evidence_predict_seconds_per_event':prediction_seconds/len(te),
            'timing_scope':'CPU evidence-model inference only; excludes telemetry extraction, base inference and I/O.',
            'counts':{'train':len(tr),'test':len(te)},**measure(b,final,labels[te],test_rows)}
    np.savez_compressed(out/'predictions.npz',indices=te,base=b,evidence=ev,final=final)
    if variant=='full':
        sensitivity=[]
        for k in cfg['sensitivity']['top_k']:
            for w in cfg['sensitivity']['weights']:
                score=fuse(b,ev,w,min(k,len(meta['services'])))
                sensitivity.append({'top_k':k,'weight':w,'metrics':ranking_metrics(score,labels[te])})
        atomic(out/'sensitivity.json',{'protocol':'Fixed descriptive grid; do not select a replacement model from test scores.','grid':sensitivity})
        variants={};rpc=np.array([n.startswith('rpc:') for n in names]);coverage=np.array([n.startswith('rpc:') and n.endswith(':observed') for n in names]);gap=np.array([n.startswith('rpc:') and ('gap_' in n or ':hot_' in n) for n in names])
        masks={'no_rpc':~rpc,'coverage_only':~rpc|coverage,'rpc_without_gap':~gap,'rpc_without_coverage':~coverage,'rpc_gap_only':~rpc|gap}
        for name in cfg['evidence_ablations']:
            if name=='evidence_only':ae=ev;af=ev
            else:
                xx=x[:,:,masks[name]];am=model_for(cfg['evidence']['depth']);am.set_params(random_state=seed,n_estimators=cfg['evidence']['trees'],min_samples_leaf=cfg['evidence']['min_samples_leaf'])
                fit(am,xx[tr],labels[tr]);ae=predict(am,xx[te]);af=fuse(b,ae,cfg['evidence']['weight'],cfg['evidence']['top_k'])
            variants[name]=measure(b,af,labels[te],test_rows)
            np.savez_compressed(out/('ablation-'+name+'.npz'),indices=te,final=af)
        atomic(out/'evidence_ablations.json',variants)
        cases=[]
        for j,i in enumerate(te):
            rank=np.argsort(-final[j],kind='stable')
            cases.append({'incident_id':rows[i]['incident_id'],'root_service':rows[i]['root_service'],'fault_type':rows[i]['fault_type'],
                          'base_top1':meta['services'][b[j].argmax()],'corrected_top1':meta['services'][rank[0]],
                          'ranking':[meta['services'][n] for n in rank],'probabilities':final[j].tolist()})
        atomic(out/'cases.json',cases)
    atomic(out/'result.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/care/full-v3.json'));args=p.parse_args()
    os.chdir(ROOT);cfg=json.loads(args.config.read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_names=['train_base.py','trace_residual.py','prepare_aiops_rpc.py','evidence_reranker.py','full_evidence.py','run_full_suite.py']
    protocol={'config':cfg,'source_sha256':{n:digest(ROOT/'scripts/care'/n) for n in source_names},
              'inputs':{d:{f:digest(Path(c['data'])/f) for f in ['features.npz','incidents.json','metadata.json']} for d,c in cfg['datasets'].items()}}
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol changed; use a new run directory')
    atomic(path,protocol);protocol_hash=digest(path)
    state={'pid':os.getpid(),'state':'preparing','started_at':time.time(),'completed':[],'total':len(cfg['datasets'])*len(cfg['base_variants'])*len(cfg['seeds'])}
    def save():state['updated_at']=time.time();atomic(out/'status.json',state)
    save();gpu_lock=(ROOT/'artifacts/care/finetuning-gpu.lock').open('w')
    try:
        for dataset,dc in cfg['datasets'].items():
            state.update(dataset=dataset,state='preparing_evidence');save()
            if dataset=='aiops2022' and not (Path(dc['rpc'])/'coverage.json').exists():raise RuntimeError('AIOps RPC extraction not complete')
            start=time.perf_counter();evdir=prepare(dataset,dc,out)
            atomic(out/dataset/'evidence_preparation.json',{'seconds':time.perf_counter()-start,'cache':str(evdir),'rpc':dc['rpc'],'rpc_sha256':digest(Path(dc['rpc'])/'features.npz')})
            for variant in cfg['base_variants']:
                for seed in cfg['seeds']:
                    ident=f'{dataset}/{variant}/seed-{seed}';state.update(job=ident,state='training_base');save();print('START',ident,flush=True)
                    job=out/dataset/variant/f'seed-{seed}';base=job/'base';job.mkdir(parents=True,exist_ok=True)
                    existing=dc['reuse_base'].get(str(seed)) if variant=='full' else None
                    if existing:base=Path(existing)
                    if (base/'evaluation.json').exists():
                        check=json.loads((base/'evaluation.json').read_text())
                        if check['features_sha256']!=protocol['inputs'][dataset]['features.npz'] or check['incidents_sha256']!=protocol['inputs'][dataset]['incidents.json']:
                            raise ValueError('Base model trained on different inputs')
                        expected={**cfg['base'],'seed':seed,'device':dc['device'],'checkpoint_selection':dc['checkpoint_selection'],'metric_patch_size':dc['metric_patch_size']}
                        if any(check['arguments'].get(k)!=v for k,v in expected.items()):raise ValueError('Reused base settings differ')
                        if check['arguments'].get('ablation','full')!=variant:raise ValueError('Base ablation differs')
                    else:
                        cmd=[sys.executable,str(ROOT/'scripts/care/train_base.py'),'--data',dc['data'],'--output',str(base),'--seed',str(seed),'--ablation',variant,
                             '--metric-patch-size',str(dc['metric_patch_size']),'--checkpoint-selection',dc['checkpoint_selection'],'--device',dc['device']]
                        for k,v in cfg['base'].items():cmd.extend(['--'+k.replace('_','-'),str(v)])
                        if dc['device']=='cuda':fcntl.flock(gpu_lock,fcntl.LOCK_EX)
                        try:
                            with (job/'train.log').open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                        finally:
                            if dc['device']=='cuda':fcntl.flock(gpu_lock,fcntl.LOCK_UN)
                    state['state']='fitting_and_evaluating_evidence';save()
                    run_evidence(dataset,variant,seed,cfg,base,job/'correction',evdir,protocol_hash)
                    state['completed'].append(ident);save();print('DONE',ident,flush=True)
        state.update(state='complete',finished_at=time.time());save()
    except BaseException as exc:
        state.update(state='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
