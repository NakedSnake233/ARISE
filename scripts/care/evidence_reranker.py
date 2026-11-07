"""Train-only, identity-free evidence ranker for candidate correction.

Features contain telemetry contrasts and invocation neighborhoods; never root
labels, fault types, incident IDs, injection parameters, or base predictions.
Training CV chooses the evidence model, validation chooses its fusion weight.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from train_base import ranking_metrics


def raw_tt_metrics(rows,services,metric_names,step,bins):
    array=np.full((len(rows),len(services),bins,len(metric_names)),np.nan)
    for i,row in enumerate(rows):
        file=Path(row['source']).parent/'metrics.json.gz'
        with gzip.open(file,'rt') as f:series=json.load(f)['data']['result']
        buckets=collections.defaultdict(list)
        for item in series:
            labels=item['metric'];name=labels.get('__name__');pod=labels.get('pod','')
            service=next((s for s in services if pod.startswith(s+'-')),None)
            if service is None or name not in metric_names:continue
            if labels.get('container') not in (None,'',service):continue
            v=np.asarray(item['values'],dtype=float)
            if len(v)<2:continue
            if name.endswith('_total'):
                v=np.column_stack([v[1:,0],np.maximum(0,np.diff(v[:,1]))/np.maximum(1e-6,np.diff(v[:,0]))])
            for stamp,value in v:
                b=int((stamp-row['start'])//step)
                if 0<=b<bins:buckets[(services.index(service),b,metric_names.index(name))].append(value)
        for (s,b,c),values in buckets.items():array[i,s,b,c]=np.mean(values)
    return array


def contrasts(values,before,channel_names,prefix,raw=False):
    values=np.asarray(values,dtype=float)
    # A log transform gives positive physical counters/rates a scale comparable
    # across CPU, memory and I/O without using cross-incident fitted statistics.
    if raw:values=np.log1p(np.maximum(0,values))
    values=np.nan_to_num(values)
    pre=values[:,:,:before];post=values[:,:,before:]
    median=np.median(pre,axis=2);mad=np.median(np.abs(pre-median[:,:,None]),axis=2)
    scale=np.maximum(mad*1.4826,.1 if raw else .5)
    delta=post-median[:,:,None]
    descriptors={'baseline':median,'level_mean':post.mean(2),
                 'delta_mean':delta.mean(2),'delta_peak':delta.max(2),'delta_min':delta.min(2),
                 'delta_q90':np.quantile(delta,.9,axis=2),
                 'positive_z':np.clip(np.quantile(delta,.9,axis=2)/scale,0,50),
                 'negative_z':np.clip(-np.quantile(delta,.1,axis=2)/scale,0,50),
                 'positive_persistence':(delta>2*scale[:,:,None]).mean(2),
                 'negative_persistence':(delta<-2*scale[:,:,None]).mean(2)}
    features=[];names=[]
    for descriptor,x in descriptors.items():
        for c,channel in enumerate(channel_names):
            features.append(x[:,:,c]);names.append(f'{prefix}:{channel}:{descriptor}')
    # Relative shock separates a local change from system-wide load variation.
    shock=np.maximum(descriptors['positive_z'],descriptors['negative_z'])
    for c,channel in enumerate(channel_names):
        x=shock[:,:,c];relative=x-np.median(x,axis=1,keepdims=True)
        features.extend([relative,x/(x.sum(1,keepdims=True)+1e-6)])
        names.extend([f'{prefix}:{channel}:relative_shock',f'{prefix}:{channel}:shock_share'])
    return features,names,shock.max(-1)


def extract(features,meta,rows,raw_metrics=None):
    before=int(meta['before']//meta['step']);services=meta['services']
    output,names,metric_shock=contrasts(features['metrics'],before,meta['metric_names'],'metric')
    if raw_metrics is not None:
        extra,nm,metric_shock=contrasts(raw_metrics,before,meta['metric_names'],'raw_metric',True)
        output.extend(extra);names.extend(nm)
    extra,nm,trace_shock=contrasts(features['traces'],before,meta['trace_names'],'trace')
    output.extend(extra);names.extend(nm)
    log=features['logs'];pre=log[:,:,:before].mean(2);post=log[:,:,before:].mean(2)
    pp=(pre+1e-6)/(pre.sum(-1,keepdims=True)+pre.shape[-1]*1e-6)
    pq=(post+1e-6)/(post.sum(-1,keepdims=True)+post.shape[-1]*1e-6);mid=(pp+pq)/2
    js=(pp*np.log(pp/mid)+pq*np.log(pq/mid)).sum(-1)/2
    extra={'template_js':js,'activity_delta':np.log1p(post.sum(-1))-np.log1p(pre.sum(-1)),
           'novel_pattern_mass':(post*(pre==0)).sum(-1)/(post.sum(-1)+1e-6)}
    err=np.array([any(k in t.lower() for k in ['exception','error','timeout','failed','refused']) for t in meta['log_templates']])
    extra['error_pattern_delta']=np.log1p(post[:,:,err].sum(-1))-np.log1p(pre[:,:,err].sum(-1))
    for key,x in extra.items():output.append(x);names.append('log:'+key)
    # Invocation direction is recorded as such. Both caller and callee contexts
    # are features, not assertions that faults propagate only caller -> callee.
    total=metric_shock+trace_shock
    for direction in ['callers','callees']:
        degree=np.zeros_like(total);neighbor=np.zeros_like(total)
        for i,row in enumerate(rows):
            neighbors=collections.defaultdict(set)
            for call in row['calls']:
                u,v=call['source'],call['target']
                if u not in services or v not in services:continue
                if direction=='callers':u,v=v,u
                neighbors[services.index(u)].add(services.index(v))
            for s,nb in neighbors.items():degree[i,s]=len(nb);neighbor[i,s]=max(total[i,list(nb)])
        output.extend([degree,neighbor,total-neighbor]);names.extend([direction+':degree',direction+':max_shock',direction+':local_minus_neighbor'])
    return np.nan_to_num(np.stack(output,-1)).astype('float32'),names


def model_for(depth):
    return ExtraTreesClassifier(n_estimators=256,max_depth=depth,min_samples_leaf=3,
                               class_weight='balanced',max_features=.7,random_state=42,n_jobs=4)


def fit(model,x,labels):
    y=(np.arange(x.shape[1])[None,:]==labels[:,None]).astype(int)
    model.fit(x.reshape(-1,x.shape[-1]),y.ravel());return model


def predict(model,x):
    p=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(x.shape[:2])
    return p/(p.sum(-1,keepdims=True)+1e-12)


def fuse(base,evidence,weight,top_k=5):
    if not 0<=weight<=1:raise ValueError('Fusion weight must be in [0,1]')
    if base.shape!=evidence.shape or not np.isfinite(evidence).all() or (evidence<0).any():
        raise ValueError('Invalid evidence probabilities')
    mask=np.zeros_like(base,dtype=bool)
    np.put_along_axis(mask,np.argsort(-base,axis=1,kind='stable')[:,:top_k],True,axis=1)
    ep=evidence*mask;total=ep.sum(-1,keepdims=True)
    ep=np.where(total>0,ep/np.maximum(total,1e-12),(base*mask)/np.maximum((base*mask).sum(-1,keepdims=True),1e-12))
    # Preserve all scores and ordering outside the candidate set.
    mass=(base*mask).sum(-1,keepdims=True)
    return np.where(mask,(1-weight)*base+weight*ep*mass,base)


def paired_statistics(base,corrected,labels):
    b=base.argmax(-1)==labels;c=corrected.argmax(-1)==labels
    fixed=int((~b&c).sum());broken=int((b&~c).sum());n=fixed+broken
    delta=c.astype(float)-b.astype(float)
    rng=np.random.default_rng(42)
    boots=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(fixed,broken)+1))/2**n) if n else 1.
    return {'fixed':fixed,'broken':broken,'top1_delta':float(delta.mean()),
            'paired_bootstrap_95_percentile_ci':np.quantile(boots,[.025,.975]).tolist(),
            'mcnemar_exact_two_sided_p':p,
            'note':'Incident-level paired bootstrap; small repeated test cohort, no correction for research iteration.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['trainticket','aiops2022'],default='trainticket')
    p.add_argument('--stage',choices=['select','evaluate'],required=True)
    p.add_argument('--output',type=Path,default=Path('artifacts/care/correction-v2-trainticket'))
    p.add_argument('--extra-features',type=Path,help='Directory containing features.npz and feature_names.json; telemetry only')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=Path('data/care')/(args.dataset+'-prepared');base=Path('artifacts/care')/('base-'+args.dataset)
    rows=json.loads((data/'incidents.json').read_text());meta=json.loads((data/'metadata.json').read_text())
    features=np.load(data/'features.npz');prob=np.load(base/'predictions.npz')['probabilities']
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
    cache=args.output/'evidence_features.npz'
    if not cache.exists():
        raw=raw_tt_metrics(rows,meta['services'],meta['metric_names'],meta['step'],features['metrics'].shape[2]) if args.dataset=='trainticket' else None
        x,names=extract(features,meta,rows,raw);np.savez_compressed(cache,x=x)
        (args.output/'feature_names.json').write_text(json.dumps(names,indent=2)+'\n')
    x=np.load(cache)['x'];names=json.loads((args.output/'feature_names.json').read_text())
    if args.extra_features:
        extra=np.load(args.extra_features/'features.npz')['x']
        if extra.shape[:2]!=x.shape[:2]:raise ValueError('Extra evidence shape does not match incidents/services')
        x=np.concatenate([x,extra],axis=-1)
        names+=json.loads((args.extra_features/'feature_names.json').read_text())
    if args.stage=='select':
        if (args.output/'selection.json').exists():raise RuntimeError('Selection already frozen; do not overwrite')
        tr=indices['train'];va=indices['validation'];labels=features['labels'][tr]
        groups=[rows[i]['overlap_group'] for i in tr]
        results=[];best=None
        for depth in [4,8,None]:
            oof=np.zeros(x[tr].shape[:2])
            for train,val in GroupKFold(5).split(tr,groups=groups):
                model=fit(model_for(depth),x[tr[train]],labels[train]);oof[val]=predict(model,x[tr[val]])
            score=ranking_metrics(oof,labels);results.append({'max_depth':depth,'training_oof':score})
            key=(score['PR@1'],score['MAP@5'])
            if best is None or key>best[0]:best=(key,depth,oof.copy())
        model=fit(model_for(best[1]),x[tr],labels);ev=predict(model,x[va]);vl=features['labels'][va]
        candidates=[]
        for weight in [0.,.25,.5,.75,1.]:
            score=ranking_metrics(fuse(prob[va],ev,weight),vl)
            candidates.append({'weight':weight,'validation':score})
        chosen=max(candidates,key=lambda c:(c['validation']['PR@1'],c['validation']['MAP@5'],-c['weight']))
        selection={'dataset':args.dataset,'selected_depth':best[1],'training_cv':results,'validation_grid':candidates,
                   'selected_weight':chosen['weight'],'validation_base':ranking_metrics(prob[va],vl),
                   'validation_evidence_only':ranking_metrics(ev,vl),'train_ids':[rows[i]['incident_id'] for i in tr],
                   'validation_ids':[rows[i]['incident_id'] for i in va],'selected_at':time.time(),
                   'protocol':'5-fold grouped training CV for evidence model; 12-event validation for fusion; no test scores read during selection',
                   'prospective_stability_check':'After freezing depth and fusion weight, refit evidence trees with seeds 42,43,44,45,46; report all, no seed selection. Base model remains fixed.',
                   'limitations':['Repeated use of previously inspected test set: exploratory confirmation only.',
                                  'Base model and preprocessing already fit on training partition; inner CV is conditional on those fixed representations, not a nested end-to-end estimate.'],
                   'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'extra_features':str(args.extra_features) if args.extra_features else None,
                   'extra_features_sha256':hashlib.sha256((args.extra_features/'features.npz').read_bytes()).hexdigest() if args.extra_features else None,
                   'features_sha256':hashlib.sha256((data/'features.npz').read_bytes()).hexdigest()}
        (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        with (args.output/'model.pkl').open('wb') as f:pickle.dump(model,f)
        np.savez_compressed(args.output/'validation.npz',indices=va,evidence=ev,base=prob[va],final=fuse(prob[va],ev,chosen['weight']))
        np.savez_compressed(args.output/'training_oof.npz',indices=tr,evidence=best[2])
        importance=sorted(zip(names,map(float,model.feature_importances_)),key=lambda t:-t[1])
        (args.output/'feature_importance.json').write_text(json.dumps(importance,indent=2)+'\n')
        print(json.dumps(selection,indent=2))
    else:
        selection=json.loads((args.output/'selection.json').read_text())
        if selection.get('extra_features')!=(str(args.extra_features) if args.extra_features else None):
            raise ValueError('Evaluation must use the frozen extra evidence configuration')
        if args.extra_features and selection['extra_features_sha256']!=hashlib.sha256((args.extra_features/'features.npz').read_bytes()).hexdigest():
            raise ValueError('Extra evidence changed after selection')
        if selection['features_sha256']!=hashlib.sha256((data/'features.npz').read_bytes()).hexdigest():
            raise ValueError('Dataset changed after selection')
        with (args.output/'model.pkl').open('rb') as f:model=pickle.load(f)
        te=indices['test'];labels=features['labels'][te];ev=predict(model,x[te]);final=fuse(prob[te],ev,selection['selected_weight'])
        np.savez_compressed(args.output/'test_predictions.npz',indices=te,base=prob[te],evidence=ev,final=final)
        b=prob[te].argmax(-1)==labels;c=final.argmax(-1)==labels
        report={'base':ranking_metrics(prob[te],labels),'corrected':ranking_metrics(final,labels),
                'incidents':len(te),'fixed':int((~b&c).sum()),'broken':int((b&~c).sum()),
                'selected_weight':selection['selected_weight'],'selection_sha256':hashlib.sha256((args.output/'selection.json').read_bytes()).hexdigest(),
                'limitations':selection['limitations']}
        report['paired_statistics']=paired_statistics(prob[te],final,labels)
        (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
