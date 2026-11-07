"""Prepare incident-disjoint CARE data from original spans/logs and local metrics.

All fitted artifacts (Drain, metric/trace scalers and log IDF) see training data
only. Host faults are reported separately because the paper ranks services.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from drain3 import TemplateMiner

SERVICES = ['adservice','cartservice','checkoutservice','currencyservice','emailservice',
            'frontend','paymentservice','productcatalogservice','recommendationservice','shippingservice']


def service_name(value):
    value = re.sub(r'-\d+$', '', str(value))
    if value in SERVICES:
        return value
    if value.endswith('2') and value[:-1] in SERVICES:
        return value[:-1]
    return None


def split_incidents(rows, seed):
    # Connected overlapping observation intervals stay together, including different faults.
    groups=[]
    for cloud in sorted({r['cloud'] for r in rows}):
        end=-1; group=[]
        for i in sorted((i for i,r in enumerate(rows) if r['cloud']==cloud), key=lambda i: rows[i]['start']):
            if rows[i]['start'] <= end:
                group.append(i); end=max(end,rows[i]['end'])
            else:
                if group: groups.append(group)
                group=[i]; end=rows[i]['end']
        if group:groups.append(group)
    np.random.default_rng(seed).shuffle(groups)
    test_count=max(1,round(len(groups)*.2)); val_count=max(1,round((len(groups)-test_count)*.1))
    for g,indices in enumerate(groups):
        split='test' if g<test_count else 'validation' if g<test_count+val_count else 'train'
        for i in indices:rows[i]['split']=split; rows[i]['overlap_group']=g
    return rows


def window_rows(frame, row):
    return frame[(frame.timestamp>=row['start']) & (frame.timestamp<row['end'])]


def normalize(array, train, baseline_bins):
    baseline=array[train,:,:baseline_bins,:]
    mean=np.nanmean(baseline, axis=(0,2),keepdims=True)
    std=np.nanstd(baseline, axis=(0,2),keepdims=True)
    available=np.isfinite(mean)
    mean=np.nan_to_num(mean);std=np.nan_to_num(std,nan=1.)
    std=np.maximum(std,1e-6)
    out=np.clip((array-mean)/std,-20,20)
    out=np.where(available,out,0.)
    return np.nan_to_num(out).astype('float32'), {'mean':mean.tolist(),'std':std.tolist(),'available':available.tolist()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',type=Path,default=Path('data/care/aiops2022-original/training_data_with_faults/training_data_with_faults'))
    p.add_argument('--metrics',type=Path,default=Path('/workspace/iwqos2026/data/2022_CCF_AIOps_challenge/raw_data/train_valid'))
    p.add_argument('--output',type=Path,default=Path('data/care/aiops2022-prepared'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--before',type=int,default=180)
    p.add_argument('--after',type=int,default=240)
    p.add_argument('--step',type=int,default=60)
    p.add_argument('--cloud',action='append',help='Optional explicit subset for integration checks; recorded in metadata')
    p.add_argument('--split-manifest',type=Path,help='Frozen train/test cohort; refit every preprocessing artifact from raw data')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.before+args.after)%args.step:raise ValueError('Window must divide into time bins')
    rows=[]; excluded=[]
    for path in sorted((args.original/'groundtruth').glob('groundtruth-*.csv')):
        match=re.fullmatch(r'groundtruth-k8s-(\d)-(\d{4}-\d{2}-\d{2})\.csv',path.name)
        if not match:continue
        cloud=f'{match[2]}-cloudbed{match[1]}'
        if args.cloud and cloud not in args.cloud:continue
        for raw in pd.read_csv(path).to_dict('records'):
            root=service_name(raw['cmdb_id'])
            if raw['level']=='node' or root is None:
                excluded.append({**raw,'cloud':cloud,'reason':'Not a uniquely labeled service fault'});continue
            timestamp=int(raw['timestamp'])
            rows.append({'incident_id':f'{cloud}:{timestamp}:{raw["cmdb_id"]}', 'cloud':cloud,
                         'root_service':root,'root_entity':raw['cmdb_id'],'fault_type':raw['failure_type'],
                         'fault_start':timestamp,'fault_end':None,'start':timestamp-args.before,
                         'end':timestamp+args.after,'source':str(path)})
    if not rows:raise ValueError('No original groundtruth files')
    # An observation must not include a different injected fault, including a host fault.
    all_events=rows+[{'cloud':r['cloud'],'fault_start':int(r['timestamp'])} for r in excluded]
    retained=[]
    for row in rows:
        others=[r for r in all_events if r is not row and r['cloud']==row['cloud']
                and row['start'] <= r['fault_start'] < row['end']]
        if others:
            excluded.append({**row,'reason':'Observation contains another fault onset'})
        else:retained.append(row)
    split_manifest=None
    if args.split_manifest:
        from paper_split import apply_manifest
        split_manifest=json.loads(args.split_manifest.read_text())
        rows=apply_manifest(retained,split_manifest)
        included={r['incident_id'] for r in rows}
        excluded.extend({**r,'reason':'Outside frozen paper-sized cohort'} for r in retained if r['incident_id'] not in included)
    else:
        rows=split_incidents(retained,args.seed)
    n,t=len(rows),(args.before+args.after)//args.step
    train=np.array([r['split']=='train' for r in rows])
    metric_names=None;metric=None;trace=np.zeros((n,10,t,3),dtype='float32')
    log_events=[[] for _ in rows];calls=[collections.Counter() for _ in rows]
    sources={};miner=TemplateMiner()
    for cloud in sorted({r['cloud'] for r in rows}):
        print('Read',cloud,flush=True)
        indices=[i for i,r in enumerate(rows) if r['cloud']==cloud]
        base=args.original/'tar'/cloud
        trace_path=base/'trace/all/trace_jaeger-span.csv'
        log_path=base/'log/all/log_filebeat-testbed-log-service.csv'
        for path in [trace_path,log_path]:
            proof=path.with_name(path.name+'.verified.json')
            if not path.exists() or not proof.exists():raise ValueError(f'Raw file not verified: {path}')
            sources[str(path)]=json.loads(proof.read_text())['sha256']
        metrics=args.metrics/cloud[:10]/('cloudbed-'+cloud[-1])/'raw_metric/container-all'
        for service in SERVICES:
            frames=[]
            for path in sorted(metrics.glob('*.csv')):
                if service_name(path.stem)==service:
                    f=pd.read_csv(path);frames.append(f)
                    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            if not frames:raise ValueError(f'Missing metrics for {cloud}/{service}')
            frame=pd.concat(frames).groupby('timestamp',as_index=False).mean(numeric_only=True)
            if metric_names is None:
                # Exclude IDs, timestamps, and container configuration; retain telemetry dimensions.
                metric_names=[c for c in frame if c.startswith(('CPU/','Memory/','Disk/','Network/','Thread/'))]
                metric=np.full((n,10,t,len(metric_names)),np.nan,dtype='float32')
            sid=SERVICES.index(service)
            for i in indices:
                f=window_rows(frame,rows[i]).copy()
                f['bin']=((f.timestamp-rows[i]['start'])//args.step).astype(int)
                for b,g in f.groupby('bin'):
                    metric[i,sid,b]=g.reindex(columns=metric_names).mean().values
        selected=[]
        for chunk in pd.read_csv(trace_path,chunksize=200000):
            chunk['timestamp']=pd.to_numeric(chunk.timestamp)/1000.
            mask=np.zeros(len(chunk),dtype=bool)
            for i in indices:mask |= (chunk.timestamp>=rows[i]['start']) & (chunk.timestamp<rows[i]['end'])
            if mask.any():selected.append(chunk[mask])
        frame=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame()
        if frame.empty:raise ValueError(f'No trace data in {cloud}')
        frame['service']=frame.cmdb_id.map(service_name)
        parent=frame.drop_duplicates(['trace_id','span_id']).set_index(['trace_id','span_id']).service.to_dict()
        for i in indices:
            f=window_rows(frame,rows[i]).copy();f['bin']=((f.timestamp-rows[i]['start'])//args.step).astype(int)
            for (svc,b),g in f.dropna(subset=['service']).groupby(['service','bin']):
                # AIOps Jaeger export uses duration in microseconds, error/status 0 for success.
                trace[i,SERVICES.index(svc),b]=[g.duration.mean()/1000.,(g.status_code!=0).mean(),len(g)/args.step]
            for span in f.itertuples():
                src=parent.get((span.trace_id,span.parent_span));dst=span.service
                if src in SERVICES and dst in SERVICES and src!=dst:calls[i][(src,dst)]+=1
        del frame,selected,parent
        for chunk in pd.read_csv(log_path,chunksize=50000):
            # Original service-log export uses Unix seconds (spans use milliseconds).
            chunk['timestamp']=pd.to_numeric(chunk.timestamp)
            entity_col=next((c for c in ['cmdb_id','service_name','service'] if c in chunk),None)
            message_col=next((c for c in ['value','log','message','log_message'] if c in chunk),None)
            if not entity_col or not message_col:raise ValueError(f'Unsupported log columns: {list(chunk)}')
            for i in indices:
                for log in window_rows(chunk,rows[i]).to_dict('records'):
                    svc=service_name(log[entity_col])
                    if svc is None:continue
                    message=str(log[message_col])
                    b=int((log['timestamp']-rows[i]['start'])//args.step)
                    log_events[i].append((SERVICES.index(svc),b,message))
                    if train[i]:miner.add_log_message(message)
    templates={c.cluster_id:c.get_template() for c in miner.drain.clusters}
    ids={key:j+1 for j,key in enumerate(sorted(templates))}
    logs=np.zeros((n,10,t,len(ids)+1),dtype='float32')
    for i,events in enumerate(log_events):
        for svc,b,message in events:
            cluster=miner.match(message)
            logs[i,svc,b,ids[cluster.cluster_id] if cluster else 0]+=1
    docs=logs[train].sum(axis=2).reshape(-1,logs.shape[-1])
    idf=np.log((1+len(docs))/(1+(docs>0).sum(axis=0)))+1
    logs=(np.log1p(logs)*idf).astype('float32')
    metric,ms=normalize(metric,train,max(1,args.before//args.step))
    trace,ts=normalize(trace,train,max(1,args.before//args.step))
    np.savez_compressed(args.output/'features.npz',metrics=metric,traces=trace,logs=logs,
                        labels=np.array([SERVICES.index(r['root_service']) for r in rows]))
    for i,row in enumerate(rows):
        row['calls']=[{'source':u,'target':v,'count':c} for (u,v),c in sorted(calls[i].items())]
    meta={'services':SERVICES,'metric_names':metric_names,'trace_names':['latency_ms','error_rate','call_frequency'],
          'log_templates':['<UNSEEN>']+[templates[k] for k in sorted(templates)],
          'metric_scaler':ms,'trace_scaler':ts,'log_idf':idf.tolist(),'seed':args.seed,
          'before':args.before,'after':args.after,'step':args.step,'source_hashes':sources,
          'selected_clouds':args.cloud,
          'split_manifest_sha256':hashlib.sha256(args.split_manifest.read_bytes()).hexdigest() if args.split_manifest else None,
          'split_protocol':split_manifest['protocol'] if split_manifest else '80/20 development/test; inner validation holdout',
          'counts':dict(collections.Counter(r['split'] for r in rows)),
          'limitations':['Metric tables were supplied already aggregated to one minute; original counters were not reprocessed.',
                         'Original labels identify fault start only; observation end is a protocol choice, not known fault recovery.',
                         'Original 128 incident IDs and random seed are unavailable; a paper-sized reconstruction is not the original split.']}
    for name,obj in [('incidents',rows),('metadata',meta),('excluded',excluded)]:
        (args.output/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    print('Prepared',len(rows),'incidents',meta['counts'],flush=True)


if __name__=='__main__':main()
