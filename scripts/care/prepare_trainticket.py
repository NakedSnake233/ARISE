"""Convert completed, restored TrainTicket incidents into the CARE feature schema."""
import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from drain3 import TemplateMiner
from prepare_aiops import split_incidents, normalize

METRICS=['container_cpu_usage_seconds_total','container_memory_working_set_bytes','container_memory_usage_bytes',
         'container_network_receive_bytes_total','container_network_transmit_bytes_total',
         'container_fs_reads_bytes_total','container_fs_writes_bytes_total']


def parse_log_timestamp(value):
    # Kubernetes emits RFC3339Nano. Python 3.10 fromisoformat rejects nine
    # fractional digits; truncate sub-microseconds, preserving UTC/offset.
    value=re.sub(r'(\.\d{6})\d+(?=Z$|[+-]\d{2}:\d{2}$)',r'\1',value)
    parsed=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('Kubernetes timestamp must include timezone')
    return parsed.timestamp()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('data/care/trainticket-collected'))
    p.add_argument('--deployment',type=Path,default=Path('artifacts/care/trainticket/deployment_manifest.json'))
    p.add_argument('--output',type=Path,default=Path('data/care/trainticket-prepared'))
    p.add_argument('--seed',type=int,default=42);p.add_argument('--step',type=int,default=5)
    args=p.parse_args();services=json.loads(args.deployment.read_text())['application_services']
    rows=[];states=[];excluded=[]
    for path in sorted(args.source.glob('*/incident.json')):
        state=json.loads(path.read_text())
        if state['status']!='complete' or 'recovery_confirmed_at' not in state:
            excluded.append({'path':str(path),'status':state['status']});continue
        cfg=state['protocol'];duration=cfg['warmup_seconds']+cfg['fault_seconds']+cfg['cooldown_seconds']
        rows.append({'incident_id':state['incident_id'],'cloud':'care-tt','root_service':state['service'],
                     'fault_type':state['fault'],'fault_start':state['injection_confirmed_at'],
                     'fault_end':state['recovery_confirmed_at'],'start':state['observation_start'],
                     'end':state['observation_start']+duration,'source':str(path)})
        states.append(state)
    if len(rows)<15:raise ValueError('Need at least 15 complete incidents for train/validation/test; pilot is not training data')
    durations={r['end']-r['start'] for r in rows}
    befores={s['protocol']['warmup_seconds'] for s in states}
    if len(durations)!=1 or len(befores)!=1:raise ValueError('Mixed collection protocols; prepare separately')
    before=befores.pop();duration=durations.pop();t=int(duration)//args.step
    rows=split_incidents(rows,args.seed);train=np.array([r['split']=='train' for r in rows]);n=len(rows)
    metric=np.full((n,len(services),t,len(METRICS)),np.nan,dtype='float32')
    traces=np.zeros((n,len(services),t,3),dtype='float32');log_events=[[] for _ in rows]
    miner=TemplateMiner();hashes={};log_audit={'raw_lines':0,'invalid_timestamps':0,'in_window_lines':0,'incident_counts':{}}
    def svc_for_pod(name):
        return next((s for s in services if name.startswith(s+'-')),None)
    for i,row in enumerate(rows):
        print('Read',row['incident_id'],flush=True)
        folder=Path(row['source']).parent;hashes[row['source']]=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()
        with gzip.open(folder/'metrics.json.gz','rt') as f:data=json.load(f)['data']['result']
        buckets=collections.defaultdict(list)
        for series in data:
            labels=series['metric'];svc=svc_for_pod(labels.get('pod',''));name=labels.get('__name__')
            if svc is None or name not in METRICS:continue
            # Docker-backed kubelet may expose only pod cgroup aggregates with no
            # container label. Every application pod here has one application container.
            if 'network' not in name and labels.get('container') not in (None,'',svc):continue
            values=np.array(series['values'],dtype=float)
            if name.endswith('_total'):
                values=values.copy();values[1:,1]=np.maximum(0,np.diff(values[:,1]))/np.maximum(1e-6,np.diff(values[:,0]));values=values[1:]
            for timestamp,value in values:
                b=int((timestamp-row['start'])//args.step)
                if 0<=b<t:buckets[(services.index(svc),b,METRICS.index(name))].append(value)
        for key,values in buckets.items():metric[(i,)+key]=np.mean(values)
        latency=collections.defaultdict(list);errors=collections.Counter();calls=collections.Counter()
        with gzip.open(folder/'traces.jsonl.gz','rt') as f:
            for line in f:
                trace=json.loads(line);processes=trace['processes']
                lookup={span['spanID']:processes[span['processID']]['serviceName'] for span in trace['spans']}
                for span in trace['spans']:
                    svc=lookup[span['spanID']];b=int((span['startTime']/1e6-row['start'])//args.step)
                    if svc not in services or not 0<=b<t:continue
                    key=(services.index(svc),b);latency[key].append(span['duration']/1000.)
                    errors[key]+=any(tag.get('key')=='error' and tag.get('value') is True for tag in span.get('tags',[]))
                    for reference in span.get('references',[]):
                        src=lookup.get(reference['spanID'])
                        if reference.get('refType')=='CHILD_OF' and src in services and src!=svc:calls[(src,svc)]+=1
        for key,values in latency.items():traces[(i,)+key]=[np.mean(values),errors[key]/len(values),len(values)/args.step]
        row['calls']=[{'source':u,'target':v,'count':count} for (u,v),count in sorted(calls.items())]
        for path in folder.glob('*.log.gz'):
            svc=svc_for_pod(path.name)
            if svc is None:continue
            with gzip.open(path,'rt') as f:
                for line in f:
                    log_audit['raw_lines']+=1
                    fields=line.rstrip().split(' ',1)
                    if len(fields)!=2:continue
                    try:timestamp=parse_log_timestamp(fields[0])
                    except ValueError:
                        log_audit['invalid_timestamps']+=1
                        continue
                    b=int((timestamp-row['start'])//args.step)
                    if 0<=b<t:
                        # Remove embedded logger timestamp; the Kubernetes timestamp above is canonical.
                        message=re.sub(r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]\d+\s*','',fields[1])
                        log_events[i].append((services.index(svc),b,message))
                        log_audit['in_window_lines']+=1
                        if train[i]:miner.add_log_message(message)
        log_audit['incident_counts'][row['incident_id']]=len(log_events[i])
        if not log_events[i]:raise ValueError('No usable application logs in incident '+row['incident_id']+'; refuse empty log modality')
        if (i+1)%10==0:print('Parsed logs:',log_audit['in_window_lines'],'templates:',len(miner.drain.clusters),flush=True)
    templates={c.cluster_id:c.get_template() for c in miner.drain.clusters};ids={k:j+1 for j,k in enumerate(sorted(templates))}
    logs=np.zeros((n,len(services),t,len(ids)+1),dtype='float32')
    for i,events in enumerate(log_events):
        for svc,b,message in events:
            match=miner.match(message);logs[i,svc,b,ids[match.cluster_id] if match else 0]+=1
    docs=logs[train].sum(2).reshape(-1,len(ids)+1);idf=np.log((len(docs)+1)/(1+(docs>0).sum(0)))+1
    logs=(np.log1p(logs)*idf).astype('float32')
    if not templates or not np.any(logs[train]>0):raise ValueError('Training log modality is empty')
    available=np.isfinite(metric[train]).any(axis=(0,1,2))
    missing=[name for name,ok in zip(METRICS,available) if not ok]
    metric=metric[:,:,:,available]
    if not available.any():raise ValueError('No real metric dimensions available')
    metric,ms=normalize(metric,train,before//args.step);traces,ts=normalize(traces,train,before//args.step)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'features.npz',metrics=metric,traces=traces,logs=logs,
                        labels=np.array([services.index(r['root_service']) for r in rows]))
    metadata={'services':services,'metric_names':[name for name,ok in zip(METRICS,available) if ok],
              'missing_metric_dimensions':missing,'trace_names':['latency_ms','error_rate','call_frequency'],
              'log_templates':['<UNSEEN>']+[templates[k] for k in sorted(templates)],'metric_scaler':ms,'trace_scaler':ts,
              'log_idf':idf.tolist(),'seed':args.seed,'before':before,'after':duration-before,'step':args.step,
              'source_hashes':hashes,'counts':dict(collections.Counter(r['split'] for r in rows)),
              'log_parse_audit':log_audit,'log_timestamp_parser':'RFC3339Nano, truncate sub-microsecond precision; preserve timezone',
              'limitations':['41 application services + 22 databases; not a verified match to the papers 64-service deployment.',
                             'Memory ramp and bounded scratch I/O implement explicit proxies for unspecified paper injections.']}
    for name,obj in [('incidents',rows),('metadata',metadata),('excluded',excluded)]:
        (args.output/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
