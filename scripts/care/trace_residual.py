"""Telemetry-only trace decomposition, without injection metadata or labels.

An RPC's client duration includes transport/queueing absent from its matching
server span. The difference implicates either endpoint, not automatically the
caller. Healthy incident edges help disambiguate the endpoints. Server exclusive
time subtracts the UNION of observed child intervals to avoid double counting.
"""
import collections
import gzip
import json
from pathlib import Path
import numpy as np


def interval_union(intervals, start, end):
    intervals=sorted((max(start,a),min(end,b)) for a,b in intervals if b>start and a<end)
    total=0.;right=start
    for a,b in intervals:
        total+=max(0,b-max(a,right));right=max(right,b)
    return total


def trace_records(trace, services):
    spans={s['spanID']:s for s in trace['spans']};processes=trace['processes']
    children=collections.defaultdict(list)
    for s in spans.values():
        for ref in s.get('references',[]):
            if ref.get('refType')=='CHILD_OF' and ref.get('traceID')==s.get('traceID'):
                children[ref['spanID']].append(s)
    def service(s):return processes[s['processID']]['serviceName']
    def kind(s):return next((t['value'] for t in s.get('tags',[]) if t['key']=='span.kind'),None)
    local=[];edges=[]
    for sid,s in spans.items():
        u=service(s)
        if u not in services:continue
        if kind(s)=='server':
            start=s['startTime'];end=start+s['duration']
            occupied=interval_union([(c['startTime'],c['startTime']+c['duration']) for c in children[sid]],start,end)
            local.append((start/1e6,u,s['duration']/1000,max(0,s['duration']-occupied)/1000))
        elif kind(s)=='client':
            targets=[c for c in children[sid] if kind(c)=='server' and service(c) in services and service(c)!=u]
            # One matched server avoids attributing batched/ambiguous fan-out.
            if len(targets)==1:
                child=targets[0]
                edges.append((s['startTime']/1e6,u,service(child),max(0,s['duration']-child['duration'])/1000))
    return local,edges


def extract_one(row,services,before):
    # Only source path and observation boundaries are read from the manifest.
    local=collections.defaultdict(list);edge=collections.defaultdict(list)
    with gzip.open(Path(row['source']).parent/'traces.jsonl.gz','rt') as f:
        for line in f:
            ls,es=trace_records(json.loads(line),set(services))
            for stamp,u,duration,exclusive in ls:
                if row['start']<=stamp<row['end']:
                    phase=int(stamp>=row['start']+before)
                    local[(u,phase,0)].append(duration);local[(u,phase,1)].append(exclusive)
            for stamp,u,v,gap in es:
                if row['start']<=stamp<row['end']:
                    edge[(u,v,int(stamp>=row['start']+before))].append(gap)
    return summarize_records(local,edge,services)


def summarize_records(local,edge,services):
    """Shared observation summary for native Jaeger and exported span records."""
    out=collections.defaultdict(dict)
    for u in services:
        for c,name in enumerate(['server','exclusive']):
            pre=local[(u,0,c)];post=local[(u,1,c)]
            out[u][name+':observed']=float(len(pre)>=3 and len(post)>=3)
            for q,label in [(.5,'median'),(.9,'p90')]:
                a=float(np.quantile(pre,q)) if pre else 0.;b=float(np.quantile(post,q)) if post else 0.
                out[u][name+':'+label+':post_log']=np.log1p(b)
                out[u][name+':'+label+':delta_log']=np.log1p(b)-np.log1p(a) if pre and post else 0.
        out[u]['exclusive_fraction:delta']=((np.mean(local[(u,1,1)])/(np.mean(local[(u,1,0)])+1e-6))-
                                           (np.mean(local[(u,0,1)])/(np.mean(local[(u,0,0)])+1e-6))) if local[(u,0,0)] and local[(u,1,0)] else 0.
    pairs={key[:2] for key in edge};summary=[]
    for u,v in sorted(pairs):
        pre=edge[(u,v,0)];post=edge[(u,v,1)]
        if len(pre)<3 or len(post)<3:continue
        a=float(np.quantile(pre,.9));b=float(np.quantile(post,.9))
        shock=max(0,np.log1p(b)-np.log1p(a))
        hot=float(b>max(5.,3*a))
        summary.append((u,v,shock,hot,max(0,b-a)))
    for u in services:
        for direction in ['incoming','outgoing','incident']:
            incident=[e for e in summary if (e[1]==u if direction=='incoming' else e[0]==u if direction=='outgoing' else u in e[:2])]
            shocks=[e[2] for e in incident];hot=[e[3] for e in incident]
            out[u][direction+':degree']=len(incident)
            out[u][direction+':gap_max_log']=max(shocks,default=0.)
            out[u][direction+':gap_mean_log']=float(np.mean(shocks)) if shocks else 0.
            out[u][direction+':hot_fraction']=float(np.mean(hot)) if hot else 0.
            out[u][direction+':hot_count']=sum(hot)
            out[u][direction+':gap_delta_ms_log']=np.log1p(max([e[4] for e in incident],default=0.))
        # Every suspect RPC implicates both ends. The endpoint whose other RPCs
        # stay healthy receives less support than an endpoint with broad impact.
    for u in services:
        peers={e[1] if e[0]==u else e[0] for e in summary if u in e[:2] and e[3]}
        peer=max([out[v]['incident:hot_fraction'] for v in peers],default=0.)
        out[u]['incident:hot_fraction_minus_peer']=out[u]['incident:hot_fraction']-peer
    names=list(out[services[0]])
    return np.array([[out[u][k] for k in names] for u in services],dtype='float32'),['rpc:'+k for k in names]


def main():
    data=Path('data/care/trainticket-prepared');out=Path('artifacts/care/trace-residual-trainticket');out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((data/'incidents.json').read_text());meta=json.loads((data/'metadata.json').read_text());arrays=[]
    for i,row in enumerate(rows):
        path=out/f'case-{i:03d}.npz'
        if path.exists():x=np.load(path)['x']
        else:
            x,names=extract_one(row,meta['services'],meta['before']);np.savez_compressed(path,x=x)
            (out/'feature_names.json').write_text(json.dumps(names,indent=2)+'\n')
        arrays.append(x)
        if (i+1)%10==0:print('Extracted',i+1,'/',len(rows),flush=True)
    np.savez_compressed(out/'features.npz',x=np.stack(arrays))


if __name__=='__main__':main()
