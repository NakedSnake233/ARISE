"""Adapt AIOps exported spans to RPC-residual evidence, without fault labels.

The export has no span.kind. Accept a client/server pair only when a direct
cross-service parent-child link names the same normalized operation. Ambiguous
multiple children and spans assigned conflicting roles are excluded. This is
inferred pairing, not recovered native span.kind; coverage is recorded.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_aiops import service_name
from trace_residual import interval_union, summarize_records


def operation(value):
    name=str(value).lstrip('/')
    if name.startswith('grpc.'):name=name[5:]
    return name


def records(frame):
    lookup={(str(s.trace_id),str(s.span_id)):s for s in frame.itertuples()}
    children=collections.defaultdict(list);pairs=[];audit=collections.Counter()
    for key,s in lookup.items():
        parent=(str(s.trace_id),str(s.parent_span));children[parent].append(s)
    for key,parent in lookup.items():
        u=service_name(parent.cmdb_id)
        if u is None:continue
        cross=[c for c in children[key] if service_name(c.cmdb_id) not in [u,None]]
        audit['cross_service_links']+=len(cross)
        matched=[c for c in cross if operation(c.operation_name)==operation(parent.operation_name)
                 and operation(c.operation_name) not in ['', 'nan']]
        if len(cross)==1 and len(matched)==1:pairs.append((parent,matched[0]))
    client={(str(a.trace_id),str(a.span_id)) for a,b in pairs}
    server={(str(b.trace_id),str(b.span_id)) for a,b in pairs};conflict=client&server
    audit['role_conflicts']=len(conflict);local=[];edges=[];seen=set()
    for parent,child in pairs:
        pk=(str(parent.trace_id),str(parent.span_id));ck=(str(child.trace_id),str(child.span_id))
        if pk in conflict or ck in conflict:continue
        if parent.duration<0 or child.duration<0:continue
        u,v=service_name(parent.cmdb_id),service_name(child.cmdb_id)
        audit['matched_pairs']+=1;audit['negative_unclipped_gap']+=parent.duration<child.duration
        edges.append((parent.timestamp/1000.,u,v,max(0,parent.duration-child.duration)/1000.))
        if ck not in seen:
            seen.add(ck);start=child.timestamp*1000.;end=start+child.duration
            intervals=[(c.timestamp*1000.,c.timestamp*1000.+c.duration) for c in children[ck]
                       if c.duration>=0]
            exclusive=max(0,child.duration-interval_union(intervals,start,end))/1000.
            local.append((child.timestamp/1000.,v,child.duration/1000.,exclusive))
    audit['observed_server_spans']=len(local)
    return local,edges,{k:int(v) for k,v in audit.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/care/aiops2022-paper128-prepared'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.data/'incidents.json').read_text());meta=json.loads((args.data/'metadata.json').read_text())
    arrays=[None]*len(rows);audits=[]
    for cloud in sorted({r['cloud'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['cloud']==cloud]
        path=Path('data/care/aiops2022-original/training_data_with_faults/training_data_with_faults/tar')/cloud/'trace/all/trace_jaeger-span.csv'
        selected=[]
        for chunk in pd.read_csv(path,chunksize=200000,dtype={'trace_id':str,'span_id':str,'parent_span':str}):
            keep=np.zeros(len(chunk),bool)
            for i in ids:keep|=(chunk.timestamp>=rows[i]['start']*1000)&(chunk.timestamp<rows[i]['end']*1000)
            if keep.any():selected.append(chunk[keep])
        frame=pd.concat(selected,ignore_index=True);ls,es,audit=records(frame);audits.append({'cloud':cloud,**audit})
        for i in ids:
            row=rows[i];local=collections.defaultdict(list);edge=collections.defaultdict(list)
            for stamp,u,duration,exclusive in ls:
                if row['start']<=stamp<row['end']:
                    phase=int(stamp>=row['start']+meta['before'])
                    local[(u,phase,0)].append(duration);local[(u,phase,1)].append(exclusive)
            for stamp,u,v,gap in es:
                if row['start']<=stamp<row['end']:
                    phase=int(stamp>=row['start']+meta['before']);edge[(u,v,phase)].append(gap)
            arrays[i],names=summarize_records(local,edge,meta['services'])
        print(cloud,audit,flush=True)
    np.savez_compressed(args.output/'features.npz',x=np.stack(arrays))
    (args.output/'feature_names.json').write_text(json.dumps(names,indent=2)+'\n')
    (args.output/'coverage.json').write_text(json.dumps({'clouds':audits,'limitations':[
        'No native span.kind in export: same-operation direct cross-service links infer RPC pairs.',
        'Unmatched or ambiguous links excluded, not imputed as confirmed pairs.',
        'Timestamp precision is milliseconds; exclusive time is approximate.']},indent=2)+'\n')


if __name__=='__main__':main()
