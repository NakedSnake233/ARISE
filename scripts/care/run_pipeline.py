"""Resume the real-data preparation/localizer/SFT pipeline with explicit status.

No synthetic replacement is generated if a source or collection is incomplete.
TrainTicket waits for ALL scheduled incidents, not only a convenient subset.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['aiops2022','trainticket'],required=True)
    p.add_argument('--wait',action='store_true');args=p.parse_args()
    output=ROOT/'artifacts/care'/('pipeline-'+args.dataset);output.mkdir(parents=True,exist_ok=True)
    status={'dataset':args.dataset,'state':'waiting_for_sources','steps':[],'pid':os.getpid()}
    def save():
        status['updated_at']=time.time()
        tmp=output/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(output/'status.json')
    def sources_ready():
        if args.dataset=='aiops2022':
            base=ROOT/'data/care/aiops2022-original'
            index=base/'upstream_index.json'
            if not index.exists():return False
            records=json.loads(index.read_text())
            records=[r for r in records if r['path'].endswith(('trace_jaeger-span.csv','log_filebeat-testbed-log-service.csv'))]
            return len(records)==14 and all((base/r['path']).is_file() and (base/(r['path']+'.verified.json')).is_file() for r in records)
        base=ROOT/'data/care/trainticket-collected';schedule=base/'schedule.json'
        if not schedule.exists():return False
        for row in json.loads(schedule.read_text()):
            path=base/row['incident_id']/'incident.json'
            if not path.exists() or json.loads(path.read_text()).get('status')!='complete':return False
        return True
    save()
    while not sources_ready():
        if not args.wait:raise RuntimeError('Sources are incomplete; use --wait or finish acquisition first')
        time.sleep(30);save()
    data=ROOT/'data/care'/(args.dataset+'-prepared');base=ROOT/'artifacts/care'/('base-'+args.dataset)
    sft=ROOT/'data/care'/('sft-'+args.dataset)
    steps=[('prepare',['prepare_aiops.py' if args.dataset=='aiops2022' else 'prepare_trainticket.py']),
           ('base',['train_base.py','--data',str(data),'--output',str(base)]),
           ('sft',['build_sft.py','--data',str(data),'--base',str(base),'--output',str(sft)])]
    try:
        for name,command in steps:
            status['state']='running_'+name;save()
            with (output/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/care'/command[0])]+command[1:],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            status['steps'].append({'stage':name,'finished_at':time.time()});save()
        status['state']='complete';status['sft_output']=str(sft);save()
    except Exception as exc:
        status['state']='failed';status['error']=str(exc);save();raise


if __name__=='__main__':main()
