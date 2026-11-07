"""Probe real request traces; optionally verify collector restart resilience.

Runs outside formal incidents and holds the same exclusive lock as collection.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import threading
import time
from collect_trainticket import ROOT,KUBE,kubectl,ready,workload,check_target_traces,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--restart-jaeger',action='store_true')
    args=p.parse_args();cfg=json.loads((ROOT/'configs/care/collection.json').read_text())
    output=ROOT/'artifacts/care/trainticket'/('trace-check-'+str(int(time.time())))
    output.mkdir(parents=True)
    lock=(ROOT/'artifacts/care/trainticket/collector.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if kubectl('-n','care-tt','get','stresschaos,networkchaos','-o','name').strip():
        raise RuntimeError('An active fault exists; refuse tracing infrastructure test')
    ready();forward={};stop=threading.Event();load=None;report={}
    def start_forward(service,ports):
        with (output/(service+'.log')).open('a') as log:
            forward[service]=subprocess.Popen(KUBE+['-n','care-tt','port-forward','svc/'+service,ports,'--address=127.0.0.1'],stdout=log,stderr=log)
    def probe():
        start=time.time();time.sleep(60);end=time.time()
        result={s:check_target_traces(s,start,end) for s in cfg['services']}
        print('Observed all',len(result),'injection target services in fresh traces',flush=True)
        return result
    def jaeger_pod():
        pods=json.loads(kubectl('-n','care-tt','get','pods','-l','app=jaeger','-o','json'))['items']
        return [{'uid':x['metadata']['uid'],'ip':x['status'].get('podIP')} for x in pods]
    try:
        for s,ports in [('ts-ui-dashboard','18080:8080'),('jaeger-query','16686:16686')]:start_forward(s,ports)
        time.sleep(4)
        if any(p.poll() is not None for p in forward.values()):raise RuntimeError('Port forwarding failed')
        load=threading.Thread(target=workload,args=(stop,output,cfg));load.start()
        report['stable_agent_ip']=json.loads(kubectl('-n','care-tt','get','svc','jaeger','-o','json'))['spec']['clusterIP']
        report['before_pod']=jaeger_pod();report['before']=probe();write_json(output/'result.json',report)
        if args.restart_jaeger:
            if report['stable_agent_ip']=='None':raise RuntimeError('Headless agent service remains')
            app_uids={p['metadata']['name']:p['metadata']['uid'] for p in ready() if p['metadata']['name'].startswith('ts-')}
            print(kubectl('-n','care-tt','rollout','restart','deployment/jaeger'),flush=True)
            print(kubectl('-n','care-tt','rollout','status','deployment/jaeger','--timeout=60s'),flush=True)
            old=forward['jaeger-query'];old.terminate();old.wait(timeout=10)
            start_forward('jaeger-query','16686:16686');time.sleep(4)
            report['after_pod']=jaeger_pod();report['after']=probe()
            assert app_uids=={p['metadata']['name']:p['metadata']['uid'] for p in ready() if p['metadata']['name'].startswith('ts-')}
            assert report['stable_agent_ip']==json.loads(kubectl('-n','care-tt','get','svc','jaeger','-o','json'))['spec']['clusterIP']
            assert report['before_pod']!=report['after_pod']
            report['application_clients_unchanged']=True
        report['passed']=True;write_json(output/'result.json',report)
        write_json(ROOT/'artifacts/care/trainticket/tracing-recovery-proof.json',report)
        print('PASS',output,flush=True)
    except BaseException as exc:
        report.update(passed=False,error=str(exc));write_json(output/'result.json',report);raise
    finally:
        stop.set()
        if load:load.join(timeout=30)
        for proc in forward.values():
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill()


if __name__=='__main__':main()
