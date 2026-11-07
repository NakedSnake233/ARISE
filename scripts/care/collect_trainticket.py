"""Serial Chaos Mesh experiments with tri-modal capture and verified rollback.

Uses only the dedicated care-tt context, namespace, and labeled pods. Each result
records requested, actual, and recovered timestamps. Incomplete captures are not
eligible for training. Run --plan first to materialize the experiment manifests.
"""
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import threading
import time
import requests

ROOT = Path(__file__).resolve().parents[2]
KUBE = ['kubectl', '--kubeconfig', str(ROOT/'artifacts/care/kubeconfig'), '--context', 'care-tt', '--request-timeout=30s']


def kubectl(*args, payload=None):
    result = subprocess.run(KUBE + list(args), input=json.dumps(payload) if payload is not None else None,
                            text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout


def chaos_manifest(service, fault, name, cfg):
    if cfg['namespace'] != 'care-tt' or cfg['context'] != 'care-tt' or not service.startswith('ts-'):
        raise ValueError('Faults must target the isolated care-tt experiment')
    spec = {'mode': 'one', 'duration': f"{cfg['fault_seconds']}s",
            'selector': {'namespaces': ['care-tt'], 'labelSelectors': {'app': service, 'care.experiment': 'trainticket'}},
            'containerNames': [service]}
    p = cfg['parameters']
    kind = 'StressChaos'
    if fault == 'cpu_load':
        spec['stressors'] = {'cpu': {'workers': 1, 'load': p['cpu_load_percent']}}
    elif fault == 'memory_ramp':
        spec['stressors'] = {'memory': {'workers': 1, 'size': p['memory_size'], 'options': ['--time', f"{cfg['fault_seconds']}s"]}}
    elif fault == 'disk_io_load':
        # In v2.8.4, stressngStressors is not forwarded to ExecStressors and
        # can panic when Stressors is nil. The supported CPU options channel
        # forwards stress-ng arguments. Keep its CPU worker at 0% load.
        spec['stressors'] = {'cpu': {'workers': 1, 'load': 0,
            'options': ['--hdd','1','--hdd-bytes',p['disk_bytes'],'--timeout',f"{cfg['fault_seconds']}s"]}}
    elif fault in ('network_delay', 'packet_loss'):
        kind = 'NetworkChaos'
        spec.pop('containerNames')
        spec['direction'] = 'to'
        spec['target'] = {'mode': 'all', 'selector': {'namespaces': ['care-tt'],
                          'labelSelectors': {'care.experiment': 'trainticket'}}}
        spec['action'] = 'delay' if fault == 'network_delay' else 'loss'
        spec[spec['action']] = ({'latency': p['network_delay'], 'jitter': '20ms', 'correlation': '0'}
                                if fault == 'network_delay' else {'loss': p['packet_loss_percent'], 'correlation': '0'})
    else:
        raise ValueError(fault)
    return {'apiVersion': 'chaos-mesh.org/v1alpha1', 'kind': kind,
            'metadata': {'name': name, 'namespace': 'care-tt', 'labels': {'care.experiment': 'trainticket'}}, 'spec': spec}


def get_json(url, **params):
    session = requests.Session()
    session.trust_env = False
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def append_json(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def write_json(path, obj):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def business_services(trace, start=None, end=None):
    """Only count application spans, not unused process entries or Jaeger itself."""
    result=set()
    for span in trace.get('spans',[]):
        timestamp=span['startTime']/1e6
        if start is not None and timestamp<start:continue
        if end is not None and timestamp>end:continue
        name=trace.get('processes',{}).get(span.get('processID'),{}).get('serviceName','')
        if name.startswith('ts-'):result.add(name)
    return result


def query_traces(service, start, end):
    """Split saturated query windows instead of silently losing excess traces."""
    result=get_json('http://127.0.0.1:16686/api/traces',service=service,
                    start=int(start*1e6),end=int(end*1e6),limit=1000)
    if result.get('errors'):raise RuntimeError('Jaeger query errors: '+str(result['errors']))
    traces=result.get('data') or []
    if len(traces)<1000:return traces
    if end-start<.001:raise RuntimeError('Trace query saturated even at sub-millisecond resolution')
    middle=(start+end)/2
    return query_traces(service,start,middle)+query_traces(service,middle,end)


def check_target_traces(service, start, end):
    traces=query_traces(service,start,end)
    matched=[t for t in traces if service in business_services(t,start,end)]
    if not matched:
        raise RuntimeError('No fresh application spans for injection target '+service+'; refuse fault injection')
    return {'service':service,'window':[start,end],'trace_count':len(matched),
            'sample_trace_ids':[t['traceID'] for t in matched[:5]]}


def workload(stop, output, cfg):
    date = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).strftime('%Y-%m-%d')
    def user(index):
        session = requests.Session(); session.trust_env = False
        while not stop.is_set():
            start = time.time()
            endpoint = '/api/v1/travelservice/trips/left' if index % 2 == 0 else '/api/v1/travel2service/trips/left'
            try:
                r = session.post('http://127.0.0.1:18080'+endpoint,
                                 json={'startingPlace': 'Shang Hai', 'endPlace': 'Su Zhou', 'departureTime': date}, timeout=20)
                row = {'time': start, 'endpoint': endpoint, 'status': r.status_code,
                       'latency': time.time()-start, 'body': r.text[:160]}
                try:
                    payload=r.json()
                    row['application_success']=payload.get('status')==1 and bool(payload.get('data'))
                except (ValueError,AttributeError):
                    row['application_success']=False
            except requests.RequestException as exc:
                row = {'time': start, 'endpoint': endpoint, 'error': str(exc)}
            append_json(output/f'workload-{index}.jsonl', row)
            stop.wait(cfg['request_interval_seconds'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['users']) as pool:
        list(pool.map(user, range(cfg['users'])))


def capture_traces(stop, path, start, errors):
    seen = {};watermarks={}
    with gzip.open(path, 'wt') as stream:
        while True:
            final_pass=stop.is_set()
            try:
                services = get_json('http://127.0.0.1:16686/api/services')['data'] or []
                # Querying Jaeger itself produces self-traces and can make an
                # otherwise empty capture appear healthy. Capture application data.
                for service in sorted(s for s in services if s.startswith('ts-')):
                    end=time.time()
                    traces=query_traces(service,max(start,watermarks.get(service,start)-30),end)
                    for trace in traces:
                        if not business_services(trace,start):continue
                        previous = seen.get(trace['traceID'])
                        if previous is None or len(trace['spans']) > len(previous['spans']):
                            seen[trace['traceID']] = trace
                    watermarks[service]=end
            except Exception as exc:
                errors.append('trace: '+str(exc))
            if final_pass:break
            stop.wait(10)
        for trace in seen.values():
            stream.write(json.dumps(trace)+'\n')
    return len(seen)


def collect_metrics(path, start, end):
    response = get_json('http://127.0.0.1:19090/api/v1/query_range',
                        query='{__name__=~"container_cpu_usage_seconds_total|container_memory_working_set_bytes|container_memory_usage_bytes|container_network_receive_bytes_total|container_network_transmit_bytes_total|container_fs_reads_bytes_total|container_fs_writes_bytes_total",namespace="care-tt",pod!=""}',
                        start=start, end=end, step=5)
    if response.get('status') != 'success' or not response['data']['result']:
        raise RuntimeError('No container metrics collected')
    with gzip.open(path, 'wt') as f:
        json.dump(response, f)


def ready():
    pods = json.loads(kubectl('-n', 'care-tt', 'get', 'pods', '-l', 'care.experiment=trainticket', '-o', 'json'))['items']
    if len(pods) < 60:
        raise RuntimeError(f'Incomplete deployment: {len(pods)} labeled pods')
    bad = [p['metadata']['name'] for p in pods if not any(c['type']=='Ready' and c['status']=='True' for c in p.get('status', {}).get('conditions', []))]
    if bad:
        raise RuntimeError('Pods not ready: '+', '.join(bad))
    return pods


def wait_condition(kind, name, condition, timeout=60):
    kubectl('-n', 'care-tt', 'wait', f'{kind}/{name}', f'--for=condition={condition}', f'--timeout={timeout}s')


def run_incident(item, cfg, output):
    folder = output/item['incident_id']; folder.mkdir(exist_ok=True)
    state_path = folder/'incident.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('status') == 'complete':
        return
    if state_path.exists():
        archive=output/'failed_attempts';archive.mkdir(exist_ok=True)
        folder.rename(archive/(item['incident_id']+'-'+str(time.time_ns())))
        folder.mkdir()
    ready()
    if shutil.disk_usage(output).free < 8*1024**3:
        raise RuntimeError('Less than 8 GiB free disk')
    manifest = chaos_manifest(item['service'], item['fault'], item['incident_id'], cfg)
    (folder/'chaos.json').write_text(json.dumps(manifest, indent=2))
    state = {**item, 'status': 'collecting', 'observation_start': time.time(), 'errors': [], 'protocol': cfg,
             'collector_version':2,
             'trace_capture_policy':'application spans only; target baseline gate; per-service overlapping query watermarks; final flush'}
    write_json(state_path,state)
    trace_stop = threading.Event(); capture = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = capture.submit(capture_traces, trace_stop, folder/'traces.jsonl.gz', state['observation_start'], state['errors'])
    applied = False
    try:
        time.sleep(cfg['warmup_seconds'])
        healthy = 0
        for path in output.glob('workload-*.jsonl'):
            for line in path.read_text().splitlines():
                request = json.loads(line)
                if request['time'] >= state['observation_start'] and request.get('status') == 200:
                    try:
                        body = json.loads(request.get('body', ''))
                        healthy += body.get('status') == 1 and bool(body.get('data'))
                    except json.JSONDecodeError:
                        # Body is a preview; a full response success is separately recorded.
                        healthy += bool(request.get('application_success'))
        if healthy == 0:
            raise RuntimeError('No successful nonempty baseline business requests; refuse fault injection')
        state['baseline_trace_check']=check_target_traces(item['service'],state['observation_start'],time.time())
        write_json(state_path,state)
        state['injection_requested_at'] = time.time()
        # Set before apply: cleanup also runs if the API accepted it but the response was lost.
        applied = True
        kubectl('apply', '-f', '-', payload=manifest)
        wait_condition(manifest['kind'], item['incident_id'], 'AllInjected')
        state['injection_confirmed_at'] = time.time()
        state['chaos_injected_status'] = json.loads(kubectl('-n','care-tt','get',manifest['kind'],item['incident_id'],'-o','json'))
        write_json(state_path,state)
        time.sleep(cfg['fault_seconds'])
        kubectl('-n', 'care-tt', 'delete', manifest['kind'], item['incident_id'], '--wait=true', '--timeout=60s')
        applied = False; state['recovery_confirmed_at'] = time.time()
        write_json(state_path,state)
        time.sleep(cfg['cooldown_seconds'])
        ready()
        state['observation_end'] = time.time()
        collect_metrics(folder/'metrics.json.gz', state['observation_start'], state['observation_end'])
        since = dt.datetime.fromtimestamp(state['observation_start'], dt.timezone.utc).isoformat().replace('+00:00','Z')
        for pod in ready():
            name = pod['metadata']['name']
            log = kubectl('-n','care-tt','logs',name,'--all-containers=true','--timestamps=true',f'--since-time={since}')
            with gzip.open(folder/(name+'.log.gz'),'wt') as f:
                f.write(log)
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'failed'; state['errors'].append(str(exc))
        if applied:
            try:
                state['chaos_failure_status']=json.loads(kubectl('-n','care-tt','get',manifest['kind'],item['incident_id'],'-o','json'))
            except Exception:
                pass
        raise
    finally:
        if applied:
            try:
                kubectl('-n','care-tt','delete',manifest['kind'],item['incident_id'],'--ignore-not-found=true','--wait=true','--timeout=60s')
                state['recovery_confirmed_at'] = time.time()
            except Exception as exc:
                state['errors'].append('RECOVERY FAILED: '+str(exc))
        trace_stop.set()
        try:
            state['trace_count'] = future.result(timeout=90)
        except Exception as exc:
            state['trace_count']=0;state['errors'].append('Trace export finalization failed: '+str(exc))
        finally:
            capture.shutdown(wait=True)
        if not state['trace_count'] or state['errors']:
            state['status'] = 'incomplete'
        if state['trace_count']:
            observed=set()
            with gzip.open(folder/'traces.jsonl.gz','rt') as f:
                for line in f:
                    trace=json.loads(line)
                    observed.update(business_services(trace,state['observation_start'],state.get('observation_end',time.time())))
            state['observed_trace_services']=sorted(observed)
            if item['service'] not in observed:
                state['status']='incomplete';state['errors'].append('Injection target absent from collected traces')
        write_json(state_path,state)
        if state['status'] != 'complete':
            print('Capture not eligible for training:', folder, flush=True)
            raise RuntimeError('Incomplete capture: inspect incident.json; campaign stopped')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'configs/care/collection.json')
    p.add_argument('--output', type=Path, default=ROOT/'data/care/trainticket-collected')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--limit', type=int)
    args = p.parse_args(); cfg=json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    lock=None
    if not args.plan:
        lock=(ROOT/'artifacts/care/trainticket/collector.lock').open('w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign={'pid':os.getpid(),'state':'starting','output':str(args.output),'started_at':time.time()}
    def campaign_status(state,**extra):
        campaign.update(state=state,updated_at=time.time(),**extra)
        write_json(args.output/'campaign-status.json',campaign)
    schedule=[{'incident_id': f'care-{s[3:-8]}-{f.replace("_","-")}-{r}', 'service':s,'fault':f,'repetition':r}
              for s in cfg['services'] for f in cfg['faults'] for r in range(cfg['repetitions'])]
    random.Random(cfg['seed']).shuffle(schedule)
    (args.output/'schedule.json').write_text(json.dumps(schedule, indent=2)+'\n')
    if args.plan:
        plans=[chaos_manifest(x['service'],x['fault'],x['incident_id'],cfg) for x in schedule]
        (args.output/'chaos-plan.json').write_text(json.dumps(plans, indent=2)+'\n')
        print(f'{len(schedule)} incidents; {len(schedule)*(cfg["warmup_seconds"]+cfg["fault_seconds"]+cfg["cooldown_seconds"])/3600:.1f} hours minimum')
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    forward=[]; stop=threading.Event(); load=None
    try:
        campaign_status('starting')
        for service, ports in [('ts-ui-dashboard','18080:8080'),('jaeger-query','16686:16686'),('care-prometheus','19090:9090')]:
            log=(args.output/f'port-forward-{service}.log').open('a')
            forward.append(subprocess.Popen(KUBE+['-n','care-tt','port-forward',f'svc/{service}',ports,'--address=127.0.0.1'],stdout=log,stderr=log))
            log.close()
        time.sleep(4)
        if any(proc.poll() is not None for proc in forward):
            raise RuntimeError('Port-forward failed; inspect logs')
        ready()
        get_json('http://127.0.0.1:16686/api/services')
        load=threading.Thread(target=workload,args=(stop,args.output,cfg)); load.start()
        for item in schedule[:args.limit]:
            print('Collecting',item['incident_id'],flush=True)
            campaign_status('collecting',incident_id=item['incident_id'])
            run_incident(item,cfg,args.output)
        campaign_status('complete')
    except BaseException as exc:
        campaign_status('failed',error=str(exc))
        raise
    finally:
        stop.set()
        if load:load.join(timeout=30)
        for proc in forward:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill()


if __name__=='__main__':
    main()
