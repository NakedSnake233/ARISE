"""Read-only /proc sampling of ARISE task trees without recording credentials."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time

from .common import ROOT, json_dump


def processes():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            result[int(path.name)] = dict(ppid=int(fields[1]), identity=fields[19],
                ticks=int(fields[11]) + int(fields[12]), threads=int(fields[17]),
                rss_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'), state=fields[0])
        except (OSError, IndexError, ValueError):
            continue
    return result


def descendants(pid, table):
    found = {pid} if pid in table else set()
    while True:
        more = {p for p, r in table.items() if r['ppid'] in found} - found
        if not more:
            return found
        found |= more


def sample():
    state = json.loads((ROOT / 'results_v2/remaining_queue/queue_state.json').read_text())
    table = processes()
    rows = []
    for name, row in state['jobs'].items():
        if row['status'] != 'running':
            continue
        pid = row['attempts'][-1]['pid']
        tree = descendants(pid, table)
        environment = {}
        # Read only thread controls for this experiment; never save argv or keys.
        for child in sorted(tree):
            try:
                entries = (Path('/proc') / str(child) / 'environ').read_bytes().split(b'\0')
                for item in entries:
                    key, _, value = item.partition(b'=')
                    if key in (b'OMP_NUM_THREADS', b'MKL_NUM_THREADS', b'OPENBLAS_NUM_THREADS'):
                        environment[key.decode()] = value.decode()
            except OSError:
                pass
        rows.append(dict(id=name, resource=row['spec']['resource'], cores=row['spec']['cores'],
            pid=pid, component=row['spec']['expected_config'].get('component'),
            processes={str(p):table[p] for p in sorted(tree)},
            tree_rss_bytes=sum(table[p]['rss_bytes'] for p in tree),
            thread_controls=environment))
    cpu = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    mem = {k:int(v.split()[0])*1024 for k,v in
           (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    return dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), epoch=time.time(),
        affinity_cores=len(os.sched_getaffinity(0)), cpu_ticks=cpu, load=os.getloadavg(),
        mem_total_bytes=mem['MemTotal'], mem_available_bytes=mem['MemAvailable'],
        queue_updated=state['updated'], counts=state['counts'],
        scheduler_policy=state.get('scheduler_policy'), storage=state.get('storage'), jobs=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label',required=True,choices=('before','after'))
    parser.add_argument('--seconds',type=int,default=20)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 60:
        parser.error('sampling duration must be between 5 and 60 seconds')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
    folder = ROOT / 'profile_output' / f'arise_parallel_{args.label}_{stamp}'
    folder.mkdir(parents=True,exist_ok=False)
    samples = [sample()]
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        time.sleep(min(4, max(0, end-time.monotonic())))
        samples.append(sample())
    first,last = samples[0],samples[-1]
    delta = [b-a for a,b in zip(first['cpu_ticks'],last['cpu_ticks'])]
    elapsed = last['epoch']-first['epoch']
    total = sum(delta)
    busy = total-delta[3]-delta[4]
    job_cpu_seconds = {}
    for previous,current in zip(samples,samples[1:]):
        prev_jobs = {j['id']:j for j in previous['jobs']}
        for job in current['jobs']:
            old = prev_jobs.get(job['id'],{}).get('processes',{})
            ticks = sum(r['ticks']-old[p]['ticks'] for p,r in job['processes'].items()
                        if p in old and r['identity']==old[p]['identity'])
            job_cpu_seconds[job['id']] = job_cpu_seconds.get(job['id'],0) + ticks/os.sysconf('SC_CLK_TCK')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True).strip()
    summary = dict(label=args.label, duration_seconds=elapsed, available_logical_cpus=last['affinity_cores'],
        host_busy_percent=100*busy/total, host_busy_core_equivalent=last['affinity_cores']*busy/total,
        persistent_arise_tree_cpu_seconds=job_cpu_seconds,
        persistent_arise_busy_core_lower_bound=sum(job_cpu_seconds.values())/elapsed,
        cpu_jobs=sum(j['resource']=='cpu' for j in last['jobs']),
        gpu_jobs=sum(j['resource']=='gpu' for j in last['jobs']),
        assigned_cpu_cores=sum(j['cores'] for j in last['jobs'] if j['resource']=='cpu'),
        tree_rss_bytes_upper_bound=sum(j['tree_rss_bytes'] for j in last['jobs']),
        mem_available_bytes=last['mem_available_bytes'],gpu=gpu,
        observer='external 4-second /proc sampler; no training source or hyperparameters modified',
        limitations='Short sample; exited processes are omitted from CPU lower bound; summed RSS includes shared pages. Different queued methods prevent a controlled speedup claim.')
    json_dump(folder/'samples.json',samples)
    json_dump(folder/'summary.json',summary)
    gib=1024**3
    report = f'''# ARISE concurrency profile ({args.label})

| Dimension | Measurement |
|---|---:|
| Available logical CPUs | {summary['available_logical_cpus']} |
| Host CPU busy | {summary['host_busy_percent']:.2f}% ({summary['host_busy_core_equivalent']:.2f} core equivalents) |
| ARISE CPU / GPU jobs | {summary['cpu_jobs']} / {summary['gpu_jobs']} |
| CPU job declared cores | {summary['assigned_cpu_cores']} |
| Persistent ARISE process CPU lower bound | {summary['persistent_arise_busy_core_lower_bound']:.2f} cores |
| Summed task-tree RSS upper bound (shared pages counted repeatedly) | {summary['tree_rss_bytes_upper_bound']/gib:.2f} GiB |
| Available host RAM | {summary['mem_available_bytes']/gib:.2f} GiB |

GPU snapshot: {gpu}.

{summary['limitations']}

## Instrumentation changelog

| File | Change type | Addition | Lines |
|---|---|---|---|
| exp_revision/profile_parallel_runtime.py | created | Standalone sampler; no inline training instrumentation | — |
| {folder.relative_to(ROOT)}/samples.json | created | Per-sample system and experiment process-tree measurements | — |
| {folder.relative_to(ROOT)}/summary.json | created | Aggregated CPU, memory, GPU and concurrency measurements | — |
| {folder.relative_to(ROOT)}/profile_report.md | created | Results and instrumentation inventory | — |

The standalone profiling files can be removed after review without modifying experiment code.
'''
    (folder/'profile_report.md').write_text(report)
    print(json.dumps(dict(output=str(folder.relative_to(ROOT)),summary=summary),ensure_ascii=False))


if __name__=='__main__': main()
