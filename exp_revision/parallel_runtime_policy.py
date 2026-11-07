"""Bound outer-job concurrency without changing frozen experiment job specs."""
from __future__ import annotations

import math
import os
from pathlib import Path

GIB = 1024 ** 3
CPU_SLOTS = 12
GPU_SLOTS = 3
LLM_SLOTS = 3
MAX_DECLARED_CORES = 64
MEMORY_FLOOR_BYTES = 24 * GIB


def cpu_capacity():
    """Affinity and applicable ancestor quotas, including cgroup v1 hosts."""
    capacity = float(len(os.sched_getaffinity(0)))
    memberships = Path('/proc/self/cgroup').read_text().splitlines()
    for line in memberships:
        _, controllers, membership = line.split(':', 2)
        if 'cpu' in controllers.split(','):
            base = Path('/sys/fs/cgroup/cpu')
            relative = membership.strip('/')
            node = base / relative
            while node.is_relative_to(base):
                try:
                    quota = int((node/'cpu.cfs_quota_us').read_text())
                    period = int((node/'cpu.cfs_period_us').read_text())
                    if quota > 0 and period > 0:
                        capacity = min(capacity,quota/period)
                except FileNotFoundError:
                    pass
                if node == base: break
                node = node.parent
        elif controllers == '':
            base = Path('/sys/fs/cgroup')
            node = base / membership.strip('/')
            while node.is_relative_to(base):
                try:
                    quota,period = (node/'cpu.max').read_text().split()
                    if quota != 'max': capacity = min(capacity,int(quota)/int(period))
                except FileNotFoundError:
                    pass
                if node == base: break
                node = node.parent
    return max(1,math.floor(capacity))


def snapshot():
    available = cpu_capacity()
    memory = {key:int(value.split()[0])*1024 for key,value in
              (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    return dict(cpu_slots=min(CPU_SLOTS,available),gpu_slots=GPU_SLOTS,llm_slots=LLM_SLOTS,
        available_logical_cpus=available,
        max_declared_cores=min(MAX_DECLARED_CORES,max(1,available//2)),
        mem_available_bytes=memory['MemAvailable'],memory_floor_bytes=MEMORY_FLOOR_BYTES,
        job_memory_reservation='4 GiB evidence / 2 GiB other CPU / 1 GiB GPU; conservative extra headroom above observed available memory',
        storage_reservations='unchanged shared DiskGuard: 512 MiB per job and 20 GiB experiment cap')


def llm_job(job):
    return job['expected_config'].get('component') == 'llm_baseline'


def memory_reservation(job):
    if job['resource']=='gpu': return GIB
    if job['expected_config'].get('component')=='evidence': return 4*GIB
    return 2*GIB


def usage(state):
    jobs = [row['spec'] for row in state['jobs'].values() if row['status']=='running']
    return dict(cpu_jobs=sum(j['resource']=='cpu' for j in jobs),
        gpu_jobs=sum(j['resource']=='gpu' for j in jobs),llm_jobs=sum(llm_job(j) for j in jobs),
        declared_cores=sum(j['cores'] for j in jobs),
        additional_memory_headroom_bytes=sum(memory_reservation(j) for j in jobs))


def block_reason(job,state,hardware,current=None):
    current=usage(state) if current is None else current
    resource=job['resource']
    if current[resource+'_jobs'] >= hardware[resource+'_slots']: return resource+'_slots'
    if llm_job(job) and current['llm_jobs'] >= hardware['llm_slots']: return 'llm_slots'
    if current['declared_cores']+job['cores'] > hardware['max_declared_cores']: return 'declared_cores'
    required=(hardware['memory_floor_bytes']+current['additional_memory_headroom_bytes']
              +memory_reservation(job))
    if hardware['mem_available_bytes'] < required: return 'memory_headroom'
    return None
