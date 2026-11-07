import copy
import json
from pathlib import Path
import signal

import pytest

from exp_revision import parallel_runtime_policy as policy
from exp_revision import extended_queue_v4 as queue


def job(name,resource='cpu',cores=1,component='evidence'):
    return dict(id=name,resource=resource,cores=cores,timeout_seconds=600,
        max_attempts=2,command=['-m','fixture'],source_commit='frozen-commit',
        expected_config=dict(task='T8',component=component,dataset='re2_ob',smoke=False),
        test_ids=['held-out-event'],score_shape=[12,1,11],expected_status=['completed_re2_scores'])


def row(spec,status='pending'):
    return dict(spec=spec,status=status,attempts=[] if status=='pending' else
        [dict(pid=100,identity='unchanged-identity',key=spec['id'],output='output/'+spec['id'],
              log='logs/'+spec['id'],status='running',runtime_timeout_seconds=600)])


def hardware():
    return dict(cpu_slots=12,gpu_slots=3,llm_slots=3,max_declared_cores=64,
        mem_available_bytes=180*policy.GIB,memory_floor_bytes=24*policy.GIB)


def test_core_and_memory_limits_count_adopted_gpu_and_cpu_jobs():
    state=dict(jobs={'cpu':row(job('cpu',cores=4),'running'),
                     'gpu':row(job('gpu','gpu',cores=1),'running')})
    new=job('new',cores=4);before=copy.deepcopy(state)
    hw=hardware();hw['max_declared_cores']=8
    assert policy.block_reason(new,state,hw)=='declared_cores'
    hw['max_declared_cores']=64
    hw['mem_available_bytes']=32*policy.GIB
    assert policy.block_reason(new,state,hw)=='memory_headroom'
    hw['mem_available_bytes']=33*policy.GIB
    assert policy.block_reason(new,state,hw) is None
    assert state==before


def test_llm_limit_does_not_block_independent_cpu_training():
    state=dict(jobs={str(i):row(job(str(i),component='llm_baseline'),'running') for i in range(3)})
    assert policy.block_reason(job('fourth',component='llm_baseline'),state,hardware())=='llm_slots'
    assert policy.block_reason(job('training',cores=4),state,hardware()) is None
    state['jobs'].update({str(i):row(job(str(i)),'running') for i in range(3,12)})
    assert policy.block_reason(job('thirteenth'),state,hardware())=='cpu_slots'


def test_cpu_capacity_respects_v1_ancestor_quota(monkeypatch):
    texts={'/proc/self/cgroup':'4:cpu,cpuacct:/user.slice/session.scope\n',
        '/sys/fs/cgroup/cpu/cpu.cfs_quota_us':'-1',
        '/sys/fs/cgroup/cpu/cpu.cfs_period_us':'100000',
        '/sys/fs/cgroup/cpu/user.slice/cpu.cfs_quota_us':'350000',
        '/sys/fs/cgroup/cpu/user.slice/cpu.cfs_period_us':'100000'}
    def read(path,*args,**kwargs):
        if str(path) not in texts: raise FileNotFoundError(path)
        return texts[str(path)]
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(policy.os,'sched_getaffinity',lambda pid:set(range(128)))
    assert policy.cpu_capacity()==3


def test_cpu_capacity_respects_v2_quota_and_affinity(monkeypatch):
    texts={'/proc/self/cgroup':'0::/user.slice/session.scope\n',
           '/sys/fs/cgroup/user.slice/cpu.max':'200000 100000',
           '/sys/fs/cgroup/cpu.max':'max 100000'}
    def read(path,*args,**kwargs):
        if str(path) not in texts: raise FileNotFoundError(path)
        return texts[str(path)]
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(policy.os,'sched_getaffinity',lambda pid:set(range(128)))
    assert policy.cpu_capacity()==2
    monkeypatch.setattr(policy.os,'sched_getaffinity',lambda pid:{0})
    assert policy.cpu_capacity()==1


def test_scheduler_adopts_children_and_fills_twelve_cpu_slots_without_changing_specs(tmp_path,monkeypatch):
    manifests=tmp_path/'manifests';manifests.mkdir()
    folder=tmp_path/'queue';folder.mkdir()
    state=dict(jobs={},manifests=[],status='running')
    for i in range(4):state['jobs'][f'old_cpu_{i}']=row(job(f'old_cpu_{i}',cores=4),'running')
    for i in range(2):state['jobs'][f'old_gpu_{i}']=row(job(f'old_gpu_{i}','gpu',component='Eadro'),'running')
    state['jobs']['z_gpu']=row(job('z_gpu','gpu',component='Eadro'))
    for i in range(5):state['jobs'][f'a_llm_{i}']=row(job(f'a_llm_{i}',component='llm_baseline'))
    for i in range(20):state['jobs'][f'b_model_{i:02d}']=row(job(f'b_model_{i:02d}',cores=4))
    before=copy.deepcopy(state);(folder/'queue_state.json').write_text(json.dumps(state))
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    monkeypatch.setattr(queue.q,'FOLDER',folder)
    monkeypatch.setattr(queue.q,'MANIFESTS',manifests)
    monkeypatch.setattr(queue.q,'load_manifests',lambda *args:None)
    monkeypatch.setattr(queue.q,'alive',lambda *args:True)
    monkeypatch.setattr(queue.q,'CPU_BUDGET',260.)
    monkeypatch.setattr(queue.q,'GPU_BUDGET',56.)
    monkeypatch.setattr(queue.q,'SLOTS',dict(cpu=8,gpu=3))
    monkeypatch.setattr(queue.compute,'apply',lambda:dict(limits=dict(cpu=260.,gpu=56.)))
    monkeypatch.setattr(queue.parallel,'snapshot',hardware)
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *args,**kwargs:'80000\n')
    adopted=[];started=[]
    class Guard:
        def __init__(self,root):pass
        def poll(self):return dict(safe_to_launch=True,abort_job_keys=[])
        def reserve(self,key,output,log,*,child_pid):adopted.append((key,child_pid))
    monkeypatch.setattr(queue,'DiskGuard',Guard)
    def start(spec,r,current,guard):
        started.append(spec['id']);r['status']='running'
        r['attempts'].append(dict(pid=200,identity='new-identity',status='running',runtime_timeout_seconds=600))
        return object()
    monkeypatch.setattr(queue,'start',start)
    signals={}
    monkeypatch.setattr(queue.signal,'signal',lambda s,f:signals.update({s:f}))
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:signals[signal.SIGTERM]())
    result=queue.run()
    assert started[0]=='z_gpu'  # GPU refill precedes additional CPU work.
    assert len(adopted)==6
    assert policy.usage(result)['cpu_jobs']==12
    assert policy.usage(result)['gpu_jobs']==3
    assert policy.usage(result)['llm_jobs']==3
    assert policy.usage(result)['declared_cores']==42
    for name,old in before['jobs'].items():
        assert result['jobs'][name]['spec']==old['spec']
        if old['status']=='running':assert result['jobs'][name]['attempts']==old['attempts']


def test_disk_refusal_occurs_before_any_launch_or_attempt_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    spec=job('disk_blocked',cores=4);r=row(spec);before=copy.deepcopy(r)
    class Guard:
        def reserve(self,*args,**kwargs):raise queue.InsufficientSpace('reserved experiment cap')
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('must not launch'))
    with pytest.raises(queue.InsufficientSpace):queue.start(spec,r,dict(jobs={'disk_blocked':r}),Guard())
    assert r==before
    assert not (tmp_path/'logs').exists()
    assert not (tmp_path/'results_v2').exists()
