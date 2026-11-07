import copy
import datetime as dt
import json
import time
from pathlib import Path
import pytest
from exp_revision import llm_transport_runtime as t
from exp_revision.common import sha256

@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'ROOT', tmp_path)
    monkeypatch.setattr(t, 'HEALTH', tmp_path / 'health.json')
    monkeypatch.setattr(t.q, 'alive', lambda _: False)
    output = tmp_path/'output/case-000'; output.mkdir(parents=True)
    record = {'endpoint':t.ENDPOINT,'error':{'code':'transport_failure','type':'SSLError'}}
    (output/'role-0.json').write_text(json.dumps(record))
    log=tmp_path/'failed.log'
    log.write_text('Traceback (most recent call last):\nfse2027_llm_client.LLMClientError: OpenCode request failed: transport_failure\n')
    row={'status':'stuck','spec':{'max_attempts':2,'expected_config':{'component':'llm_baseline','expected_model':'gpt-5.6-luna'}},
         'attempts':[{'status':'failed','output':'output','log':'failed.log','end_time':dt.datetime.fromtimestamp(time.time()-60,dt.timezone.utc).isoformat(),'elapsed_seconds':45}]}
    health=tmp_path/'response.json'
    health.write_text(json.dumps({'endpoint':t.ENDPOINT,'error':None,'provider_response':{'model':'gpt-5.6-luna','status':'completed'},'http_attempts':[{'http_status':200}]}))
    witness={'epoch':time.time()-1,'record':'response.json','sha256':sha256(health)}
    t.HEALTH.write_text(json.dumps(witness))
    return row,witness,tmp_path


def test_original_attempt_and_spec_preserved(fixture):
    row,_,_=fixture; before=copy.deepcopy(row); state={'jobs':{'job':row}}
    assert t.recover(state)==['job']
    assert row['status']=='pending'
    assert row['spec']==before['spec'] and row['attempts']==before['attempts']
    assert t.recover(state)==[]

@pytest.mark.parametrize('problem',['stale','before_failure','future','tamper','wrong_model','http_error','incomplete'])
def test_health_gate(fixture,problem):
    row,w,p=fixture
    if problem=='stale': w['epoch']=time.time()-1201
    if problem=='before_failure': w['epoch']=time.time()-65
    if problem=='future': w['epoch']=time.time()+60
    if problem=='tamper': w['sha256']='wrong'
    if problem in ['wrong_model','http_error','incomplete']:
        r=json.loads((p/'response.json').read_text())
        if problem=='wrong_model':r['provider_response']['model']='other'
        if problem=='http_error':r['http_attempts'][0]['http_status']=403
        if problem=='incomplete':r['provider_response']['status']='incomplete'
        (p/'response.json').write_text(json.dumps(r)); w['sha256']=sha256(p/'response.json')
    assert not t.healthy_after(row,w)

@pytest.mark.parametrize('problem',['quality','code','auth','not_llm','live','bad_record'])
def test_fail_closed(fixture,monkeypatch,problem):
    row,_,p=fixture
    if problem in ['quality','code']:(p/'failed.log').write_text('RuntimeError: '+problem)
    if problem=='auth':(p/'output/case-000/role-0.json').write_text(json.dumps({'endpoint':t.ENDPOINT,'error':{'code':'http_error','type':'SSLError'}}))
    if problem=='not_llm':row['spec']['expected_config']['component']='official_baseline'
    if problem=='live':monkeypatch.setattr(t.q,'alive',lambda _:True)
    if problem=='bad_record':(p/'output/case-000/role-0.json').write_text('{')
    assert not t.transport_failure(row)
    assert not t.recover({'jobs':{'job':row}})


def test_exhausted_stays_failed(fixture):
    row,_,_=fixture;row['attempts']*=2;state={'jobs':{'job':row}}
    assert t.transport_failure(row) and t.paused(state)
    assert not t.recover(state) and row['status']=='stuck'


def test_paused_pool_does_not_include_gpu(fixture):
    row,_,_=fixture
    gpu={'spec':{'expected_config':{'component':'model'}}}
    assert t.paused({'jobs':{'failed':row}})
    assert t.is_llm(row) and not t.is_llm(gpu)

@pytest.mark.parametrize('failure',['transport','ordinary'])
def test_scheduler_continues_local_work_only_for_transport(fixture,monkeypatch,failure):
    import signal
    from exp_revision import extended_queue_v6 as queue
    from exp_revision.tests.test_parallel_runtime import job, row, hardware
    failed,_,root=fixture
    if failure=='ordinary':(root/'failed.log').write_text('RuntimeError: ranking quality gate')
    t.HEALTH.unlink()  # provider has not recovered
    failed['spec'].update(resource='cpu',cores=1,timeout_seconds=3600,id='failed')
    manifests=root/'manifests';manifests.mkdir()
    folder=root/'queue';folder.mkdir()
    state={'status':'stuck','manifests':[],'jobs':{'failed':failed,
        'local_gpu':row(job('local_gpu','gpu',component='Eadro')),
        'pending_llm':row(job('pending_llm',component='llm_baseline'))}}
    before=copy.deepcopy(failed)
    (folder/'queue_state.json').write_text(json.dumps(state))
    monkeypatch.setattr(queue,'ROOT',root)
    monkeypatch.setattr(queue.q,'FOLDER',folder)
    monkeypatch.setattr(queue.q,'MANIFESTS',manifests)
    monkeypatch.setattr(queue.q,'load_manifests',lambda *_:None)
    monkeypatch.setattr(queue.compute,'apply',lambda:{})
    monkeypatch.setattr(queue.parallel,'snapshot',hardware)
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *_args,**_kwargs:'80000\n')
    # Root is a fixture, but policy provenance paths are real files.
    monkeypatch.setattr(queue,'sha256',lambda _: 'fixture-sha')
    class Guard:
        def __init__(self,*_):pass
        def poll(self):return {'safe_to_launch':True}
    monkeypatch.setattr(queue,'DiskGuard',Guard)
    started=[]
    def start(spec,r,*_):
        started.append(spec['id']);r['status']='running';r['attempts'].append({'status':'running'})
    monkeypatch.setattr(queue,'start',start)
    handlers={}
    monkeypatch.setattr(queue.signal,'signal',lambda s,f:handlers.update({s:f}))
    monkeypatch.setattr(queue.time,'sleep',lambda _:handlers[signal.SIGTERM]())
    result=queue.run()
    assert started==(['local_gpu'] if failure=='transport' else [])
    assert result['jobs']['failed']==before
    assert result['jobs']['pending_llm']['status']=='pending'


def test_twenty_minute_tick_not_skipped_by_probe_duration(fixture,monkeypatch):
    import sys
    from types import SimpleNamespace
    from exp_revision import disk_guard_fast
    row,witness,root=fixture
    folder=root/'queue';folder.mkdir()
    (folder/'queue_state.json').write_text(json.dumps({'jobs':{'failed':row}}))
    (root/'docs').mkdir();(root/'docs/llm_external_authorization.json').write_text('{"approved":true}')
    now=time.time()
    witness.update(started_epoch=now-1200,epoch=now-1170)
    t.HEALTH.write_text(json.dumps(witness))
    monkeypatch.setattr(t.q,'FOLDER',folder)
    events=[]
    class Guard:
        def __init__(self,*_):pass
        def reserve(self,*_):events.append('reserved')
        def release(self,*_):events.append('released')
    monkeypatch.setattr(disk_guard_fast,'DiskGuard',Guard)
    def request(*args,**kwargs):
        assert kwargs['retries']==0 and kwargs['max_tokens']==128
        events.append('probed')
        return json.loads((root/'response.json').read_text())
    monkeypatch.setitem(sys.modules,'fse2027_llm_client',SimpleNamespace(request_chat=request,LLMClientError=RuntimeError))
    t.probe_if_needed()
    assert events==['reserved','probed','released']
    t.probe_if_needed()  # same tick must not send again
    assert len(events)==3
