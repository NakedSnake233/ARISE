import json
import time
from types import SimpleNamespace
from exp_revision import remaining_queue as q


def test_admission_reserves_full_deadlines_and_counts_core_hours():
    state={"jobs":{
        "cpu":{"spec":{"resource":"cpu","cores":4,"timeout_seconds":890},
               "attempts":[{"elapsed_seconds":900},{"status":"running"}]},
        "gpu":{"spec":{"resource":"gpu","cores":4,"timeout_seconds":3590},
               "attempts":[{"elapsed_seconds":1800},{"status":"running"}]}}}
    assert q.charged(state,"cpu")==41.
    assert q.charged(state,"cpu",True)==42.
    assert q.charged(state,"gpu")==8.5
    assert q.charged(state,"gpu",True)==9.5


def test_launch_and_output_errors_are_stuck_but_sigterm_retries(tmp_path,monkeypatch):
    monkeypatch.setattr(q,"ROOT",tmp_path)
    monkeypatch.setattr(q,"verified",lambda *_:False)
    released=[]
    guard=SimpleNamespace(release=released.append)
    job={"timeout_seconds":100,"max_attempts":2}
    for code,expected in [(2,"stuck"),(0,"stuck"),(143,"pending")]:
        log=tmp_path/f"{code}.log";log.write_text("error: unrecognized arguments\n" if code==2 else "")
        attempt={"key":str(code),"output":"no-output","log":log.name,"start_epoch":time.time()}
        row={"status":"running","attempts":[attempt]}
        q.finish(job,row,attempt,SimpleNamespace(poll=lambda:code),guard)
        assert row["status"]==expected
        assert 0<=attempt["elapsed_seconds"]<=110
    assert released==["2","0","143"]


def test_unknown_dead_adopted_child_is_bounded_and_does_not_delete_old_files(tmp_path,monkeypatch):
    monkeypatch.setattr(q,"ROOT",tmp_path)
    monkeypatch.setattr(q,"verified",lambda *_:False)
    (tmp_path/"old.log").write_text("interrupted before output\n")
    attempt={"key":"old","output":"old-output","log":"old.log","start_epoch":time.time()-10000}
    row={"status":"running","attempts":[attempt]}
    q.finish({"timeout_seconds":100,"max_attempts":2},row,attempt,None,SimpleNamespace(release=lambda *_:None))
    assert row["status"]=="pending"
    assert attempt["elapsed_seconds"]==110
    assert (tmp_path/"old.log").read_text()=="interrupted before output\n"
