"""Twenty-minute pair queue recovery/stage admission; never bypass approval."""
import fcntl
import json
import os
import subprocess
import sys
import time
from .common import ROOT,json_dump
from .tt_pair_data import FOLDER
from .tt_pair_queue import budget


def check():
    FOLDER.mkdir(parents=True,exist_ok=True)
    with (FOLDER/'.monitor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'monitor_busy'}
        s=json.loads((FOLDER/'queue_state.json').read_text());record=dict(epoch=time.time(),stage=s['stage'],status=s['status'],counts=s['counts'])
        with (FOLDER/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if not budget()['approved']:return {**record,'action':'awaiting_explicit_budget_authorization'}
        with (FOLDER/'.queue.lock').open('a') as probe:
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return {**record,'action':'queue_busy'}
        if s['status'] in ('stuck','forecast_budget_limited','dependency_blocked'):return {**record,'action':'needs_diagnosis'}
        if (FOLDER/'final_evaluation.json').exists():return {**record,'action':'finished'}
        stage=s['stage']
        if s['status']=='completed':stage={'smoke':'pilot','pilot':'full','full':'collect'}[stage]
        command=[sys.executable,'-m','exp_revision.tt_pair_collect'] if stage=='collect' else [sys.executable,'-m','exp_revision.tt_pair_queue','--stage',stage]
        with (ROOT/'logs/tt_pair_resumer.log').open('a') as f:
            proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(action='resume_or_next_stage',pid=proc.pid,next_stage=stage);json_dump(FOLDER/'latest_monitor.json',record);return record


if __name__=='__main__':print(json.dumps(check()))
