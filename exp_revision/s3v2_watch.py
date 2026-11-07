"""Twenty-minute local progress check; resume under the queue's exclusive lock."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT,json_dump
from .s3v2_queue import FOLDER,OFFICIAL_GROUPS,official_spec


def check():
    FOLDER.mkdir(parents=True,exist_ok=True)
    with (FOLDER/'.monitor.lock').open('a') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'another_monitor_running'}
        state=json.loads((FOLDER/'queue_state.json').read_text())
        with (FOLDER/'.queue.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);busy=False
            except BlockingIOError:busy=True
        record=dict(epoch=time.time(),queue_busy=busy,queue_status=state.get('status'),counts=dict(Counter(r['status'] for r in state['jobs'].values())),cpu_hours=state.get('cpu_hours'),gpu_hours=state.get('gpu_hours'))
        with (FOLDER/'monitor_history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        if busy:return record
        if state.get('status') in ('stuck','budget_limited','storage_limited'):
            # No blind retry of scientific/code errors or changes to approved budgets.
            # Storage pressure may clear, and interrupted processes are reconciled by q.run;
            # explicit stuck errors are left for the active assistant to diagnose.
            if state.get('status')=='stuck':return {**record,'action':'needs_diagnosis; no result or failure state rewritten'}
        missing=[official_spec(ds,f,seed,m)['id'] for ds,m in OFFICIAL_GROUPS for f in range(5) for seed in range(42,47)
                 if official_spec(ds,f,seed,m)['id'] not in state['jobs']]
        unfinished=any(r['status'] not in ('completed','scientific_blocked') for r in state['jobs'].values())
        if not missing and not unfinished:
            if (FOLDER/'final_evaluation.json').exists():return {**record,'action':'matrix_finished_and_collected'}
            with (ROOT/'logs/s3v2_final_collection.log').open('a') as stream:
                process=subprocess.Popen([sys.executable,'-m','exp_revision.s3v2_report','--finalize'],cwd=ROOT,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            return {**record,'action':'final_collection_started','pid':process.pid}
        log=ROOT/'logs/s3v2_resumer.log'
        with log.open('a') as stream:
            process=subprocess.Popen([sys.executable,'-m','exp_revision.s3v2_queue','--official','--full'],cwd=ROOT,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(action='resume_or_admit_successful_remaining_official_groups',pid=process.pid,missing_jobs=len(missing))
        json_dump(FOLDER/'latest_monitor.json',record)
        return record


if __name__=='__main__':print(json.dumps(check()))
