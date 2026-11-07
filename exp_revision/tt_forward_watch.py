"""Twenty-minute local-only recovery and collection for chronological TT."""
import fcntl
import json
import os
import subprocess
import sys
import time
from .common import ROOT, json_dump
from .tt_forward_queue import FOLDER


def check():
    FOLDER.mkdir(parents=True, exist_ok=True)
    with (FOLDER/'.monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'monitor_busy'}
        path = FOLDER/'queue_state.json'
        state = json.loads(path.read_text()) if path.exists() else {'status': 'new', 'stage': 'pilot', 'counts': {}}
        record = dict(epoch=time.time(), stage=state['stage'], status=state['status'], counts=state.get('counts', {}))
        with (FOLDER/'monitor_history.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if (FOLDER/'final_evaluation.json').exists():
            record['action'] = 'finished'
        else:
            with (FOLDER/'.queue.lock').open('a') as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return {**record, 'action': 'queue_busy'}
            if state['status'] in ('stuck', 'budget_limited'):
                record['action'] = 'needs_diagnosis'
            else:
                module = 'exp_revision.tt_forward_collect' if state['status'] == 'completed' else 'exp_revision.tt_forward_queue'
                with (ROOT/'logs/tt_forward_resumer.log').open('a') as stream:
                    child = subprocess.Popen([sys.executable, '-m', module], cwd=ROOT,
                        env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record.update(action='collect' if state['status'] == 'completed' else 'resume', pid=child.pid)
        json_dump(FOLDER/'latest_monitor.json', record)
        return record


if __name__ == '__main__':
    print(json.dumps(check()))
