"""Bounded transport-only recovery; original attempts and job specs stay intact."""
from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path
import time
from .common import ROOT, json_dump, sha256
from . import remaining_queue as q

HEALTH = q.FOLDER / 'llm_transport_health.json'
TYPES = {'SSLError', 'ConnectionError', 'ConnectTimeout', 'ReadTimeout', 'Timeout'}
ENDPOINT = 'https://opencode.ai/zen/go/v1/responses'


def is_llm(row):
    return row['spec']['expected_config'].get('component') == 'llm_baseline'


def transport_failure(row):
    if row['status'] != 'stuck' or not is_llm(row) or not row['attempts']:
        return False
    a = row['attempts'][-1]
    if a.get('status') != 'failed' or q.alive(a):
        return False
    tail = q.log_tail(ROOT / a['log']).strip()
    if not tail.endswith('fse2027_llm_client.LLMClientError: OpenCode request failed: transport_failure'):
        return False
    errors = []
    for path in (ROOT / a['output']).glob('case-*/*.json'):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        if record.get('error'):
            errors.append(record)
    return bool(errors) and all(r.get('endpoint') == ENDPOINT and
        r['error'].get('code') == 'transport_failure' and r['error'].get('type') in TYPES for r in errors)


def healthy_after(row, witness, now=None):
    now = time.time() if now is None else now
    try:
        stamp = float(witness['epoch'])
        failed = dt.datetime.fromisoformat(row['attempts'][-1]['end_time']).timestamp()
        path = ROOT / witness['record']
        if not (failed < stamp <= now and now - stamp < 1200 and sha256(path) == witness['sha256']):
            return False
        r = json.loads(path.read_text()); raw = r.get('provider_response') or {}
        return (not r.get('error') and r.get('endpoint') == ENDPOINT and
            raw.get('model') == row['spec']['expected_config']['expected_model'] and
            raw.get('status') == 'completed' and r.get('http_attempts', [{}])[-1].get('http_status') == 200)
    except (KeyError, ValueError, TypeError, OSError, IndexError):
        return False


def recover(state):
    """Called only by the scheduler holding .queue.lock."""
    try: witness = json.loads(HEALTH.read_text())
    except (OSError, ValueError): return []
    recovered = []
    for name, row in state['jobs'].items():
        if (len(row['attempts']) < row['spec']['max_attempts'] and transport_failure(row)
                and healthy_after(row, witness)):
            row.setdefault('transport_recoveries', []).append(dict(time=q.now(), health=witness,
                previous_attempt=len(row['attempts']), next_attempt=len(row['attempts']) + 1,
                policy_sha256=sha256(Path(__file__)), reason='verified transport failure; healthy provider; original retry allowance'))
            row['status'] = 'pending'
            recovered.append(name)
    return recovered


def paused(state):
    return any(transport_failure(row) for row in state['jobs'].values())


def probe_if_needed():
    """20-minute checker probes; queue owner alone mutates job status."""
    state = json.loads((q.FOLDER / 'queue_state.json').read_text())
    rows = [r for r in state['jobs'].values() if transport_failure(r) and len(r['attempts']) < r['spec']['max_attempts']]
    if not rows: return
    approval = json.loads((ROOT / 'docs/llm_external_authorization.json').read_text())
    if approval.get('approved') is not True: return
    try:
        old = json.loads(HEALTH.read_text())
        # Cron jitter and request duration must not turn a 20-minute check into 40 minutes.
        if 0 <= time.time() - old.get('started_epoch', old['epoch']) < 1140: return
    except (OSError, ValueError, KeyError): pass
    from .disk_guard_fast import DiskGuard
    started_epoch = time.time()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = q.FOLDER / ('transport_health_' + stamp)
    key = 'transport_health_' + stamp
    guard = DiskGuard(ROOT)
    guard.reserve(key, output, ROOT / 'logs' / (key + '.log'))
    try:
        sys.path.insert(0, str(ROOT / 'scripts/care'))
        import fse2027_llm_client as client
        # Fresh check processes normally have an unset stop event. Do not clear it.
        try:
            record = client.request_chat([{'role':'user','content':'Connectivity check, no experiment data. Reply OK.'}],
                max_tokens=128, timeout_seconds=30, retries=0)
        except client.LLMClientError as error: record = error.record
        path = output / 'response.json'
        json_dump(path, record)
        json_dump(HEALTH, dict(epoch=time.time(), started_epoch=started_epoch, record=str(path.relative_to(ROOT)), sha256=sha256(path)))
    finally: guard.release(key)
