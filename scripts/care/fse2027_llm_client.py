"""OpenCode Go Responses transport for all current FSE LLM treatments.

Credentials are read at request time and never serialized. Exact Responses
bodies are retained alongside a clearly identified parser-compatible view.
The historical Chat Completions campaign remains a separate immutable record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get('CARE_LLM_CONFIG', str(ROOT / 'configs/agents.yaml')))
MODEL = 'gpt-5.6-luna'
BASE_URL = 'https://opencode.ai/zen/go/v1'
USER_AGENT = 'care-research-agent/1.0'
INFRASTRUCTURE_STOP = threading.Event()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


class LLMClientError(RuntimeError):
    """Carries a redacted, precisely counted failed or unsubmitted request."""
    def __init__(self, record: dict[str, Any]):
        self.record = record
        error = record.get('error') or {}
        super().__init__('OpenCode request failed: ' + str(error.get('code') or error.get('type')))


def settings() -> dict[str, Any]:
    if not CONFIG.is_file():
        return {}
    return (yaml.safe_load(CONFIG.read_text()) or {}).get('llm', {})


def load_connection() -> tuple[str, str]:
    config = settings()
    base_url = os.environ.get('CARE_LLM_BASE_URL') or config.get('base_url') or BASE_URL
    base_url = str(base_url).rstrip('/')
    u = urlsplit(base_url)
    if (u.scheme, u.hostname, u.path) != ('https', 'opencode.ai', '/zen/go/v1') or u.username or u.password or u.query or u.fragment:
        raise ValueError('Only the official OpenCode Go base URL is enabled; legacy endpoints are disabled')
    if config.get('model', MODEL) != MODEL:
        raise ValueError('The evaluated model must be gpt-5.6-luna')
    credential = os.environ.get('OPENCODE_API_KEY')
    if not credential:
        env_name = str(config.get('api_key_env') or 'OPENCODE_API_KEY')
        credential = os.environ.get(env_name) or config.get('api_key')
    if not credential:
        key_file = Path(str(config.get('api_key_file') or 'configs/opencode.key')).expanduser()
        if not key_file.is_absolute():
            key_file = ROOT / key_file
        if key_file.is_file():
            credential = key_file.read_text().strip()
    if not credential:
        raise ValueError('OpenCode API key not found; configure OPENCODE_API_KEY or configs/opencode.key')
    credential = str(credential).strip()
    if any(c.isspace() for c in credential):
        raise ValueError('OpenCode API key must be a single token')
    return base_url, credential


def request_body(messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    config = settings()
    if config.get('seed_supported'):
        raise ValueError('Responses transport does not support a request seed')
    body: dict[str, Any] = {
        'model': MODEL,
        'input': [{'role': m['role'], 'content': m['content']} for m in messages],
        'max_output_tokens': max_tokens,
        'store': False,
    }
    for field, default in [('temperature', 0), ('top_p', 1)]:
        value = config.get(field, default)
        if value is not None:
            body[field] = value
    return body


def parser_view(raw: dict[str, Any]) -> dict[str, Any]:
    texts = []
    for item in raw.get('output') or []:
        if item.get('type') == 'message' and item.get('role') == 'assistant':
            for part in item.get('content') or []:
                if part.get('type') == 'output_text':
                    texts.append(str(part.get('text') or ''))
    usage = raw.get('usage') or {}
    return {
        'id': raw.get('id'), 'model': raw.get('model'),
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''.join(texts)},
                     'finish_reason': 'stop' if raw.get('status') == 'completed' else 'length'}],
        'usage': {'prompt_tokens': int(usage.get('input_tokens') or 0),
                  'completion_tokens': int(usage.get('output_tokens') or 0),
                  'total_tokens': int(usage.get('total_tokens') or 0)},
    }


def request_chat(messages: list[dict[str, str]], *, max_tokens: int,
                 timeout_seconds: float = 60.0, retries: int = 3,
                 run_seed: int = 2027, session_id: str | None = None) -> dict[str, Any]:
    base_url, credential = load_connection()
    body = request_body(messages, max_tokens)
    session_id = session_id or str(uuid.uuid4())
    record: dict[str, Any] = {
        'transport': 'opencode-go-responses', 'endpoint': base_url + '/responses',
        'request': body, 'request_sha256': sha256_json(body),
        'requested_seed': run_seed, 'seed_sent': False,
        'response': None, 'provider_response': None, 'attempts': 0,
        'submitted': False, 'http_attempts': [], 'wall_seconds': None,
        'routing': {'user_agent': USER_AGENT, 'session_id': session_id},
    }
    if INFRASTRUCTURE_STOP.is_set():
        record['error'] = {'type': 'infrastructure_skip', 'code': 'campaign_stopped'}
        raise LLMClientError(record)
    for attempt in range(retries + 1):
        if INFRASTRUCTURE_STOP.is_set():
            record['error'] = {'type': 'infrastructure_skip', 'code': 'campaign_stopped'}
            raise LLMClientError(record)
        start = time.perf_counter()
        attempt_record: dict[str, Any] = {'attempt': attempt + 1}
        record['attempts'] += 1
        record['submitted'] = True
        try:
            response = requests.post(base_url + '/responses',
                                     headers={'Authorization': 'Bearer ' + credential,
                                              'Content-Type': 'application/json',
                                              'User-Agent': USER_AGENT,
                                              'x-opencode-session': session_id},
                                     json=body, timeout=timeout_seconds)
            attempt_record.update(http_status=response.status_code, wall_seconds=time.perf_counter() - start)
            try:
                raw = response.json()
            except ValueError:
                raw = None
            if not response.ok:
                err = (raw or {}).get('error') or {}
                if isinstance(err, str):
                    err = {'message': err}
                err = json.loads(json.dumps(err, ensure_ascii=False).replace(credential, '[REDACTED]'))
                record['error'] = {'type': 'http_error', 'http_status': response.status_code,
                                   'code': err.get('code') or 'http_' + str(response.status_code),
                                   'message': err.get('message')}
                record['http_attempts'].append(attempt_record)
                # Permanent auth/quota errors halt the entire campaign, without retries.
                if response.status_code in (401, 403):
                    INFRASTRUCTURE_STOP.set()
                if response.status_code not in (429,) and response.status_code < 500:
                    INFRASTRUCTURE_STOP.set()
                    raise LLMClientError(record)
                if attempt >= retries:
                    INFRASTRUCTURE_STOP.set()
                    raise LLMClientError(record)
            elif not isinstance(raw, dict):
                raise ValueError('Responses endpoint did not return a JSON object')
            else:
                record['http_attempts'].append(attempt_record)
                record['provider_response'] = raw
                record['provider_response_sha256'] = sha256_json(raw)
                record['response'] = parser_view(raw)
                record['response_sha256'] = sha256_json(record['response'])
                record['response_is_normalized_parser_view'] = True
                record['wall_seconds'] = attempt_record['wall_seconds']
                record['error'] = None
                return record
        except LLMClientError:
            raise
        except (requests.RequestException, ValueError) as error:
            attempt_record.update(wall_seconds=time.perf_counter() - start,
                                  error_type=type(error).__name__)
            record['http_attempts'].append(attempt_record)
            record['error'] = {'type': type(error).__name__, 'code': 'transport_failure'}
            if attempt >= retries:
                INFRASTRUCTURE_STOP.set()
                raise LLMClientError(record) from None
        time.sleep(2**attempt)
    raise LLMClientError(record)


def smoke(output: Path) -> None:
    messages = [{'role': 'system', 'content': 'Return only a JSON object with key ok and value true.'},
                {'role': 'user', 'content': 'Protocol compatibility smoke test; no study data are included.'}]
    try:
        record = request_chat(messages, max_tokens=128)
    except LLMClientError as error:
        record = error.record
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    view = record.get('response') or {}
    print(json.dumps({'status': 'error' if record.get('error') else 'ok',
                      'requested_model': MODEL, 'returned_model': view.get('model'),
                      'usage': view.get('usage'), 'attempts': record['attempts'],
                      'seed_sent': record['seed_sent'], 'error': record.get('error'),
                      'artifact': str(output.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    if record.get('error'):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/care/fse2027-baselines/protocol/opencode-go-smoke.json')
    args = parser.parse_args()
    if not args.smoke:
        parser.error('select --smoke')
    smoke(args.output)


if __name__ == '__main__':
    main()
