"""Offline transport invariants: exact evidence, accounting, and provider isolation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/care'))
import fse2027_llm_client as client


@pytest.fixture(autouse=True)
def clean_client(monkeypatch):
    client.INFRASTRUCTURE_STOP.clear()
    monkeypatch.setattr(client, 'settings', lambda: {})
    yield
    client.INFRASTRUCTURE_STOP.clear()


def test_responses_body_and_original_evidence(monkeypatch):
    credential = 'test-private-token'
    monkeypatch.setattr(client, 'load_connection', lambda: (client.BASE_URL, credential))
    raw = {'id': 'response-test', 'model': 'gpt-5.6-luna', 'status': 'completed',
           'output': [{'type': 'reasoning', 'summary': []},
                      {'type': 'message', 'role': 'assistant', 'content': [
                          {'type': 'output_text', 'text': '{"ok":true}'}]}],
           'usage': {'input_tokens': 7, 'output_tokens': 5, 'total_tokens': 12,
                     'output_tokens_details': {'reasoning_tokens': 2}}}
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(ok=True, status_code=200, json=lambda: raw)
    monkeypatch.setattr(client.requests, 'post', post)
    messages = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'U'}]
    record = client.request_chat(messages, max_tokens=512, session_id='test-conversation-id')
    assert calls[0][0] == 'https://opencode.ai/zen/go/v1/responses'
    assert calls[0][1]['headers']['x-opencode-session'] == 'test-conversation-id'
    assert calls[0][1]['headers']['User-Agent'] == client.USER_AGENT
    assert calls[0][1]['json']['input'] == messages
    assert calls[0][1]['json']['max_output_tokens'] == 512
    assert 'seed' not in calls[0][1]['json']
    assert record['seed_sent'] is False
    assert record['provider_response'] == raw
    assert record['provider_response_sha256'] == client.sha256_json(raw)
    assert record['response']['choices'][0]['message']['content'] == '{"ok":true}'
    assert record['response']['usage'] == {'prompt_tokens': 7, 'completion_tokens': 5, 'total_tokens': 12}
    assert credential not in json.dumps(record)
    assert record['attempts'] == 1


def test_quota_halts_without_fake_http_attempts_or_secret(monkeypatch):
    credential = 'test-private-token'
    monkeypatch.setattr(client, 'load_connection', lambda: (client.BASE_URL, credential))
    calls = []
    def post(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(ok=False, status_code=403,
                               json=lambda: {'error': {'code': 'insufficient_user_quota', 'message': credential}})
    monkeypatch.setattr(client.requests, 'post', post)
    with pytest.raises(client.LLMClientError) as first:
        client.request_chat([{'role': 'user', 'content': 'U'}], max_tokens=512)
    assert first.value.record['attempts'] == 1
    assert first.value.record['error']['http_status'] == 403
    assert credential not in json.dumps(first.value.record)
    with pytest.raises(client.LLMClientError) as skipped:
        client.request_chat([{'role': 'user', 'content': 'U'}], max_tokens=512)
    assert skipped.value.record['attempts'] == 0
    assert skipped.value.record['submitted'] is False
    assert len(calls) == 1


def test_legacy_endpoint_cannot_receive_credentials(monkeypatch):
    monkeypatch.setattr(client, 'settings', lambda: {'base_url': 'https://api.gpt.ge/v1'})
    monkeypatch.delenv('CARE_LLM_BASE_URL', raising=False)
    with pytest.raises(ValueError, match='legacy endpoints are disabled'):
        client.load_connection()


def test_actual_go_sampling_parameters_are_omitted(monkeypatch):
    monkeypatch.setattr(client, 'settings', lambda: {
        'temperature': None, 'top_p': None, 'seed_supported': False})
    body = client.request_body([{'role': 'user', 'content': 'U'}], 512)
    assert all(name not in body for name in ('temperature', 'top_p', 'seed'))
    assert body['model'] == 'gpt-5.6-luna'


def test_transient_retry_has_two_real_attempts(monkeypatch):
    monkeypatch.setattr(client, 'load_connection', lambda: (client.BASE_URL, 'test-private-token'))
    replies = [SimpleNamespace(ok=False, status_code=429, json=lambda: {'error': {'code': 'rate_limit'}}),
               SimpleNamespace(ok=True, status_code=200, json=lambda: {
                   'model': client.MODEL, 'status': 'completed', 'output': [],
                   'usage': {'input_tokens': 10, 'output_tokens': 1, 'total_tokens': 11}})]
    monkeypatch.setattr(client.requests, 'post', lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(client.time, 'sleep', lambda _: None)
    record = client.request_chat([{'role': 'user', 'content': 'U'}], max_tokens=512)
    assert record['attempts'] == 2
    assert [a['http_status'] for a in record['http_attempts']] == [429, 200]
    assert record['error'] is None
    assert record['response']['usage']['total_tokens'] == 11


def test_exhausted_proxy_failure_halts_campaign(monkeypatch):
    monkeypatch.setattr(client, 'load_connection', lambda: (client.BASE_URL, 'test-private-token'))
    monkeypatch.setattr(client.time, 'sleep', lambda _: None)
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise client.requests.exceptions.ProxyError('proxy unavailable')
    monkeypatch.setattr(client.requests, 'post', fail)
    with pytest.raises(client.LLMClientError) as failure:
        client.request_chat([{'role': 'user', 'content': 'U'}], max_tokens=512)
    assert failure.value.record['attempts'] == 4
    assert failure.value.record['error']['type'] == 'ProxyError'
    assert client.INFRASTRUCTURE_STOP.is_set()
    assert len(calls) == 4


def test_interrupted_last_run_is_not_a_complete_baseline(tmp_path):
    import numpy as np
    import aggregate_fse2027_results as aggregate
    for run in range(3):
        p = tmp_path / 'trainticket/ReAct' / f'api-run-{run}'
        p.mkdir(parents=True)
        np.savez(p / 'scores.npz', test_scores=np.ones((30, 41)))
    (tmp_path / 'status.json').write_text(json.dumps({
        'status': 'blocked_infrastructure',
        'completed_run_sets': [['trainticket', 'ReAct', 0], ['trainticket', 'ReAct', 1]]}))
    matrices = aggregate.load_saved_scores('trainticket', 'ReAct', ['api-run-0', 'api-run-1', 'api-run-2'], tmp_path)
    assert matrices == []


def test_complete_status_with_failed_cache_is_rejected(tmp_path):
    import numpy as np
    import aggregate_fse2027_results as aggregate
    completed = []
    for run in range(3):
        result = tmp_path / 'trainticket/ReAct' / f'api-run-{run}'
        result.mkdir(parents=True)
        np.savez(result / 'scores.npz', test_scores=np.ones((30, 41)))
        cache = tmp_path / 'llm-cache/ReAct/trainticket' / f'api-run-{run}'
        cache.mkdir(parents=True)
        record = {'response': {'model': client.MODEL}, 'error': None}
        if run == 2:
            record = {'response': None, 'error': {'type': 'ProxyError'}}
        (cache / 'call.json').write_text(json.dumps(record))
        completed.append(['trainticket', 'ReAct', run])
    (tmp_path / 'status.json').write_text(json.dumps({
        'status': 'complete', 'completed_run_sets': completed}))
    assert aggregate.load_saved_scores(
        'trainticket', 'ReAct', ['api-run-0', 'api-run-1', 'api-run-2'], tmp_path) == []


def test_prior_project_adapter_uses_real_client_and_propagates_auth_failure(monkeypatch):
    from opencode_legacy_adapter import OpenCodeGoClient
    adapter = OpenCodeGoClient({'model': client.MODEL, 'base_url': client.BASE_URL})
    assert not hasattr(adapter, 'chat')  # Old runtime must not insert gpt-4 defaults.
    received = []
    def auth_failure(messages, *, max_tokens, session_id):
        received.append(messages)
        raise client.LLMClientError({'error': {'code': 'http_401'}})
    monkeypatch.setattr(client, 'request_chat', auth_failure)
    with pytest.raises(client.LLMClientError):
        adapter.generate('real prompt')
    assert received == [[{'role': 'user', 'content': 'real prompt'}]]


def test_cached_auth_error_cannot_be_counted_as_completed_after_crash(tmp_path):
    import fse2027_llm_recovery as recovery
    messages = [{'role': 'user', 'content': 'U'}]
    request = client.request_body(messages, 512)
    cache = tmp_path / 'cached.json'
    cache.write_text(json.dumps({
        'request': request, 'request_sha256': client.sha256_json(request),
        'error': {'http_status': 401, 'code': 'http_401'},
        'response': None, 'provider_response': None}))
    record = recovery.cache_call(cache, messages, 512)
    assert record['error']['http_status'] == 401
    assert client.INFRASTRUCTURE_STOP.is_set()


def test_budget_v2_diff_is_limited_to_output_ceilings():
    import fse2027_llm_budget_recovery as budget
    budget.validate_prompt_amendment()


def test_budget_v2_rejects_provider_truncation(tmp_path):
    import fse2027_llm_budget_recovery as budget
    messages = [{'role': 'user', 'content': 'U'}]
    request = client.request_body(messages, 2048)
    raw = {
        'model': client.MODEL,
        'status': 'incomplete',
        'incomplete_details': {'reason': 'max_output_tokens'},
        'output': [],
        'usage': {'input_tokens': 10, 'output_tokens': 2048, 'total_tokens': 2058},
    }
    response = client.parser_view(raw)
    cache = tmp_path / 'cached.json'
    cache.write_text(json.dumps({
        'request': request, 'request_sha256': client.sha256_json(request),
        'response': response, 'response_sha256': client.sha256_json(response),
        'provider_response': raw, 'provider_response_sha256': client.sha256_json(raw),
        'error': None,
    }))
    record = budget.cache_call(cache, messages, 2048)
    assert record['error'] == {
        'type': 'provider_generation_incomplete', 'code': 'max_output_tokens'}
    assert client.INFRASTRUCTURE_STOP.is_set()
