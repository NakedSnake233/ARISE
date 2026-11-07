"""Freeze and run one isolated OpenCode Go campaign, without using old responses."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import uuid
from pathlib import Path

import numpy as np

import fse2027_baseline_common as common
import fse2027_llm_baselines as runner
import fse2027_llm_client as client

ROOT = common.ROOT
CAMPAIGNS = ROOT / 'artifacts/care/fse2027-opencode'
METHODS = ('ReAct', 'mABC', 'CARE+GPT')
DATASETS = ('trainticket', 'aiops2022')


def cache_call(path: Path, messages: list[dict[str, str]], max_tokens: int) -> dict:
    expected = client.request_body(messages, max_tokens)
    if path.is_file():
        record = json.loads(path.read_text())
        if record.get('request') != expected:
            raise RuntimeError('Cached OpenCode request does not match frozen request')
        for field in ('request', 'response', 'provider_response'):
            if record.get(field) is not None and record.get(field + '_sha256') != client.sha256_json(record[field]):
                raise RuntimeError('OpenCode cache hash mismatch: ' + field)
        if record.get('error') or record.get('response') is None:
            client.INFRASTRUCTURE_STOP.set()
        return record
    try:
        session = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.parent.relative_to(ROOT))))
        record = client.request_chat(messages, max_tokens=max_tokens, session_id=session)
    except client.LLMClientError as error:
        record = error.record
        client.INFRASTRUCTURE_STOP.set()
    if record.get('response') and record['response'].get('model') != runner.RETURNED_MODEL:
        record['error'] = {'type': 'model_identity_mismatch', 'code': 'provider_model_changed'}
        client.INFRASTRUCTURE_STOP.set()
    common.atomic_json(path, record)
    return record


def activate_campaign(directory: Path, manifest: dict) -> None:
    directory = directory.resolve()
    if directory == common.ROOT / 'artifacts/care/fse2027-baselines':
        raise ValueError('Cannot write the historical API campaign')
    common.OUTPUT = directory
    runner.OUTPUT = directory
    runner.RETURNED_MODEL = manifest['returned_model']
    runner.call_cached = cache_call


def freeze_campaign(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / 'API_CAMPAIGN_FROZEN.json'
    sources = [Path(runner.__file__), Path(client.__file__), Path(common.__file__), Path(__file__),
               ROOT / 'scripts/care/fse2027_llm_prompts.json',
               ROOT / 'scripts/care/fse2027_go_pipeline.py',
               ROOT / 'scripts/care/aggregate_fse2027_results.py',
               ROOT / 'scripts/care/fse2027_llm_accounting.py',
               ROOT / 'scripts/care/fse2027_llm_replay.py',
               ROOT / 'scripts/care/generate_fse2027_tables.py']
    hashes = {str(p.relative_to(ROOT)): common.sha256(p) for p in sources}
    if frozen.exists():
        manifest = json.loads(frozen.read_text())
        if manifest['software_sha256'] != hashes:
            raise RuntimeError('Frozen campaign software changed; do not mix implementations')
        for name, digest in manifest['data_sha256'].items():
            if common.sha256(ROOT / name) != digest:
                raise RuntimeError('Frozen campaign input changed')
        if manifest['request_options'] != {k: v for k, v in client.request_body([], 128).items() if k != 'input'}:
            raise RuntimeError('Frozen decoding options changed')
        return manifest
    runner.verify_protocols()
    client.load_connection()  # Missing key fails before any study request.
    smoke = directory / 'protocol/opencode-smoke.json'
    client.smoke(smoke)
    record = json.loads(smoke.read_text())
    model = record['response']['model']
    if not re.fullmatch(r'gpt-5\.6-luna(?:-\d{4}-\d{2}-\d{2})?', str(model)):
        raise RuntimeError('Unexpected returned model; no test requests accepted')
    parsed = runner.extract_json(record['response']['choices'][0]['message']['content'])
    if parsed != {'ok': True}:
        raise RuntimeError('Compatibility smoke did not produce complete valid JSON')
    data_hashes = {}
    for name in DATASETS:
        bundle = common.load_dataset(name)
        for file in ('features.npz', 'metadata.json', 'incidents.json'):
            p = bundle.path / file
            data_hashes[str(p.relative_to(ROOT))] = common.sha256(p)
        for seed in range(42, 47):
            p = ROOT / f'artifacts/care/review-response-v1/{name}/seed-{seed}/predictions.npz'
            data_hashes[str(p.relative_to(ROOT))] = common.sha256(p)
    manifest = {
        'frozen_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'campaign': directory.name, 'provider': 'opencode-go',
        'endpoint': client.BASE_URL + '/responses', 'requested_model': client.MODEL,
        'returned_model': model, 'datasets': list(DATASETS), 'methods': list(METHODS),
        'runs': [0, 1, 2], 'concurrency': 8, 'timeout_seconds': 60,
        'retryable_errors': {'retries_after_initial': 3, 'backoff_seconds': [1, 2, 4]},
        'terminal_request_error': 'after configured retries, halt entire campaign; unsubmitted calls have zero HTTP attempts',
        'request_seed': {'requested': 2027, 'sent': False, 'reason': 'Responses has no seed field'},
        'request_options': {k: v for k, v in record['request'].items() if k != 'input'},
        'per_call_output_limits': 'unchanged limits from frozen method prompts; mapped to max_output_tokens',
        'parent_protocol_sha256': runner.PROTOCOL_SHA,
        'parent_execution_sha256': runner.EXECUTION_SHA,
        'parent_clarification_sha256': runner.CLARIFICATION_SHA,
        'prompt_sha256': common.sha256(runner.PROMPTS),
        'software_sha256': hashes, 'data_sha256': data_hashes,
        'historical_campaign': 'retained separately; no responses reused',
        'scientific_protocol': 'unchanged telemetry, prompts, tools, score parser, fallback, gate and service universe',
        'selection_rule': 'one fresh complete campaign fixed by provider correction; no best-response selection',
        'raw_response': 'provider_response retains exact Responses JSON; response is a normalized parser view',
        'routing': {'user_agent': client.USER_AGENT, 'session_rule': 'stable UUID per campaign/method/dataset/run/incident; x-opencode-session header'},
        'smoke_sha256': common.sha256(smoke),
    }
    common.atomic_json(frozen, manifest)
    frozen.chmod(0o444)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', default='opencode-go-20260926')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.campaign):
        parser.error('campaign must be a simple identifier')
    directory = CAMPAIGNS / args.campaign
    status_file = directory / 'status.json'
    if status_file.is_file() and json.loads(status_file.read_text()).get('status') == 'blocked_infrastructure':
        raise SystemExit('This interrupted campaign is retained unchanged; use a new campaign identifier after the infrastructure issue is resolved')
    manifest = freeze_campaign(directory)
    if args.check_only:
        print(json.dumps({'status': 'preflight_pass', 'campaign': args.campaign,
                          'returned_model': manifest['returned_model']}))
        return
    activate_campaign(directory, manifest)
    protocol, prompts = runner.verify_protocols()
    status = {'campaign': args.campaign, 'status': 'running', 'completed_run_sets': []}
    common.atomic_json(directory / 'status.json', status)
    for dataset in DATASETS:
        bundle = common.load_dataset(dataset)
        for method in METHODS:
            for run in range(3):
                result, _ = runner.run_api_method(bundle, method, run, prompts, protocol)
                if client.INFRASTRUCTURE_STOP.is_set():
                    status.update(status='blocked_infrastructure', halted_at=[dataset, method, run])
                    common.atomic_json(directory / 'status.json', status)
                    raise SystemExit('OpenCode request failure stopped the campaign; affected run is incomplete')
                status['completed_run_sets'].append([dataset, method, run])
                common.atomic_json(directory / 'status.json', status)
                print(json.dumps({'dataset': dataset, 'method': method, 'run': run,
                                  'metrics': result['metrics'], 'cost': result['provenance']['cost']},
                                 ensure_ascii=False), flush=True)
            runner.run_method(bundle, method, prompts, protocol, [0, 1, 2])
    status['status'] = 'complete'
    common.atomic_json(directory / 'status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
