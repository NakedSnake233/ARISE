"""Diagnose one retained failed request on the active OpenCode Go endpoint.

The result is an excluded infrastructure diagnostic; it never replaces a
retained treatment response. No historical provider receives new requests.
"""
import argparse
import json
from pathlib import Path
from fse2027_llm_client import request_chat, LLMClientError

parser = argparse.ArgumentParser()
parser.add_argument('cache', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
record = json.loads(args.cache.read_text())
assert record.get('error'), 'diagnostic requires a retained failed call'
request = record['request']
messages = request.get('input') or request['messages']
limit = int(request.get('max_output_tokens') or request['max_tokens'])
try:
    result = request_chat(messages, max_tokens=limit)
except LLMClientError as error:
    result = error.record
result['scope'] = 'excluded infrastructure diagnostic; never substituted into experiment'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'endpoint': result['endpoint'], 'error': result.get('error'),
                  'returned_model': (result.get('response') or {}).get('model'),
                  'attempts': result['attempts'], 'scope': result['scope']}, indent=2))
