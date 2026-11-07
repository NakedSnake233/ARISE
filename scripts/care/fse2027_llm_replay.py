"""Replay the retained three-run API rankings without making network requests.

Validate cache hashes and previously saved metrics before running the frozen
parser, fallback, and CARE gate again. This command intentionally refuses an
incomplete cache rather than filling missing requests from a live endpoint.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

import fse2027_llm_baselines as runner
import fse2027_llm_client as client
from fse2027_llm_recovery import activate_campaign
from fse2027_llm_client import sha256_json
from fse2027_baseline_common import OUTPUT, load_dataset


def forbid_request(*args, **kwargs):
    raise RuntimeError("offline replay attempted an uncached API request")


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["trainticket", "aiops2022"], default=["trainticket", "aiops2022"])
    parser.add_argument("--methods", nargs="+", choices=["ReAct", "mABC", "CARE+GPT"], default=["ReAct", "mABC", "CARE+GPT"])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    manifest_file = OUTPUT / 'API_CAMPAIGN_FROZEN.json'
    if manifest_file.is_file():
        activate_campaign(OUTPUT, json.loads(manifest_file.read_text()))
        status = json.loads((OUTPUT / 'status.json').read_text())
        if status.get('status') != 'complete':
            raise RuntimeError('Only a complete API campaign can be replayed')
        completed = {tuple(x) for x in status.get('completed_run_sets', [])}
        for dataset in args.datasets:
            for method in args.methods:
                if any((dataset, method, run) not in completed for run in range(3)):
                    raise RuntimeError('Incomplete or infrastructure-interrupted treatment cannot be replayed as complete')
                cache_files = list((OUTPUT / 'llm-cache' / method / dataset).rglob('*.json'))
                if not cache_files:
                    raise RuntimeError('Complete campaign is missing API cache records')
                for cache_file in cache_files:
                    cache_record = json.loads(cache_file.read_text())
                    if cache_record.get('error') or cache_record.get('response') is None:
                        raise RuntimeError(f'failed API cache cannot be replayed as complete: {cache_file}')
    protocol, prompts = runner.verify_protocols()
    expected = {}
    for dataset in args.datasets:
        for method in args.methods:
            for run in range(3):
                directory = OUTPUT / dataset / method / f"api-run-{run}"
                audit_path = directory / "llm_audit.json"
                scores_path = directory / "scores.npz"
                if not audit_path.is_file() or not scores_path.is_file():
                    raise RuntimeError(f"incomplete replay inputs: {directory}")
                expected[(dataset, method, run)] = np.load(scores_path)["test_scores"].copy()
                for audit in json.loads(audit_path.read_text()):
                    for relative in audit["call_paths"]:
                        path = runner.ROOT / relative
                        record = json.loads(path.read_text())
                        if record.get('error') or record.get('response') is None:
                            raise RuntimeError(f"failed API cache cannot be replayed as complete: {path}")
                        for field in ("request", "response", "provider_response"):
                            if record.get(field) is None:
                                continue
                            digest = record.get(field + "_sha256")
                            if digest is not None and digest != sha256_json(record[field]):
                                raise RuntimeError(f"cache {field} hash mismatch: {path}")
    runner.request_chat = forbid_request
    client.request_chat = forbid_request
    for dataset in args.datasets:
        bundle = load_dataset(dataset)
        for method in args.methods:
            runner.run_method(bundle, method, prompts, protocol, [0, 1, 2])
            for run in range(3):
                scores = np.load(OUTPUT / dataset / method / f"api-run-{run}" / "scores.npz")["test_scores"]
                np.testing.assert_array_equal(scores, expected[(dataset, method, run)])
    print(json.dumps({"status": "pass", "replayed_run_sets": len(expected),
                      "network_requests": 0, "score_match": "exact"}, indent=2))


if __name__ == "__main__":
    main()
