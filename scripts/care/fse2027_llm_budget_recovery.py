"""Run the separately frozen output-budget compatibility campaign.

This amendment changes only maximum output-token ceilings. A provider response
whose status is not ``completed`` is a terminal campaign error, so truncated
generations can never enter an aggregate table through the anomaly fallback.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import uuid
from pathlib import Path

import fse2027_baseline_common as common
import fse2027_llm_baselines as runner
import fse2027_llm_client as client

ROOT = common.ROOT
CAMPAIGNS = ROOT / "artifacts/care/fse2027-opencode"
PROMPTS = ROOT / "scripts/care/fse2027_llm_prompts_budget_v2.json"
PARENT_PROMPTS = ROOT / "scripts/care/fse2027_llm_prompts.json"
PARENT_CAMPAIGN = ROOT / "artifacts/care/fse2027-opencode/opencode-go-20260927"
METHODS = ("ReAct", "mABC", "CARE+GPT")
DATASETS = ("trainticket", "aiops2022")


def validate_prompt_amendment() -> None:
    parent = json.loads(PARENT_PROMPTS.read_text())
    amended = json.loads(PROMPTS.read_text())
    amended.pop("amendment", None)
    parent["protocol_version"] = amended["protocol_version"]
    fields = (
        ("react", "maximum_output_tokens"),
        ("mabc", "maximum_role_output_tokens"),
        ("mabc", "maximum_aggregation_output_tokens"),
        ("care_proposal", "maximum_output_tokens"),
    )
    for section, field in fields:
        parent[section][field] = amended[section][field]
    if parent != amended:
        raise RuntimeError("Budget-v2 prompts changed content beyond declared token ceilings")


def _validate_record(path: Path, record: dict, expected: dict) -> dict:
    if record.get("request") != expected:
        client.INFRASTRUCTURE_STOP.set()
        raise RuntimeError("Cached OpenCode request does not match frozen budget-v2 request")
    for field in ("request", "response", "provider_response"):
        value = record.get(field)
        if value is not None and record.get(field + "_sha256") != client.sha256_json(value):
            client.INFRASTRUCTURE_STOP.set()
            raise RuntimeError("OpenCode cache hash mismatch: " + field)
    raw = record.get("provider_response") or {}
    if record.get("error") or record.get("response") is None or raw.get("status") != "completed":
        if not record.get("error"):
            record["error"] = {
                "type": "provider_generation_incomplete",
                "code": str(raw.get("incomplete_details", {}).get("reason") or raw.get("status") or "missing"),
            }
            common.atomic_json(path, record)
        client.INFRASTRUCTURE_STOP.set()
    return record


def cache_call(path: Path, messages: list[dict[str, str]], max_tokens: int) -> dict:
    expected = client.request_body(messages, max_tokens)
    if path.is_file():
        return _validate_record(path, json.loads(path.read_text()), expected)
    try:
        session = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.parent.relative_to(ROOT))))
        record = client.request_chat(messages, max_tokens=max_tokens, session_id=session)
    except client.LLMClientError as error:
        record = error.record
        client.INFRASTRUCTURE_STOP.set()
    if record.get("response") and record["response"].get("model") != runner.RETURNED_MODEL:
        record["error"] = {"type": "model_identity_mismatch", "code": "provider_model_changed"}
        client.INFRASTRUCTURE_STOP.set()
    common.atomic_json(path, record)
    return _validate_record(path, record, expected)


def activate_campaign(directory: Path, manifest: dict) -> None:
    directory = directory.resolve()
    if directory in (common.ROOT / "artifacts/care/fse2027-baselines", PARENT_CAMPAIGN):
        raise ValueError("Cannot write a historical API campaign")
    common.OUTPUT = directory
    runner.OUTPUT = directory
    runner.PROMPTS = PROMPTS
    runner.RETURNED_MODEL = manifest["returned_model"]
    runner.call_cached = cache_call


def freeze_campaign(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    validate_prompt_amendment()
    runner.PROMPTS = PROMPTS
    frozen = directory / "API_CAMPAIGN_FROZEN.json"
    sources = [
        Path(runner.__file__), Path(client.__file__), Path(common.__file__), Path(__file__), PROMPTS,
        ROOT / "scripts/care/fse2027_llm_prompts.json",
        ROOT / "scripts/care/fse2027_llm_budget_pipeline.py",
        ROOT / "scripts/care/aggregate_fse2027_results.py",
        ROOT / "scripts/care/fse2027_llm_accounting.py",
        ROOT / "scripts/care/fse2027_llm_budget_replay.py",
        ROOT / "scripts/care/fse2027_opencode_cost.py",
        ROOT / "scripts/care/generate_fse2027_tables.py",
    ]
    hashes = {str(path.relative_to(ROOT)): common.sha256(path) for path in sources}
    if frozen.exists():
        manifest = json.loads(frozen.read_text())
        if manifest["software_sha256"] != hashes:
            raise RuntimeError("Frozen budget-v2 campaign software changed")
        for name, digest in manifest["data_sha256"].items():
            if common.sha256(ROOT / name) != digest:
                raise RuntimeError("Frozen budget-v2 campaign input changed")
        return manifest
    if not (PARENT_CAMPAIGN / "API_CAMPAIGN_FROZEN.json").is_file():
        raise RuntimeError("Parent campaign manifest is missing")
    runner.verify_protocols()
    client.load_connection()
    smoke = directory / "protocol/opencode-smoke.json"
    client.smoke(smoke)
    record = json.loads(smoke.read_text())
    model = record["response"]["model"]
    if not re.fullmatch(r"gpt-5\.6-luna(?:-\d{4}-\d{2}-\d{2})?", str(model)):
        raise RuntimeError("Unexpected returned model")
    parsed = runner.extract_json(record["response"]["choices"][0]["message"]["content"])
    if parsed != {"ok": True} or (record.get("provider_response") or {}).get("status") != "completed":
        raise RuntimeError("Compatibility smoke did not produce complete valid JSON")
    data_hashes = {}
    for name in DATASETS:
        bundle = common.load_dataset(name)
        for filename in ("features.npz", "metadata.json", "incidents.json"):
            path = bundle.path / filename
            data_hashes[str(path.relative_to(ROOT))] = common.sha256(path)
        for seed in range(42, 47):
            path = ROOT / f"artifacts/care/review-response-v1/{name}/seed-{seed}/predictions.npz"
            data_hashes[str(path.relative_to(ROOT))] = common.sha256(path)
    parent_manifest = PARENT_CAMPAIGN / "API_CAMPAIGN_FROZEN.json"
    manifest = {
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "campaign": directory.name,
        "campaign_role": "output-budget compatibility amendment",
        "parent_campaign": str(PARENT_CAMPAIGN.relative_to(ROOT)),
        "parent_campaign_manifest_sha256": common.sha256(parent_manifest),
        "amendment_trigger": "provider completion status only: 1,223 of 1,989 parent responses were incomplete because max_output_tokens was reached",
        "amendment_scope": "only four maximum output-token fields increased; prompts, inputs, methods, call counts, tools, parser, fallback, gate, splits, and metrics unchanged",
        "label_independence": "token ceilings selected from provider completion status and token-limit mechanics, not ground-truth labels or ranking accuracy",
        "provider": "opencode-go", "endpoint": client.BASE_URL + "/responses",
        "requested_model": client.MODEL, "returned_model": model,
        "datasets": list(DATASETS), "methods": list(METHODS), "runs": [0, 1, 2],
        "concurrency": 8, "timeout_seconds": 60,
        "retryable_errors": {"retries_after_initial": 3, "backoff_seconds": [1, 2, 4]},
        "terminal_request_error": "any transport error after retries, non-completed provider status, model mismatch, or cache-integrity failure halts the entire campaign",
        "provider_completion_requirement": "every retained study response must have provider_response.status == completed",
        "request_seed": {"requested": 2027, "sent": False, "reason": "Responses has no seed field"},
        "request_options": {key: value for key, value in record["request"].items() if key != "input"},
        "per_call_output_limits": {
            "ReAct": 2048, "mABC-role": 2048, "mABC-aggregation": 2048, "CARE+GPT": 2048,
        },
        "parent_protocol_sha256": runner.PROTOCOL_SHA,
        "parent_execution_sha256": runner.EXECUTION_SHA,
        "parent_clarification_sha256": runner.CLARIFICATION_SHA,
        "prompt_sha256": common.sha256(PROMPTS),
        "software_sha256": hashes, "data_sha256": data_hashes,
        "selection_rule": "one complete amendment campaign; no response or accuracy based selection",
        "raw_response": "exact provider Responses JSON plus hashed normalized parser view",
        "smoke_sha256": common.sha256(smoke),
    }
    common.atomic_json(frozen, manifest)
    frozen.chmod(0o444)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="opencode-go-budget-v2-20260927")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.campaign):
        parser.error("campaign must be a simple identifier")
    directory = CAMPAIGNS / args.campaign
    status_file = directory / "status.json"
    if status_file.is_file() and json.loads(status_file.read_text()).get("status") == "blocked_infrastructure":
        raise SystemExit("Interrupted budget-v2 campaign is immutable; use a new campaign identifier")
    manifest = freeze_campaign(directory)
    if args.check_only:
        print(json.dumps({"status": "preflight_pass", "campaign": args.campaign,
                          "returned_model": manifest["returned_model"]}))
        return
    activate_campaign(directory, manifest)
    protocol, prompts = runner.verify_protocols()
    status = {"campaign": args.campaign, "status": "running", "completed_run_sets": []}
    common.atomic_json(status_file, status)
    for dataset in DATASETS:
        bundle = common.load_dataset(dataset)
        for method in METHODS:
            for run in range(3):
                result, _ = runner.run_api_method(bundle, method, run, prompts, protocol)
                if client.INFRASTRUCTURE_STOP.is_set():
                    status.update(status="blocked_infrastructure", halted_at=[dataset, method, run])
                    common.atomic_json(status_file, status)
                    raise SystemExit("Incomplete generation or request failure stopped budget-v2 campaign")
                status["completed_run_sets"].append([dataset, method, run])
                common.atomic_json(status_file, status)
                print(json.dumps({"dataset": dataset, "method": method, "run": run,
                                  "metrics": result["metrics"], "cost": result["provenance"]["cost"]},
                                 ensure_ascii=False), flush=True)
            runner.run_method(bundle, method, prompts, protocol, [0, 1, 2])
    status["status"] = "complete"
    common.atomic_json(status_file, status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
