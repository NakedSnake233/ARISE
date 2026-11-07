"""Freeze the remaining LLM serialization, parsing, and acceptance details."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from fse2027_baseline_common import OUTPUT, ROOT, sha256


PARENT = OUTPUT / "BASELINE_PROTOCOL_FROZEN.json"
PROMPTS = ROOT / "scripts/care/fse2027_llm_prompts.json"
TARGET = OUTPUT / "LLM_EXECUTION_PROTOCOL_FROZEN.json"
HASH = OUTPUT / "LLM_EXECUTION_PROTOCOL_FROZEN.sha256"
EXPECTED_PARENT = "81d0a737e261cddfda665d4a59fb81d567e371e846893dee30f58631e3d676da"


def main() -> None:
    if TARGET.exists() or HASH.exists():
        raise SystemExit("refusing to overwrite frozen LLM execution protocol")
    if sha256(PARENT) != EXPECTED_PARENT:
        raise SystemExit("parent protocol hash mismatch")
    payload = {
        "protocol_version": 1,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parent_protocol_sha256": EXPECTED_PARENT,
        "prompt_sha256": sha256(PROMPTS),
        "test_output_status_at_freeze": "no ReAct, mABC, or CARE+GPT test request sent",
        "serialization": {
            "format": "canonical compact JSON sorted by key",
            "numeric_decimal_places": 4,
            "services": "all services in metadata order",
            "per_service_fields": [
                "service", "metric_magnitude", "metric_persistence", "metric_onset_bin",
                "log_magnitude", "log_persistence", "log_onset_bin", "trace_magnitude",
                "trace_persistence", "trace_onset_bin", "history_similarity", "fallback_rank",
            ],
            "channel_content": "numeric aggregates only; no raw logs or log-template text",
            "dependency_edges": "observed directed edges ordered by descending count then source and target; retain first 120 and state total/retained counts",
            "history": "for each candidate, cosine similarity to the fit-only prototype of incidents whose root label is that candidate; missing prototypes score zero",
            "prohibited": ["test root label", "test fault type", "incident id", "dataset name", "result-derived hint"],
        },
        "normalization": {
            "magnitude": "frozen robust-deviation anomaly_score followed by per-incident min-max scaling",
            "persistence": "fraction of post-boundary service-time-channel values with robust absolute deviation above 2",
            "onset": "first post-boundary bin with any robust absolute deviation above 2; null if absent",
            "history_vector": "candidate's row-normalized metric/log/trace magnitudes concatenated in that order",
            "fallback": "mean of row-min-max metric/log/trace magnitude; stable metadata-order ties",
        },
        "react_tools": {
            "inspect_service": "return the named service record",
            "inspect_neighbors": "return retained incoming/outgoing dependency edges and endpoint service records",
            "compare_history": "return all candidates ordered by history_similarity",
            "finish": "parse ranking",
            "invalid_action": "return parser error as TOOL_RESULT while calls remain; fallback after budget exhaustion",
        },
        "parser": {
            "json": "strip Markdown fences, locate the first left brace, and decode the first complete JSON object",
            "ranking": "accept known service names once in listed order; ignore duplicates and unknown names; append unused services in frozen fallback order",
            "valid": "at least one known service is present and required keys have the declared JSON types",
            "failure": "use the complete frozen fallback ranking",
        },
        "care_gate": {
            "candidate_set": "top five of mean CARE final scores across seeds 42--46",
            "evidence_ids": "M:<service>, L:<service>, T:<service>, H:<service>, or E:<source>-><target>",
            "promotion_requirement": "the proposed top differs from the base top, belongs to the base top five, and cites an exact retrieved evidence id naming that service",
            "evidence_strength": "0.40 metric magnitude + 0.25 log magnitude + 0.25 trace magnitude + 0.10 history similarity",
            "margin": "promoted candidate strength minus base-top strength must be at least 0.05",
            "propagation_requirement": "a retained outgoing observed edge connects the promoted candidate to another anomalous base candidate; promoted onset is no later than endpoint onset plus one bin",
            "base_confidence_requirement": "mean CARE top score below 0.8",
            "accepted_action": "move only the promoted service to rank one and preserve relative order of every other base-ranked service",
            "failure": "preserve the complete base ranking",
        },
        "execution": {
            "model": "gpt-5.6-luna",
            "temperature": 0,
            "top_p": 1,
            "seed": 2027,
            "runs": [0, 1, 2],
            "maximum_concurrency": 8,
            "timeout_seconds": 60,
            "retries_after_initial": 3,
            "backoff_seconds": [1, 2, 4],
            "cache": "per-method/per-dataset/per-run/per-call crash recovery only",
            "cross_run_cache": False,
            "best_response_selection": False,
            "exact_sanitized_request_and_response_retained": True,
        },
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    digest = sha256(TARGET)
    HASH.write_text(f"{digest}  {TARGET.name}\n")
    os.chmod(TARGET, 0o444)
    os.chmod(HASH, 0o444)
    print(json.dumps({"status": "frozen", "artifact": str(TARGET.relative_to(ROOT)), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
