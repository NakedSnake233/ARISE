"""Close two scalar edge cases before the first test-set LLM request."""

from __future__ import annotations

import datetime as dt
import json
import os

from fse2027_baseline_common import OUTPUT, ROOT, sha256


PARENT = OUTPUT / "LLM_EXECUTION_PROTOCOL_FROZEN.json"
TARGET = OUTPUT / "LLM_EXECUTION_PROTOCOL_CLARIFICATION_FROZEN.json"
HASH = OUTPUT / "LLM_EXECUTION_PROTOCOL_CLARIFICATION_FROZEN.sha256"
EXPECTED_PARENT = "6b088951dd075be0c4e0e254d63705e8fd62723764236b44c37e3322cddaf7d9"


def main() -> None:
    if TARGET.exists() or HASH.exists():
        raise SystemExit("refusing to overwrite frozen clarification")
    if sha256(PARENT) != EXPECTED_PARENT:
        raise SystemExit("parent execution protocol hash mismatch")
    payload = {
        "protocol_version": 1,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parent_execution_protocol_sha256": EXPECTED_PARENT,
        "test_output_status_at_freeze": "no ReAct, mABC, or CARE+GPT test request sent",
        "history_similarity_edge_case": "cosine similarity is mapped from [-1,1] to [0,1]; a zero-norm current vector or prototype receives 0",
        "care_anomalous_candidate": "evidence_strength at least 0.50",
        "care_candidate_onset": "minimum non-null metric/log/trace onset bin; missing all three onsets fails the propagation gate",
        "care_edge_scope": "the observed outgoing edge may point to any other base-top-five candidate that meets the anomalous-candidate rule",
        "response_model_validation": "the returned model identifier must equal the smoke-test identifier gpt-5.6-luna-2026-07-09; otherwise the call is invalid and follows the frozen fallback",
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    digest = sha256(TARGET)
    HASH.write_text(f"{digest}  {TARGET.name}\n")
    os.chmod(TARGET, 0o444)
    os.chmod(HASH, 0o444)
    print(json.dumps({"status": "frozen", "artifact": str(TARGET.relative_to(ROOT)), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
