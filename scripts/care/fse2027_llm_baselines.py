"""Run the frozen ReAct, mABC, and optional CARE+GPT protocols.

Every request artifact contains the exact sanitized request and response but no
credential or HTTP header.  Dataset labels are used only by the common evaluator
after a complete service ranking has been produced.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fse2027_baseline_common import (
        OUTPUT, ROOT, DatasetBundle, anomaly_score, atomic_json, fallback_scores,
        load_dataset, robust_deviation, row_unit_interval, save_method_run, sha256,
        summarize_repeats,
    )
    from fse2027_llm_client import request_chat
except ImportError:
    from .fse2027_baseline_common import (
        OUTPUT, ROOT, DatasetBundle, anomaly_score, atomic_json, fallback_scores,
        load_dataset, robust_deviation, row_unit_interval, save_method_run, sha256,
        summarize_repeats,
    )
    from .fse2027_llm_client import request_chat


PROTOCOL = OUTPUT / "BASELINE_PROTOCOL_FROZEN.json"
EXECUTION = OUTPUT / "LLM_EXECUTION_PROTOCOL_FROZEN.json"
CLARIFICATION = OUTPUT / "LLM_EXECUTION_PROTOCOL_CLARIFICATION_FROZEN.json"
PROMPTS = ROOT / "scripts/care/fse2027_llm_prompts.json"
PROTOCOL_SHA = "81d0a737e261cddfda665d4a59fb81d567e371e846893dee30f58631e3d676da"
EXECUTION_SHA = "6b088951dd075be0c4e0e254d63705e8fd62723764236b44c37e3322cddaf7d9"
CLARIFICATION_SHA = "bd56398613962948a887e2a3dd0fcae06535531bd35b3b74de64c3ddd7925727"
RETURNED_MODEL = "gpt-5.6-luna-2026-07-09"


def verify_protocols() -> tuple[dict[str, Any], dict[str, Any]]:
    expected = [(PROTOCOL, PROTOCOL_SHA), (EXECUTION, EXECUTION_SHA), (CLARIFICATION, CLARIFICATION_SHA)]
    for path, digest in expected:
        if sha256(path) != digest:
            raise RuntimeError(f"frozen protocol hash mismatch: {path}")
    return json.loads(PROTOCOL.read_text()), json.loads(PROMPTS.read_text())


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rounded(value: float) -> float:
    return round(float(value), 4)


def rank_order(scores: np.ndarray) -> list[int]:
    return np.argsort(-np.asarray(scores), kind="stable").tolist()


def ranking_scores(order: list[int], count: int) -> np.ndarray:
    scores = np.zeros(count, dtype=np.float64)
    for position, service_index in enumerate(order):
        scores[service_index] = count - position
    return scores


@dataclass
class TelemetryBuilder:
    bundle: DatasetBundle

    def __post_init__(self) -> None:
        self.fallback = fallback_scores(self.bundle)
        self.magnitude: dict[str, np.ndarray] = {}
        self.persistence: dict[str, np.ndarray] = {}
        self.onset: dict[str, list[list[int | None]]] = {}
        for key, modality in (("metric", "metrics"), ("log", "logs"), ("trace", "traces")):
            self.magnitude[key] = row_unit_interval(
                anomaly_score(self.bundle.features[modality], self.bundle.split_bin)
            )
            deviation = robust_deviation(self.bundle.features[modality], self.bundle.split_bin)
            self.persistence[key] = (deviation > 2.0).mean(axis=(2, 3))
            active = (deviation > 2.0).any(axis=-1)
            onset_rows: list[list[int | None]] = []
            for incident in active:
                onset_rows.append([
                    int(np.flatnonzero(service)[0]) if service.any() else None
                    for service in incident
                ])
            self.onset[key] = onset_rows
        vectors = np.stack(
            [self.magnitude["metric"], self.magnitude["log"], self.magnitude["trace"]], axis=-1
        )
        prototypes = np.zeros((self.bundle.service_count, 3), dtype=np.float64)
        available = np.zeros(self.bundle.service_count, dtype=bool)
        for service in range(self.bundle.service_count):
            incidents = self.bundle.fit[self.bundle.labels[self.bundle.fit] == service]
            if len(incidents):
                prototypes[service] = vectors[incidents, service].mean(axis=0)
                available[service] = True
        similarity = np.zeros((len(self.bundle.rows), self.bundle.service_count), dtype=np.float64)
        for incident in range(len(self.bundle.rows)):
            for service in range(self.bundle.service_count):
                current, prototype = vectors[incident, service], prototypes[service]
                denominator = np.linalg.norm(current) * np.linalg.norm(prototype)
                if available[service] and denominator > 1e-15:
                    cosine = float(np.dot(current, prototype) / denominator)
                    similarity[incident, service] = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)
        self.history_similarity = similarity

    def service_record(self, index: int, service_index: int) -> dict[str, Any]:
        order = rank_order(self.fallback[index])
        fallback_rank = order.index(service_index) + 1
        record: dict[str, Any] = {
            "service": self.bundle.services[service_index],
            "fallback_rank": fallback_rank,
            "history_similarity": rounded(self.history_similarity[index, service_index]),
        }
        for modality in ("metric", "log", "trace"):
            record[f"{modality}_magnitude"] = rounded(self.magnitude[modality][index, service_index])
            record[f"{modality}_persistence"] = rounded(self.persistence[modality][index, service_index])
            record[f"{modality}_onset_bin"] = self.onset[modality][index][service_index]
        return record

    def retained_edges(self, index: int) -> tuple[list[dict[str, Any]], int]:
        services = set(self.bundle.services)
        edges = [
            {
                "source": str(edge["source"]),
                "target": str(edge["target"]),
                "count": int(max(float(edge.get("count", 1)), 0)),
            }
            for edge in self.bundle.rows[index].get("calls", [])
            if edge.get("source") in services and edge.get("target") in services
        ]
        edges.sort(key=lambda edge: (-edge["count"], edge["source"], edge["target"]))
        return edges[:120], len(edges)

    def observation(self, index: int) -> dict[str, Any]:
        edges, total = self.retained_edges(index)
        return {
            "candidate_services": self.bundle.services,
            "service_records": [
                self.service_record(index, service) for service in range(self.bundle.service_count)
            ],
            "dependency": {
                "orientation": "source calls target",
                "total_observed_edges": total,
                "retained_edges": len(edges),
                "edges": edges,
            },
            "fallback_ranking": [self.bundle.services[position] for position in rank_order(self.fallback[index])],
        }

    def tool_result(self, index: int, action: str, service: Any) -> dict[str, Any]:
        observation = self.observation(index)
        if action == "compare_history":
            records = sorted(
                observation["service_records"],
                key=lambda item: (-item["history_similarity"], self.bundle.services.index(item["service"])),
            )
            return {"history_order": [{"service": item["service"], "similarity": item["history_similarity"]} for item in records]}
        if not isinstance(service, str) or service not in self.bundle.services:
            return {"error": "unknown service"}
        record = observation["service_records"][self.bundle.services.index(service)]
        if action == "inspect_service":
            return record
        if action == "inspect_neighbors":
            edges = [edge for edge in observation["dependency"]["edges"] if service in (edge["source"], edge["target"])]
            endpoints = {service}
            for edge in edges:
                endpoints.update([edge["source"], edge["target"]])
            return {
                "edges": edges,
                "service_records": [item for item in observation["service_records"] if item["service"] in endpoints],
            }
        return {"error": "unknown action"}


def extract_json(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def complete_ranking(raw: Any, services: list[str], fallback: list[str]) -> tuple[list[str], bool]:
    if not isinstance(raw, list):
        return fallback.copy(), False
    ranking: list[str] = []
    for item in raw:
        service = item if isinstance(item, str) else item.get("service") if isinstance(item, dict) else None
        if service in services and service not in ranking:
            ranking.append(service)
    valid = bool(ranking)
    ranking.extend(service for service in fallback if service not in ranking)
    return ranking, valid


def response_content(record: dict[str, Any]) -> tuple[str | None, str | None]:
    if record.get("error"):
        return None, str(record["error"])
    response = record.get("response", {})
    if response.get("model") != RETURNED_MODEL:
        return None, "returned_model_mismatch"
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, "missing_response_content"
    return str(content), None


def call_cached(path: Path, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        record = request_chat(messages, max_tokens=max_tokens)
        record["error"] = None
    except Exception as error:
        record = {
            "request": {
                "model": "gpt-5.6-luna",
                "messages": messages,
                "temperature": 0,
                "top_p": 1,
                "seed": 2027,
                "max_tokens": max_tokens,
            },
            "response": None,
            "wall_seconds": None,
            "attempts": 4,
            "error": type(error).__name__,
        }
    atomic_json(path, record)
    return record


def common_rules(prompts: dict[str, Any]) -> str:
    return "\n".join(f"- {rule}" for rule in prompts["common_rules"])


def run_react_incident(
    builder: TelemetryBuilder, index: int, run_dir: Path, prompts: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    observation = builder.observation(index)
    telemetry = canonical(observation)
    fallback = observation["fallback_ranking"]
    template = prompts["react"]
    previous = None
    tool_result = None
    call_paths = []
    ranking = fallback.copy()
    valid = False
    reason = "call_budget_exhausted"
    for call_index in range(3):
        remaining = 3 - call_index
        if call_index == 0:
            user = template["first_user"].format(
                common_rules=common_rules(prompts), telemetry=telemetry, remaining_calls=remaining
            )
        else:
            user = template["followup_user"].format(
                common_rules=common_rules(prompts), telemetry=telemetry,
                previous_response=previous, tool_result=canonical(tool_result), remaining_calls=remaining,
            )
        messages = [{"role": "system", "content": template["system"]}, {"role": "user", "content": user}]
        path = run_dir / f"case-{int(np.flatnonzero(builder.bundle.test == index)[0]):03d}" / f"call-{call_index}.json"
        call_paths.append(str(path.relative_to(ROOT)))
        record = call_cached(path, messages, int(template["maximum_output_tokens"]))
        content, error = response_content(record)
        if error:
            previous = canonical({"error": error})
            tool_result = {"error": error}
            reason = error
            continue
        previous = content
        parsed = extract_json(content or "")
        if not parsed:
            tool_result = {"error": "invalid JSON"}
            reason = "invalid_json"
            continue
        action = parsed.get("action")
        if action == "finish":
            ranking, valid = complete_ranking(parsed.get("ranking"), builder.bundle.services, fallback)
            reason = "finish" if valid else "invalid_finish_ranking"
            if valid:
                break
        else:
            tool_result = builder.tool_result(index, str(action), parsed.get("service"))
            reason = "continued"
    return ranking, {"valid": valid, "reason": reason, "ranking": ranking, "call_paths": call_paths}


def run_mabc_incident(
    builder: TelemetryBuilder, index: int, run_dir: Path, prompts: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    observation = builder.observation(index)
    telemetry = canonical(observation)
    fallback = observation["fallback_ranking"]
    template = prompts["mabc"]
    case_position = int(np.flatnonzero(builder.bundle.test == index)[0])
    call_paths = []
    role_outputs = []
    role_valid = 0
    for role_index, (role, focus) in enumerate(template["roles"]):
        user = template["role_user"].format(
            common_rules=common_rules(prompts), role=role, focus=focus, telemetry=telemetry
        )
        messages = [{"role": "system", "content": template["system"]}, {"role": "user", "content": user}]
        path = run_dir / f"case-{case_position:03d}" / f"role-{role_index}.json"
        call_paths.append(str(path.relative_to(ROOT)))
        record = call_cached(path, messages, int(template["maximum_role_output_tokens"]))
        content, error = response_content(record)
        parsed = extract_json(content or "") if not error else None
        ranking, valid = complete_ranking(parsed.get("ranking") if parsed else None, builder.bundle.services, fallback)
        role_valid += int(valid)
        role_outputs.append({
            "role": role,
            "analysis": parsed.get("analysis") if parsed and isinstance(parsed.get("analysis"), str) else None,
            "ranking": ranking[:5],
            "valid": valid,
            "error": error,
        })
    aggregate_user = template["aggregate_user"].format(
        common_rules=common_rules(prompts), telemetry=telemetry, role_outputs=canonical(role_outputs)
    )
    aggregate_messages = [
        {"role": "system", "content": template["aggregate_system"]},
        {"role": "user", "content": aggregate_user},
    ]
    aggregate_path = run_dir / f"case-{case_position:03d}" / "aggregate.json"
    call_paths.append(str(aggregate_path.relative_to(ROOT)))
    record = call_cached(aggregate_path, aggregate_messages, int(template["maximum_aggregation_output_tokens"]))
    content, error = response_content(record)
    parsed = extract_json(content or "") if not error else None
    ranking, valid = complete_ranking(parsed.get("ranking") if parsed else None, builder.bundle.services, fallback)
    return ranking, {
        "valid": valid,
        "reason": "aggregate" if valid else (error or "invalid_aggregate"),
        "valid_role_outputs": role_valid,
        "ranking": ranking,
        "call_paths": call_paths,
    }


def care_base_scores(bundle: DatasetBundle) -> np.ndarray:
    matrices = []
    for seed in (42, 43, 44, 45, 46):
        archive = np.load(ROOT / f"artifacts/care/review-response-v1/{bundle.name}/seed-{seed}/predictions.npz")
        np.testing.assert_array_equal(archive["indices"], bundle.test)
        matrices.append(np.asarray(archive["final"], dtype=np.float64))
    return np.mean(matrices, axis=0)


def minimum_onset(record: dict[str, Any]) -> int | None:
    values = [record[f"{modality}_onset_bin"] for modality in ("metric", "log", "trace")]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def evidence_strength(record: dict[str, Any]) -> float:
    return (
        0.40 * record["metric_magnitude"]
        + 0.25 * record["log_magnitude"]
        + 0.25 * record["trace_magnitude"]
        + 0.10 * record["history_similarity"]
    )


def run_care_incident(
    builder: TelemetryBuilder,
    index: int,
    base_scores: np.ndarray,
    run_dir: Path,
    prompts: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    observation = builder.observation(index)
    position = int(np.flatnonzero(builder.bundle.test == index)[0])
    full_order = rank_order(base_scores[position])
    base_ranking = [builder.bundle.services[item] for item in full_order]
    top_five = base_ranking[:5]
    records = {item["service"]: item for item in observation["service_records"]}
    candidates = [{"service": service, "base_score": rounded(base_scores[position, builder.bundle.services.index(service)])} for service in top_five]
    evidence: list[dict[str, Any]] = []
    for service in top_five:
        record = records[service]
        for code, modality in (("M", "metric"), ("L", "log"), ("T", "trace")):
            evidence.append({
                "id": f"{code}:{service}", "service": service, "kind": modality,
                "magnitude": record[f"{modality}_magnitude"],
                "persistence": record[f"{modality}_persistence"],
                "onset_bin": record[f"{modality}_onset_bin"],
            })
        evidence.append({"id": f"H:{service}", "service": service, "kind": "history", "similarity": record["history_similarity"]})
    for edge in observation["dependency"]["edges"]:
        if edge["source"] in top_five and edge["target"] in top_five:
            evidence.append({"id": f"E:{edge['source']}->{edge['target']}", "kind": "observed_call", **edge})
    template = prompts["care_proposal"]
    user = template["user"].format(
        common_rules=common_rules(prompts),
        base_candidates=canonical(candidates),
        retrieved_evidence=canonical(evidence),
        telemetry=canonical(observation),
    )
    messages = [{"role": "system", "content": template["system"]}, {"role": "user", "content": user}]
    path = run_dir / f"case-{position:03d}" / "proposal.json"
    response = call_cached(path, messages, int(template["maximum_output_tokens"]))
    content, error = response_content(response)
    parsed = extract_json(content or "") if not error else None
    raw_ranking = parsed.get("ranking") if parsed else None
    proposed, parse_valid = complete_ranking(raw_ranking, top_five, top_five)
    proposed_top = proposed[0] if parse_valid else base_ranking[0]
    cited_id = None
    if isinstance(raw_ranking, list):
        for item in raw_ranking:
            if isinstance(item, dict) and item.get("service") == proposed_top:
                cited_id = item.get("evidence_id")
                break
    evidence_by_id = {item["id"]: item for item in evidence}
    cited = evidence_by_id.get(cited_id)
    citation_valid = bool(
        cited
        and (
            cited.get("service") == proposed_top
            or (cited.get("kind") == "observed_call" and cited.get("source") == proposed_top)
        )
    )
    base_top = base_ranking[0]
    promoted_record, base_record = records[proposed_top], records[base_top]
    promoted_strength, base_strength = evidence_strength(promoted_record), evidence_strength(base_record)
    candidate_onset = minimum_onset(promoted_record)
    edge_gate = False
    edge_id = None
    for edge in observation["dependency"]["edges"]:
        target = edge["target"]
        if edge["source"] != proposed_top or target not in top_five or target == proposed_top:
            continue
        target_record = records[target]
        target_onset = minimum_onset(target_record)
        if (
            evidence_strength(target_record) >= 0.50
            and candidate_onset is not None
            and target_onset is not None
            and candidate_onset <= target_onset + 1
        ):
            edge_gate = True
            edge_id = f"E:{proposed_top}->{target}"
            break
    base_confidence = float(np.max(base_scores[position]))
    gates = {
        "parsed": parse_valid,
        "changed_top": proposed_top != base_top,
        "candidate_in_base_top_five": proposed_top in top_five,
        "exact_candidate_citation": citation_valid,
        "evidence_margin_at_least_0.05": promoted_strength - base_strength >= 0.05,
        "base_confidence_below_0.8": base_confidence < 0.8,
        "observed_propagation_support": edge_gate,
    }
    accepted = all(gates.values())
    if accepted:
        final_ranking = [proposed_top] + [service for service in base_ranking if service != proposed_top]
    else:
        final_ranking = base_ranking
    return final_ranking, {
        "valid": parse_valid and error is None,
        "accepted": accepted,
        "base_top": base_top,
        "proposed_top": proposed_top,
        "base_confidence": base_confidence,
        "promoted_strength": promoted_strength,
        "base_top_strength": base_strength,
        "cited_evidence_id": cited_id,
        "supporting_edge_id": edge_id,
        "gates": gates,
        "ranking": final_ranking,
        "call_paths": [str(path.relative_to(ROOT))],
        "response_error": error,
    }


def cost_summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
    paths = sorted({path for audit in audits for path in audit["call_paths"]})
    prompt_tokens = completion_tokens = total_tokens = 0
    wall_seconds = 0.0
    successful = failed = attempts = 0
    models: dict[str, int] = {}
    for relative in paths:
        record = json.loads((ROOT / relative).read_text())
        attempts += int(record.get("attempts") or 0)
        if record.get("response"):
            successful += 1
            usage = record["response"].get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
            model = str(record["response"].get("model"))
            models[model] = models.get(model, 0) + 1
        else:
            failed += 1
        wall_seconds += float(record.get("wall_seconds") or 0.0)
    return {
        "logical_calls": len(paths),
        "successful_calls": successful,
        "failed_calls": failed,
        "http_attempts": attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "summed_call_wall_seconds": wall_seconds,
        "returned_models": models,
        "currency_cost": None,
        "pricing_source": None,
    }


def run_api_method(
    bundle: DatasetBundle,
    method: str,
    run_index: int,
    prompts: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    builder = TelemetryBuilder(bundle)
    scores = fallback_scores(bundle)
    base = care_base_scores(bundle) if method == "CARE+GPT" else None
    run_dir = OUTPUT / "llm-cache" / method / bundle.name / f"api-run-{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def worker(index: int):
        if method == "ReAct":
            return run_react_incident(builder, index, run_dir, prompts)
        if method == "mABC":
            return run_mabc_incident(builder, index, run_dir, prompts)
        assert base is not None
        return run_care_incident(builder, index, base, run_dir, prompts)

    outputs: dict[int, tuple[list[str], dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(worker, int(index)): int(index) for index in bundle.test}
        for future in concurrent.futures.as_completed(future_map):
            index = future_map[future]
            try:
                outputs[index] = future.result()
            except Exception as error:
                fallback = builder.observation(index)["fallback_ranking"]
                outputs[index] = (fallback, {
                    "valid": False, "reason": type(error).__name__, "ranking": fallback, "call_paths": []
                })
    audits = []
    for index in bundle.test:
        ranking, audit = outputs[int(index)]
        order = [bundle.services.index(service) for service in ranking]
        scores[index] = ranking_scores(order, bundle.service_count)
        audits.append({"test_position": int(np.flatnonzero(bundle.test == index)[0]), **audit})
    audit_path = OUTPUT / bundle.name / method / f"api-run-{run_index}" / "llm_audit.json"
    atomic_json(audit_path, audits)
    costs = cost_summary(audits)
    valid_count = sum(bool(audit.get("valid")) for audit in audits)
    accepted_count = sum(bool(audit.get("accepted")) for audit in audits)
    provenance = {
        "protocol_sha256": PROTOCOL_SHA,
        "llm_execution_protocol_sha256": EXECUTION_SHA,
        "llm_clarification_sha256": CLARIFICATION_SHA,
        "implementation_class": protocol["methods"][method]["implementation_class"],
        "material_adapter": protocol["methods"][method]["material_adapter"],
        "adaptation": protocol["methods"][method]["adaptation"],
        "modalities": protocol["methods"][method]["modalities"],
        "api_run_index": run_index,
        "request_seed": 2027,
        "valid_incident_outputs": valid_count,
        "invalid_incident_outputs": len(audits) - valid_count,
        "accepted_care_corrections": accepted_count if method == "CARE+GPT" else None,
        "cost": costs,
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": sha256(audit_path),
        "code": str(Path(__file__).resolve().relative_to(ROOT)),
        "code_sha256": sha256(Path(__file__)),
    }
    result = save_method_run(
        bundle, method, scores, provenance, run_name=f"api-run-{run_index}"
    )
    return result, scores[bundle.test]


def run_method(
    bundle: DatasetBundle, method: str, prompts: dict[str, Any], protocol: dict[str, Any], runs: list[int]
) -> dict[str, Any]:
    results, matrices = [], []
    for run_index in runs:
        result, matrix = run_api_method(bundle, method, run_index, prompts, protocol)
        results.append(result)
        matrices.append(matrix)
    summary = {
        "dataset": bundle.name,
        "method": method,
        "protocol_sha256": PROTOCOL_SHA,
        "llm_execution_protocol_sha256": EXECUTION_SHA,
        "complete_run_count": len(results),
        "required_run_count": 3,
        "metrics": summarize_repeats(matrices, bundle.labels[bundle.test]),
        "runs": [
            {
                "run_name": result["run_name"],
                "metrics": result["metrics"],
                "scores_sha256": result["scores_sha256"],
                "cost": result["provenance"]["cost"],
                "valid_incident_outputs": result["provenance"]["valid_incident_outputs"],
                "accepted_care_corrections": result["provenance"]["accepted_care_corrections"],
            }
            for result in results
        ],
    }
    atomic_json(OUTPUT / bundle.name / method / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["trainticket", "aiops2022"], default=["trainticket", "aiops2022"])
    parser.add_argument("--methods", nargs="+", choices=["ReAct", "mABC", "CARE+GPT"], default=["ReAct", "mABC", "CARE+GPT"])
    parser.add_argument("--runs", nargs="+", type=int, choices=[0, 1, 2], default=[0, 1, 2])
    parser.parse_args()
    raise SystemExit(
        "Historical live launcher is disabled after provider correction. "
        "Use scripts/care/fse2027_llm_recovery.py for an isolated OpenCode Go campaign; "
        "the retained historical cache remains available for offline replay."
    )


if __name__ == "__main__":
    main()
