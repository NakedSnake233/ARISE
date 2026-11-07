"""Freeze the matched-baseline protocol before opening any baseline test result."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import yaml

from fse2027_baseline_common import OUTPUT, ROOT, dataset_protocol, load_dataset, sha256


PROMPTS = ROOT / "scripts/care/fse2027_llm_prompts.json"
SOURCES = ROOT / "third_party/fse2027_baselines/SOURCE_ARCHIVES.json"
SMOKE = OUTPUT / "protocol/llm-smoke.json"
TARGET = OUTPUT / "BASELINE_PROTOCOL_FROZEN.json"
TARGET_HASH = OUTPUT / "BASELINE_PROTOCOL_FROZEN.sha256"
CONFIG = Path(os.environ.get("CARE_LLM_CONFIG", str(ROOT / "configs/agents.yaml")))


def grid(**kwargs):
    return kwargs


def source_index() -> dict[str, dict]:
    return {entry["name"]: entry for entry in json.loads(SOURCES.read_text())}


def source(name: str, url: str, license_name: str, note: str = "") -> dict:
    entry = source_index()[name]
    return {
        "url": url,
        "downloaded_archive": entry["archive"],
        "archive_sha256": entry["sha256"],
        "snapshot_file_count": entry["files"],
        "license": license_name,
        "note": note,
    }


def method(
    implementation_class: str,
    source_record: dict | None,
    native_input: str,
    harmonized_input: str,
    native_output: str,
    harmonized_output: str,
    adaptation: str,
    modalities: list[str],
    tuning: dict,
    execution: dict,
) -> dict:
    return {
        "implementation_class": implementation_class,
        "source": source_record,
        "native_input": native_input,
        "harmonized_input": harmonized_input,
        "native_output": native_output,
        "harmonized_output": harmonized_output,
        "material_adapter": implementation_class != "exact-upstream",
        "adaptation": adaptation,
        "modalities": modalities,
        "tuning": tuning,
        "execution": execution,
        "test_failure_policy": "keep incident; use frozen label-independent fallback ranking",
        "output_semantics": "one finite score per metadata-ordered candidate service; stable descending sort",
    }


def build() -> dict:
    datasets = {name: dataset_protocol(load_dataset(name)) for name in ("trainticket", "aiops2022")}
    smoke = json.loads(SMOKE.read_text())
    returned_model = smoke["response"].get("model")
    endpoint = str(yaml.safe_load(CONFIG.read_text())["llm"]["base_url"]).rstrip("/")

    deterministic = {"result_runs": 1, "selection_seed": 2027}
    stochastic = {
        "selection_seed": 2027,
        "result_seeds": [42, 43, 44, 45, 46],
        "required_complete_runs": 5,
    }
    api = {
        "complete_runs": 3,
        "run_indices": [0, 1, 2],
        "request_seed_each_run": 2027,
        "temperature": 0,
        "top_p": 1,
        "timeout_seconds": 60,
        "retries_after_initial_attempt": 3,
        "retry_backoff_seconds": [1, 2, 4],
        "concurrency": 8,
        "cache_scope": "within a run for crash recovery; never reused across complete runs",
        "best_response_selection": False,
    }

    methods = {
        "MicroScope": method(
            "paper-derived", None,
            "service performance indicators and a learned causal graph",
            "pre/post metric tensor and observed per-incident call graph",
            "ranked anomalous services",
            "candidate-service score vector",
            "Three-sigma-style metric anomaly seeds are propagated on the observed call graph. The retained tensors cannot reproduce the paper's learned metric-level causal graph.",
            ["metric", "graph"],
            grid(anomaly_threshold=[2.0, 3.0], damping=[0.65, 0.85, 0.95], reverse=[False, True], direct_mix=[0.25, 0.5, 0.75]),
            deterministic,
        ),
        "MicroRCA": method(
            "source-adapted",
            source("microrca", "https://github.com/elastisys/MicroRCA", "no explicit license in snapshot", "source inspected but not redistributed in the anonymous package"),
            "per-invocation response time, container resource metrics, and service/host attributed graph",
            "aggregated service metrics, trace edge summaries, and observed service graph",
            "PageRank-style service/host root ranking",
            "service-only score vector",
            "Drops unavailable host nodes and replaces per-invocation latency with retained trace summaries while preserving attributed anomaly propagation and personalized graph ranking.",
            ["metric", "trace", "graph"],
            grid(damping=[0.65, 0.85, 0.95], reverse=[False, True], direct_mix=[0.25, 0.5, 0.75]),
            deterministic,
        ),
        "SBLD": method(
            "paper-derived", None,
            "test-execution log-event spectra with pass/fail outcomes",
            "service/template post-boundary event spectra with training incident labels only",
            "suspicious log events",
            "service score vector obtained by frozen event-to-service aggregation",
            "Changes event-level diagnosis to service ranking because the frozen datasets contain service-binned templates rather than test-case event sequences.",
            ["log"],
            grid(formula=["ochiai", "tarantula", "jaccard", "dstar2"], active_threshold=[0.0, 1.0], service_aggregation=["max", "sum"]),
            deterministic,
        ),
        "LogFaultFlagger": method(
            "paper-derived", None,
            "historical test logs represented by clustered messages for bug prediction/localization",
            "training-only service log descriptors and incident root-service labels",
            "bug flag and suspicious log regions",
            "service-class probability vector",
            "Uses a class-balanced multinomial logistic service classifier because the original prediction target and test-log granularity are unavailable; this is not the native tool.",
            ["log"],
            grid(C=[0.1, 1.0, 10.0], class_weight=["balanced"], max_iter=[2000]),
            deterministic,
        ),
        "TraceAnomaly": method(
            "source-adapted",
            source("traceanomaly", "https://github.com/NetManAIOps/TraceAnomaly", "no explicit license in snapshot", "TensorFlow-1 source inspected; no upstream code redistributed"),
            "individual traces encoded as service-level vectors for deep Bayesian-network anomaly detection",
            "retained per-service/per-bin trace feature vectors",
            "trace anomaly probability",
            "per-service reconstruction anomaly score",
            "Replaces unavailable individual-trace Bayesian networks with a small variational autoencoder on training pre-fault service-bin vectors, then aggregates post-boundary reconstruction error by service.",
            ["trace"],
            grid(latent_dim=[1, 2], aggregation=["mean", "max", "q90"], hidden_dim=[16], learning_rate=[0.001], epochs=[80]),
            stochastic,
        ),
        "MicroRank": method(
            "source-adapted",
            source("microrank", "https://github.com/IntelligentDDS/MicroRank", "MIT"),
            "individual normal/anomalous request traces and service-operation graph",
            "time bins treated as request units with service trace observations and observed service graph",
            "spectrum-weighted PageRank service ranking",
            "candidate-service score vector",
            "Preserves anomalous/normal partition, bipartite ranking, and spectrum scoring, but substitutes retained time bins for discarded request identifiers.",
            ["trace", "graph"],
            grid(anomaly_quantile=[0.75, 0.85, 0.95], spectrum=["ochiai", "tarantula"], damping=[0.65, 0.85]),
            deterministic,
        ),
        "Nezha": method(
            "source-adapted",
            source("nezha", "https://github.com/IntelligentDDS/Nezha", "MIT"),
            "raw metric/log/trace events transformed to multimodal event graphs and differential patterns",
            "thresholded service/time/channel events and observed call edges",
            "ranked fine-grained root-cause event patterns",
            "service score vector",
            "Reconstructs differential event-pattern support from retained tensors and aggregates event nodes to services; raw log text, spans, and fine-grained entities are unavailable.",
            ["metric", "log", "trace", "graph"],
            grid(event_threshold=[1.5, 2.0, 2.5], service_aggregation=["sum", "max"], graph_mix=[0.0, 0.25, 0.5]),
            deterministic,
        ),
        "Eadro": method(
            "source-adapted",
            source("eadro", "https://github.com/BEbillionaireUSD/Eadro", "no explicit license in snapshot", "source inspected but not redistributed in the anonymous package"),
            "metric/trace time series, log events, and service dependency graph for joint anomaly detection and localization",
            "frozen metric/log/trace tensors and observed service graph with incident labels from fit only",
            "joint anomaly decision and root-service logits",
            "root-service probability vector",
            "Retains modality encoders, graph attention, global pooling, and service localization; omits the separate anomaly detector and replaces unavailable source preprocessing. Source naming defects are corrected in the adapter.",
            ["metric", "log", "trace", "graph"],
            grid(learning_rate=[0.001, 0.003], dropout=[0.2, 0.4], hidden_dim=[32], attention_heads=[4], max_epochs=[120], patience=[15], batch_size=[16], weight_decay=[0.0001]),
            stochastic,
        ),
        "MULAN": method(
            "paper-derived", None,
            "raw multimodal telemetry and learned causal structure",
            "frozen metric/log/trace tensors and observed call graph",
            "causal-structure random-walk root ranking",
            "candidate-service score vector",
            "Rebuilds modality encoders, contrastive alignment, KPI-conditioned fusion, and graph random walk from the paper; the authors' preprocessing and implementation were not available.",
            ["metric", "log", "trace", "graph"],
            grid(temperature=[0.1, 0.2], graph_mix=[0.25, 0.5, 0.75], hidden_dim=[32], learning_rate=[0.001], max_epochs=[120], patience=[15], batch_size=[16], weight_decay=[0.0001]),
            stochastic,
        ),
        "ReAct": method(
            "paper-derived", None,
            "language model plus interactive environment-specific tools",
            "gpt-5.6-luna with three frozen telemetry inspection actions",
            "free-form answer or action trajectory",
            "parsed top-five service ranking",
            "Instantiates ReAct for the service-ranking task with inspect_service, inspect_neighbors, compare_history, and finish; the original work does not define this RCA environment.",
            ["metric", "log", "trace", "graph", "history", "llm"],
            {"selection": "none; exact frozen prompt", "prompt_file": str(PROMPTS.relative_to(ROOT)), "prompt_sha256": sha256(PROMPTS), "maximum_calls_per_incident": 3, "maximum_output_tokens_per_call": 600},
            api,
        ),
        "mABC": method(
            "source-adapted",
            source("mabc", "https://github.com/knediny/mABC", "Apache-2.0"),
            "seven role agents, shared microservice observations, and an aggregation/voting stage",
            "gpt-5.6-luna; seven frozen role calls and one aggregation call over the common telemetry package",
            "textual diagnosis assembled by role discussion/voting",
            "parsed top-five service ranking",
            "Preserves the seven public role identities and aggregation structure. It replaces hard-coded demonstration utilities and the source snapshot's disabled vote path with dataset adapters and a fixed JSON vote.",
            ["metric", "log", "trace", "graph", "history", "llm"],
            {"selection": "none; exact frozen prompt", "prompt_file": str(PROMPTS.relative_to(ROOT)), "prompt_sha256": sha256(PROMPTS), "role_calls": 7, "aggregation_calls": 1, "maximum_role_output_tokens": 350, "maximum_aggregation_output_tokens": 500},
            api,
        ),
        "CARE": method(
            "project-native", None,
            "metric/log/trace tensors, observed graph, training history, and incident labels for fit only",
            "same",
            "service ranking",
            "candidate-service score vector",
            "Frozen five-seed evidence-model configuration from the review-response experiment; no LLM call on this path.",
            ["metric", "log", "trace", "graph", "history"],
            {"selection": "already frozen in artifacts/care/review-response-v1/config.json"},
            stochastic,
        ),
        "CARE+GPT": method(
            "project-native", None,
            "CARE base candidates plus retrieved evidence and common telemetry",
            "same; gpt-5.6-luna proposes one evidence-cited reordering subject to the frozen deterministic acceptance gate",
            "service ranking and cited evidence records",
            "candidate-service score vector plus audit trail",
            "Optional one-call proposal path; reported separately from the zero-call CARE row and never substituted according to outcome.",
            ["metric", "log", "trace", "graph", "history", "llm"],
            {"selection": "none; exact frozen prompt and gate", "prompt_file": str(PROMPTS.relative_to(ROOT)), "prompt_sha256": sha256(PROMPTS), "maximum_calls_per_incident": 1, "maximum_output_tokens": 450},
            api,
        ),
    }

    return {
        "protocol_version": 1,
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pre_result_assertion": "No new baseline adapter has been evaluated on test labels or test metrics before this file was frozen. Earlier CARE development had inspected both test cohorts, so all paper comparisons remain exploratory.",
        "evidence_boundary": {
            "status": "post-hoc exploratory",
            "practically_notable_gap_absolute": 0.05,
            "ordering_uncertain_when_paired_bootstrap_crosses_zero": True,
            "prohibited_claims": ["confirmatory significance", "out-of-sample generalization", "superiority", "state of the art"],
        },
        "datasets": datasets,
        "common_evaluator": {
            "metrics": ["PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5"],
            "single_ground_truth": True,
            "ties": "stable ascending metadata service order",
            "bootstrap": {"unit": "unique incident", "draws": 10000, "seed": 2027, "interval": "percentile 95%; conditional on fixed local seeds/API runs"},
            "validation_selection": ["PR@1", "MAP@5", "PR@3", "lexical configuration JSON"],
            "no_test_label_access": True,
            "fallback": "equal mean of row-normalized label-independent metric, log, and trace anomaly magnitudes; stable service-order ties",
            "invalid_incidents_remain_in_denominator": True,
            "test_result_reader": "scripts/care/fse2027_baseline_common.py",
        },
        "llm_protocol": {
            "requested_model": "gpt-5.6-luna",
            "smoke_returned_model": returned_model,
            "endpoint": endpoint,
            "api_route": "/chat/completions",
            "smoke_artifact": str(SMOKE.relative_to(ROOT)),
            "smoke_sha256": sha256(SMOKE),
            "access_date_utc": "2026-09-26",
            "prompt_file": str(PROMPTS.relative_to(ROOT)),
            "prompt_sha256": sha256(PROMPTS),
            "credentials": "read at request time from existing config; never serialized",
            "telemetry_package": "metadata-order services; row-normalized modality anomaly/persistence/onset summaries; observed directed dependency edges; training-only nearest history; no labels, test fault types, or result-derived hints",
            "parser": "first complete JSON object; validate keys, known distinct services, and complete with label-independent fallback order",
            "fallback": "common label-independent anomaly ranking included in the prompt",
            "pricing": None,
            **api,
        },
        "methods": methods,
        "integrity_checks_required": [
            "shape", "finite scores", "stable ties", "label blindness", "degenerate all-zero input",
            "empty graph", "empty retrieval", "invalid LLM JSON", "unknown LLM service", "timeout fallback",
            "upstream bundled example when executable", "no test tuning", "complete denominator",
        ],
    }


def main() -> None:
    if TARGET.exists() or TARGET_HASH.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {TARGET}")
    if not PROMPTS.exists() or not SOURCES.exists() or not SMOKE.exists():
        raise SystemExit("missing prompt, source, or LLM smoke artifact")
    payload = build()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    digest = sha256(TARGET)
    TARGET_HASH.write_text(f"{digest}  {TARGET.name}\n")
    os.chmod(TARGET, 0o444)
    os.chmod(TARGET_HASH, 0o444)
    print(json.dumps({"status": "frozen", "protocol": str(TARGET.relative_to(ROOT)), "sha256": digest, "methods": len(payload["methods"]), "datasets": list(payload["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
