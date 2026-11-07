"""Local Qwen3-8B LoRA proposals for frozen CARE, without test-label use.

Run ``infer`` before ``score``. The former does not use any root label; the latter
scores only after every case has a persisted model response. Existing
dataset-specific LoRA checkpoints are reused, without changing the frozen CARE
or CARE+GPT artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

try:
    from fse2027_baseline_common import ROOT, load_dataset, ranking_metrics
except ImportError:
    from .fse2027_baseline_common import ROOT, load_dataset, ranking_metrics


OUTPUT = ROOT / "artifacts/care/care-local-qwen3-8b-v3"
MODEL = ROOT / "models/Qwen3-8B"
ADAPTERS = {
    "trainticket": ROOT / "artifacts/care/qwen3-care-lora-trainticket",
    "aiops2022": ROOT / "artifacts/care/lora-aiops-paper128-v2/adapter",
}
PROTOCOL = {
    "version": 3,
    "model": "Qwen3-8B with separately fitted, dataset-specific local LoRA adapters",
    "ranking_input": "arithmetic mean of five frozen CARE seed distributions, seeds 42-46",
    "trainticket_input": "frozen legacy TrainTicket LoRA test inputs; independently generated old-base top-five candidates, telemetry and training-only graph facts",
    "aiops2022_input": "all-service, label-free existing Qwen LoRA prompt, fixed option-order seed 42",
    "trainticket_generation_tokens": 64,
    "gate": {
        "candidate_in_care_top5": True,
        "minimum_standard_expert_advantage": 0.05,
        "care_top_probability_below": 0.8,
        "accepted_action": "move proposed service to first; preserve other CARE order",
        "rejected_action": "preserve full CARE order",
    },
    "reference": "three-run CARE+GPT mean on the same frozen test incidents",
    "status": "exploratory; these test cohorts were inspected in previous project iterations",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def care_scores(dataset: str, key: str = "final") -> np.ndarray:
    bundle = load_dataset(dataset)
    values = []
    for seed in range(42, 47):
        path = ROOT / f"artifacts/care/review-response-v1/{dataset}/seed-{seed}/predictions.npz"
        with np.load(path) as archive:
            np.testing.assert_array_equal(archive["indices"], bundle.test)
            values.append(np.asarray(archive[key], dtype=np.float64))
    return np.mean(values, axis=0)


def protocol_file() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != PROTOCOL:
            raise RuntimeError("Frozen local LLM protocol differs; choose a new run directory")
    else:
        path.write_text(json.dumps(PROTOCOL, indent=2, ensure_ascii=False) + "\n")
    return path


def trainticket_inputs(bundle, scores: np.ndarray):
    """Reuse the frozen format on which the existing TrainTicket LoRA was fitted."""
    inputs = json.loads((ROOT / "data/care/sft-trainticket/test_inputs.json").read_text())
    if len(inputs) != len(bundle.test):
        raise RuntimeError("TrainTicket local LLM prompt count mismatch")
    for position, index in enumerate(bundle.test):
        row = inputs[position]
        observation = json.loads(row["input"])
        if observation["incident_id"] != f"case-{index:05d}":
            raise RuntimeError("TrainTicket local LLM prompt order mismatch")
        yield position, int(index), row["instruction"] + "\n" + row["input"]


def aiops_inputs(bundle):
    rows = json.loads((ROOT / "data/care/sft-aiops-paper128-v2/test_inputs.json").read_text())
    selected = [row for row in rows if row["order_seed"] == 42]
    if [row["index"] for row in selected] != bundle.test.tolist():
        raise RuntimeError("AIOps Qwen prompt indices differ from frozen test split")
    for position, row in enumerate(selected):
        yield position, int(row["index"]), row["instruction"] + "\n" + row["input"], row["mapping"]


def infer(dataset: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol_file()
    if not torch.cuda.is_available():
        raise RuntimeError("Host CUDA is required; run with the recorded local runtime")
    bundle = load_dataset(dataset)
    scores = care_scores(dataset)
    output = OUTPUT / dataset / "responses"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTERS[dataset], is_trainable=False).eval()
    letters = "ABCDEFGHIJ"
    letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in letters]
    if dataset == "aiops2022" and any(len(tokenizer.encode(c, add_special_tokens=False)) != 1 for c in letters):
        raise RuntimeError("Option letters are not single tokens")
    inputs = trainticket_inputs(bundle, scores) if dataset == "trainticket" else aiops_inputs(bundle)
    for row in inputs:
        if dataset == "trainticket":
            position, index, prompt = row
            mapping = None
        else:
            position, index, prompt, mapping = row
        path = output / f"case-{position:03d}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing["index"] != index or existing["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
                raise RuntimeError(f"Cached prompt mismatch: {path}")
            print(dataset, position + 1, "/", len(bundle.test), "cached", flush=True)
            continue
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
        ).to("cuda:0")
        start = time.perf_counter()
        with torch.inference_mode():
            if dataset == "aiops2022":
                logits = model(input_ids=encoded, attention_mask=torch.ones_like(encoded),
                               use_cache=False, logits_to_keep=1).logits[0, -1].float()
                probabilities = logits[letter_ids].softmax(-1).cpu().numpy()
                top_letter = letters[int(np.argmax(probabilities))]
                proposal = mapping[top_letter]
                response = {"letter": top_letter, "letter_probabilities": probabilities.tolist()}
            else:
                generated = model.generate(input_ids=encoded, attention_mask=torch.ones_like(encoded),
                                           max_new_tokens=64, do_sample=False)
                response_text = tokenizer.decode(generated[0][encoded.shape[1]:], skip_special_tokens=True)
                match = re.search(r'"root_cause"\s*:\s*"([^"]+)"', response_text)
                proposal = match.group(1) if match else None
                response = {"text": response_text}
        elapsed = time.perf_counter() - start
        payload = {
            "index": index, "test_position": position,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": int(encoded.shape[1]), "output_tokens": 1 if dataset == "aiops2022" else int(generated.shape[1] - encoded.shape[1]),
            "seconds": elapsed, "proposal": proposal, "response": response,
            "model": str(MODEL.relative_to(ROOT)), "adapter": str(ADAPTERS[dataset].relative_to(ROOT)),
            "mapping": mapping,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(dataset, position + 1, "/", len(bundle.test), proposal, f"{elapsed:.2f}s", flush=True)


def score(dataset: str) -> None:
    protocol_file()
    bundle = load_dataset(dataset)
    base = care_scores(dataset)
    standard = care_scores(dataset, "standard_evidence")
    response_dir = OUTPUT / dataset / "responses"
    paths = [response_dir / f"case-{position:03d}.json" for position in range(len(bundle.test))]
    if not all(p.exists() for p in paths):
        raise RuntimeError("All test inference records must exist before scoring")
    records = [json.loads(p.read_text()) for p in paths]
    prompt_rows = list(trainticket_inputs(bundle, base)) if dataset == "trainticket" else list(aiops_inputs(bundle))
    if dataset == "trainticket":
        source_inputs = json.loads((ROOT / "data/care/sft-trainticket/test_inputs.json").read_text())
        allowed_rows = [json.loads(row["input"])["candidates"] for row in source_inputs]
    else:
        allowed_rows = [list(row[3].values()) for row in prompt_rows]
    for position, row in enumerate(records):
        if row["index"] != int(bundle.test[position]):
            raise RuntimeError("Inference record index mismatch")
        prompt = prompt_rows[position][2]
        if row["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
            raise RuntimeError("Inference prompt checksum mismatch")
    scores = np.zeros_like(base)
    decisions = []
    for position, record in enumerate(records):
        order = np.argsort(-base[position], kind="stable").tolist()
        leader = order[0]
        proposal = record["proposal"]
        candidate = bundle.services.index(proposal) if proposal in bundle.services else None
        allowed = allowed_rows[position]
        advantage = float(standard[position, candidate] - standard[position, leader]) if candidate is not None else None
        accept = bool(
            candidate is not None and proposal in allowed and candidate != leader and candidate in order[:5]
            and advantage >= PROTOCOL["gate"]["minimum_standard_expert_advantage"]
            and base[position, leader] < PROTOCOL["gate"]["care_top_probability_below"]
        )
        if accept:
            order.remove(candidate)
            order.insert(0, candidate)
        scores[position, order] = np.arange(bundle.service_count, 0, -1)
        decisions.append({
            "test_position": position, "index": int(bundle.test[position]),
            "care_top": bundle.services[leader], "proposal": proposal,
            "accepted": accept, "standard_expert_advantage": advantage,
            "care_top_confidence": float(base[position, leader]),
            "ranking": [bundle.services[j] for j in order],
            "response_sha256": digest(paths[position]),
        })
    # Only the evaluator uses held-out labels; inference and gating above do not.
    labels = bundle.labels[bundle.test]
    reference_path = ROOT / "Paper/fse2027_revision_20260926/replication/main_results.csv"
    import csv
    with reference_path.open(newline="") as f:
        reference = next(row for row in csv.DictReader(f) if row["dataset"] == dataset and row["method"] == "CARE+GPT")
    metrics = ranking_metrics(scores, labels)
    care_metrics = ranking_metrics(base, labels)
    checks = {metric: metrics[metric] + 1e-12 >= float(reference[f"{metric}_mean"]) for metric in metrics}
    report = {
        "dataset": dataset, "model": PROTOCOL["model"],
        "protocol_sha256": digest(OUTPUT / "protocol.json"),
        "code_sha256": digest(Path(__file__)),
        "cases": len(records), "accepted_proposals": sum(row["accepted"] for row in decisions),
        "care_ensemble": care_metrics, "care_local": metrics,
        "care_gpt_three_run_mean": {metric: float(reference[f"{metric}_mean"]) for metric in metrics},
        "at_least_care_gpt_by_metric": checks, "all_metrics_at_least_care_gpt": all(checks.values()),
        "mean_local_seconds": float(np.mean([r["seconds"] for r in records])),
        "mean_input_tokens": float(np.mean([r["input_tokens"] for r in records])),
        "mean_output_tokens": float(np.mean([r["output_tokens"] for r in records])),
        "limitations": PROTOCOL["status"],
    }
    if dataset == "trainticket":
        legacy_proposals = []
        control = np.zeros_like(base)
        control_accepted = 0
        for position, prompt_row in enumerate(source_inputs):
            observation = json.loads(prompt_row["input"])
            legacy = max(observation["candidates"],
                         key=lambda service: float(observation["telemetry"][service]["base_score"]))
            legacy_proposals.append(legacy)
            order = np.argsort(-base[position], kind="stable").tolist()
            candidate = bundle.services.index(legacy)
            leader = order[0]
            if (candidate != leader and candidate in order[:5]
                    and standard[position, candidate] - standard[position, leader] >= 0.05
                    and base[position, leader] < 0.8):
                order.remove(candidate)
                order.insert(0, candidate)
                control_accepted += 1
            control[position, order] = np.arange(bundle.service_count, 0, -1)
        report["no_llm_legacy_top1_control"] = {
            "llm_agrees_with_legacy_top1": sum(a == b["proposal"] for a, b in zip(legacy_proposals, records)),
            "accepted_proposals": control_accepted,
            "metrics": ranking_metrics(control, labels),
            "identical_final_rankings_to_local_llm": bool(np.array_equal(control, scores)),
        }
        np.savez_compressed(OUTPUT / dataset / "no_llm_control.npz", indices=bundle.test,
                            scores=control)
    directory = OUTPUT / dataset
    np.savez_compressed(directory / "predictions.npz", indices=bundle.test, scores=scores,
                        care_ensemble=base, standard_evidence=standard)
    (directory / "decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
    (directory / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("infer", "score"))
    parser.add_argument("--dataset", choices=tuple(ADAPTERS), required=True)
    args = parser.parse_args()
    if args.phase == "infer":
        infer(args.dataset)
    else:
        score(args.dataset)


if __name__ == "__main__":
    main()
