"""Aggregate measured CARE LLM training/inference costs without rerunning models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def deployment_adapter_bytes(path: Path) -> int:
    return sum(
        (path / name).stat().st_size
        for name in ("adapter_model.safetensors", "adapter_config.json")
    )


def parameter_summary(paths: list[Path]) -> dict[str, int]:
    """Count tensor elements without materializing model weights."""
    tensors = parameters = 0
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors += 1
                shape = handle.get_slice(key).get_shape()
                parameters += int(np.prod(shape, dtype=np.int64))
    return {"tensors": tensors, "parameters": parameters}


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "total": float(array.sum()),
    }


def exact_free_form_cost(tokenizer) -> dict:
    data = ROOT / "data/care/sft-trainticket"
    artifact = ROOT / "artifacts/care/correction-trainticket"
    inputs = json.loads((data / "test_inputs.json").read_text())
    rows = json.loads(
        (ROOT / "data/care/trainticket-prepared/incidents.json").read_text()
    )
    indices = [i for i, row in enumerate(rows) if row["split"] == "test"]
    input_tokens, output_tokens, seconds = [], [], []
    for item, index in zip(inputs, indices):
        record = json.loads((artifact / f"case-{index:05d}.json").read_text())
        content = item["instruction"] + "\n" + item["input"]
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_tokens.append(len(token_ids))
        output_tokens.append(
            len(tokenizer.encode(record["response"], add_special_tokens=False))
        )
        seconds.append(record["seconds"])
    training = json.loads(
        (ROOT / "artifacts/care/qwen3-care-lora-trainticket/train_results.json").read_text()
    )
    return {
        "variant": "free_form_json",
        "dataset": "trainticket",
        "inference_input_tokens": distribution(input_tokens),
        "inference_output_tokens": distribution(output_tokens),
        "inference_seconds": distribution(seconds),
        "training_seconds": training["train_runtime"],
        "training_epochs": training["epoch"],
        "deployment_adapter_bytes": deployment_adapter_bytes(
            ROOT / "artifacts/care/qwen3-care-lora-trainticket"
        ),
        "adapter_parameter_summary": parameter_summary([
            ROOT / "artifacts/care/qwen3-care-lora-trainticket/adapter_model.safetensors"
        ]),
        "complete_training_artifact_bytes": directory_bytes(
            ROOT / "artifacts/care/qwen3-care-lora-trainticket"
        ),
        "output_token_scope": (
            "Tokens in the decoded visible response; terminal special tokens "
            "removed by the original skip_special_tokens decode are unavailable."
        ),
        "actual_api_charge_usd": 0.0,
    }


def compact_trainticket_cost(tokenizer) -> dict:
    artifact = ROOT / "artifacts/care/contrast-prompt-trainticket"
    records = []
    for split in ("validation", "test"):
        for path in sorted(artifact.glob(f"{split}-*.json")):
            record = json.loads(path.read_text())
            records.append((split, record))
    by_split = {}
    for split in ("validation", "test"):
        selected = [record for name, record in records if name == split]
        if not selected:
            continue
        by_split[split] = {
            "input_tokens": distribution([record["input_tokens"] for record in selected]),
            "output_tokens": distribution([
                len(tokenizer.encode(record["response"], add_special_tokens=False))
                for record in selected
            ]),
            "seconds": distribution([record["seconds"] for record in selected]),
        }
    selection = json.loads((artifact / "selection.json").read_text())
    evaluation = (
        json.loads((artifact / "evaluation.json").read_text())
        if (artifact / "evaluation.json").exists()
        else None
    )
    return {
        "variant": "compact_json",
        "dataset": "trainticket",
        "splits": by_split,
        "validation_selected_fusion_weight": selection["selected_weight"],
        "actual_api_charge_usd": 0.0,
        "adapter_reused_from_free_form": True,
        "peak_gpu_memory_bytes": (
            evaluation.get("cost", {}).get("peak_gpu_memory_bytes")
            if evaluation
            else None
        ),
    }


def aiops_cost() -> dict:
    artifact = ROOT / "artifacts/care/lora-aiops-paper128-v2"
    records = json.loads((artifact / "lora-inference.json").read_text())
    training = json.loads((artifact / "adapter/train_results.json").read_text())
    groups = {}
    for seed in sorted({record["order_seed"] for record in records}):
        selected = [record for record in records if record["order_seed"] == seed]
        groups[str(seed)] = {
            "input_tokens": distribution([record["input_tokens"] for record in selected]),
            "output_tokens": distribution([1 for _ in selected]),
            "seconds": distribution([record["seconds"] for record in selected]),
        }
    return {
        "variant": "constrained_single_token",
        "dataset": "aiops2022",
        "inference_order_seeds": groups,
        "training_seconds": training["train_runtime"],
        "training_epochs": training["epoch"],
        "deployment_adapter_bytes": deployment_adapter_bytes(artifact / "adapter"),
        "adapter_parameter_summary": parameter_summary([
            artifact / "adapter/adapter_model.safetensors"
        ]),
        "complete_training_artifact_bytes": directory_bytes(artifact / "adapter"),
        "actual_api_charge_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/care/review-response-v1/llm_costs.json"),
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models/Qwen3-8B", local_files_only=True
    )
    result = {
        "hardware": "NVIDIA A100-SXM4-80GB for recorded local LLM runs",
        "base_model": "Qwen3-8B",
        "base_model_bytes": directory_bytes(ROOT / "models/Qwen3-8B"),
        "base_model_parameter_summary": parameter_summary(
            sorted((ROOT / "models/Qwen3-8B").glob("*.safetensors"))
        ),
        "charging_scope": (
            "Measured local wall time, tokens, and storage. No commercial API was "
            "called, so actual API charge is zero; electricity and cloud-equivalent "
            "prices were not instrumented and are not estimated."
        ),
        "variants": [
            exact_free_form_cost(tokenizer),
            compact_trainticket_cost(tokenizer),
            aiops_cost(),
        ],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
