"""Derive OpenCode Go completion and usage-equivalent accounting from raw caches."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path

from fse2027_baseline_common import ROOT, atomic_json

PRICES = {"standard_input": 0.20, "output": 1.20, "cached_read": 0.02, "cached_write": 0.25}


def accumulate(target: Counter, record: dict) -> None:
    usage = (record.get("provider_response") or {}).get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int(input_details.get("cached_tokens") or 0)
    written = int(input_details.get("cache_write_tokens") or 0)
    target["calls"] += 1
    target["input_tokens"] += input_tokens
    target["cached_read_tokens"] += cached
    target["cache_write_tokens"] += written
    target["standard_input_tokens"] += max(input_tokens - cached - written, 0)
    target["output_tokens"] += int(usage.get("output_tokens") or 0)
    target["reasoning_tokens_in_output"] += int(output_details.get("reasoning_tokens") or 0)
    target["reported_total_tokens"] += int(usage.get("total_tokens") or 0)


def priced(values: Counter) -> dict:
    result = dict(values)
    result["usage_equivalent_usd"] = round(
        values["standard_input_tokens"] * PRICES["standard_input"] / 1_000_000
        + values["output_tokens"] * PRICES["output"] / 1_000_000
        + values["cached_read_tokens"] * PRICES["cached_read"] / 1_000_000
        + values["cache_write_tokens"] * PRICES["cached_write"] / 1_000_000,
        6,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    status = json.loads((output / "status.json").read_text())
    if status.get("status") != "complete":
        raise RuntimeError("Cost report requires a complete campaign")
    totals = Counter()
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    completions = Counter()
    models = Counter()
    sampling = Counter()
    files = sorted((output / "llm-cache").rglob("*.json"))
    for path in files:
        relative = path.relative_to(output / "llm-cache")
        method, dataset = relative.parts[:2]
        record = json.loads(path.read_text())
        if record.get("error") or record.get("response") is None:
            raise RuntimeError("Failed cache record in complete campaign: " + str(path))
        raw = record.get("provider_response") or {}
        provider_status = str(raw.get("status") or "missing")
        completions[provider_status] += 1
        if provider_status != "completed":
            raise RuntimeError("Non-completed provider response in complete campaign: " + str(path))
        models[str(raw.get("model") or "missing")] += 1
        sampling[(raw.get("temperature"), raw.get("top_p"),
                  (raw.get("reasoning") or {}).get("effort"))] += 1
        accumulate(totals, record)
        accumulate(groups[(dataset, method)], record)
    if not files:
        raise RuntimeError("No study cache records found")
    if set(models) != {"gpt-5.6-luna"}:
        raise RuntimeError("Unexpected returned model set: " + repr(dict(models)))
    report = {
        "status": "derived_from_explicit_provider_usage_categories",
        "currency": "USD", "actual_incremental_charge": None,
        "actual_charge_reason": "OpenCode Go is a subscription. This is a usage-equivalent estimate, not an invoice or additional charge.",
        "pricing_source": "https://opencode.ai/docs/go/",
        "pricing_access_date_utc": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "model": "gpt-5.6-luna", "price_per_million_tokens": PRICES,
        "estimation_rule": "Price provider-reported standard input, cached read, cache write, and output categories; standard input is input minus cached-read minus cache-write.",
        "campaign": priced(totals),
        "by_dataset_method": {f"{dataset}/{method}": priced(values)
                              for (dataset, method), values in sorted(groups.items())},
    }
    completion = {
        "status": "pass", "campaign": output.name, "study_cache_records": len(files),
        "provider_status_counts": dict(completions), "returned_models": dict(models),
        "observed_sampling": [{"temperature": key[0], "top_p": key[1],
                               "reasoning_effort": key[2], "responses": count}
                              for key, count in sampling.items()],
        "requirement": "every study response has provider status completed",
    }
    atomic_json(output / "COST_ESTIMATE.json", report)
    atomic_json(output / "GENERATION_COMPLETION.json", completion)
    print(json.dumps({"cost": report["campaign"], "completion": completion}, indent=2))


if __name__ == "__main__":
    main()
