"""Exact Drain acceleration with atomic preprocessing/recovery provenance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .common import ROOT, json_dump, sha256
from . import remaining_queue as q
from . import re2_data as data
from . import re2_preprocessing_runtime as runtime
from .drain_bitset_compact import accelerated


def ancestry(original_id):
    reverse = {}
    for path in q.MANIFESTS.glob("*.json"):
        for original, recipe in json.loads(path.read_text()).get("recovery_recipes", {}).items():
            child = recipe["recovery_job"]
            if child in reverse: raise ValueError("ambiguous recovery parent")
            reverse[child] = original
    result = []; seen = {original_id}; node = original_id
    while node in reverse:
        node = reverse[node]
        if node in seen: raise ValueError("cyclic recovery ancestry")
        seen.add(node); result.append(node)
    return result


def ready():
    source = ROOT / "exp_revision/drain_bitset_compact.py"
    spec = ROOT / "exp_revision/drain_bitset_compact_spec.json"
    for name in ("T8_drain_bitset_compact_witness", "T8_drain_bitset_compact_followdoc"):
        witness = json.loads((ROOT / "results_v2" / name / "witness.json").read_text())
        if (witness["status"] != "passed_exact_bitset_witness" or witness["seed"] != 2027
                or witness["comparisons"] != 4000 or witness["distinct_messages"] < 60000
                or witness["source_sha256"] != sha256(source) or witness["spec_sha256"] != sha256(spec)):
            raise ValueError("exact Drain witness missing or stale")


def prepare(dataset, fold, output, original_id=None):
    ready(); ancestors = ancestry(original_id) if original_id else []
    original_data_dump = data.json_dump; original_runtime_dump = runtime.json_dump
    evidence = dict(algorithm_changed=False, implementation="exact compact inverted Drain bitsets",
        source_sha256=sha256(ROOT / "exp_revision/drain_bitset_compact.py"),
        spec_sha256=sha256(ROOT / "exp_revision/drain_bitset_compact_spec.json"),
        witness_sha256={name: sha256(ROOT / f"results_v2/{name}/witness.json") for name in
            ("T8_drain_bitset_compact_witness", "T8_drain_bitset_compact_followdoc")})

    def pending(path, value):
        if path == output / "run_meta.json" and value.get("status") == "completed_re2_fold":
            value = {**value, "status": "pending_accelerator_receipt"}
        return original_data_dump(path, value)

    def final(path, value):
        if path == output / "run_meta.json" and value.get("status") == "completed_re2_fold":
            value = {**value, "acceleration": evidence}
            if original_id:
                value["recovery"] = {**value["recovery"], "original_ancestors": ancestors,
                    "reason": "observed slow preprocessing; identical parser outputs, corpus, order and settings"}
        return original_runtime_dump(path, value)

    data.json_dump = pending; runtime.json_dump = final
    try:
        with accelerated(): runtime.prepare(dataset, fold, output, original_id)
        if not original_id:
            path = output / "run_meta.json"; value = json.loads(path.read_text())
            final(path, {**value, "status": "completed_re2_fold"})
    finally:
        data.json_dump = original_data_dump; runtime.json_dump = original_runtime_dump


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", choices=data.DATASETS, required=True)
    parser.add_argument("--fold", type=int, required=True); parser.add_argument("--original-id")
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    prepare(args.dataset, args.fold, args.output, args.original_id)


if __name__ == "__main__": main()
