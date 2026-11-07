"""Read-only offline replay for the output-budget compatibility campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fse2027_llm_baselines as runner
import fse2027_llm_budget_recovery as budget
import fse2027_llm_client as client
from fse2027_baseline_common import ROOT, load_dataset
from fse2027_llm_client import sha256_json


def forbid_request(*args, **kwargs):
    raise RuntimeError("offline replay attempted a network request")


def discard_write(*args, **kwargs):
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = budget.freeze_campaign(output)
    budget.activate_campaign(output, manifest)
    status = json.loads((output / "status.json").read_text())
    if status.get("status") != "complete":
        raise RuntimeError("Only a complete budget-v2 campaign can be replayed")
    completed = {tuple(item) for item in status.get("completed_run_sets", [])}
    required = {(dataset, method, run) for dataset in budget.DATASETS
                for method in budget.METHODS for run in range(3)}
    if completed != required:
        raise RuntimeError("Budget-v2 status lacks exact run-set coverage")

    referenced: set[Path] = set()
    expected: dict[tuple[str, str, int], np.ndarray] = {}
    stored_results: dict[tuple[str, str, int], dict] = {}
    for dataset, method, run in sorted(required):
        directory = output / dataset / method / f"api-run-{run}"
        audit_path = directory / "llm_audit.json"
        scores_path = directory / "scores.npz"
        result_path = directory / "result.json"
        if not all(path.is_file() for path in (audit_path, scores_path, result_path)):
            raise RuntimeError("Missing replay evidence: " + str(directory))
        archive = np.load(scores_path)
        expected[(dataset, method, run)] = archive["all_scores"].copy()
        stored_results[(dataset, method, run)] = json.loads(result_path.read_text())
        for audit in json.loads(audit_path.read_text()):
            for relative in audit["call_paths"]:
                path = (ROOT / relative).resolve()
                if path in referenced:
                    raise RuntimeError("A cache record is referenced more than once: " + str(path))
                referenced.add(path)
                record = json.loads(path.read_text())
                if record.get("error") or record.get("response") is None:
                    raise RuntimeError("Failed cache in complete campaign: " + str(path))
                if (record.get("provider_response") or {}).get("status") != "completed":
                    raise RuntimeError("Non-completed provider generation: " + str(path))
                for field in ("request", "response", "provider_response"):
                    if record.get(field) is None or record.get(field + "_sha256") != sha256_json(record[field]):
                        raise RuntimeError("Missing or invalid cache hash: " + str(path) + ":" + field)
    present = {path.resolve() for path in (output / "llm-cache").rglob("*.json")}
    if present != referenced:
        raise RuntimeError("Cache coverage mismatch: orphan or unreferenced records exist")

    current_key: list[tuple[str, str, int] | None] = [None]

    def validate_save(bundle, method, scores, provenance, run_name):
        run = int(run_name.rsplit("-", 1)[1])
        key = (bundle.name, method, run)
        current_key[0] = key
        np.testing.assert_array_equal(np.asarray(scores), expected[key])
        return stored_results[key]

    runner.atomic_json = discard_write
    runner.save_method_run = validate_save
    budget.client.request_chat = forbid_request
    client.request_chat = forbid_request
    protocol, prompts = runner.verify_protocols()
    replayed = 0
    for dataset in budget.DATASETS:
        bundle = load_dataset(dataset)
        for method in budget.METHODS:
            for run in range(3):
                _, scores = runner.run_api_method(bundle, method, run, prompts, protocol)
                np.testing.assert_array_equal(scores, expected[(dataset, method, run)][bundle.test])
                replayed += 1
    print(json.dumps({"status": "pass", "replayed_run_sets": replayed,
                      "validated_cache_records": len(referenced), "network_requests": 0,
                      "score_match": "exact", "evidence_files_rewritten": 0}, indent=2))


if __name__ == "__main__":
    main()
