"""Keep original transport/algorithm/masks and label the approved input variant."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from .common import ROOT, sha256
from . import official_baselines as baseline


def run(dataset, fold, seed, output, smoke=False):
    approval = ROOT / "docs/circa_preprocessing_authorization.json"
    if not approval.exists() or json.loads(approval.read_text()).get("approved") is not True:
        raise ValueError("author approval required for CIRCA input variant")
    original_run = baseline.subprocess.run
    original_dump = baseline.json_dump

    def subprocess_run(command, *args, **kwargs):
        if isinstance(command, list) and "exp_revision.rcaeval_worker" in command:
            command = list(command); command[command.index("exp_revision.rcaeval_worker")] = "exp_revision.rcaeval_rank_worker"
        return original_run(command, *args, **kwargs)

    def dump(path, value):
        if path == output / "run_meta.json":
            value = {**value, "config": {**value["config"], "method": "CIRCA+rank-prune"},
                "input_policy": value["input_policy"] + "; author-approved CIRCA+rank-prune policy",
                "official_algorithm_changed": False,
                "preprocessing_policy_sha256": sha256(ROOT / "exp_revision/circa_rank_prune_policy.json"),
                "author_approval_sha256": sha256(approval)}
            value["config_sha256"] = hashlib.sha256(json.dumps(value["config"], sort_keys=True).encode()).hexdigest()
            value["command"] = "python -m exp_revision.circa_rank_prune"
        return original_dump(path, value)

    baseline.subprocess.run = subprocess_run
    baseline.json_dump = dump
    try: baseline.run(dataset, fold, seed, "CIRCA", output, smoke)
    finally:
        baseline.subprocess.run = original_run
        baseline.json_dump = original_dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    run(args.dataset, args.fold, args.seed, args.output, args.smoke)


if __name__ == "__main__": main()
