"""Seeded equality and actual-log witness for the warm-env bitset accelerator."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

from .common import ROOT, json_dump, sha256
from .drain_kernel_env import guarded
from . import drain_bitset_compact as drain_bitset


def witness(output):
    from drain3.drain import Drain
    from drain3 import TemplateMiner
    from .re2_data import log_records
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    rng = np.random.default_rng(2027); original = Drain.fast_match; comparisons = 0
    for width in (0, 1, 3, 8, 20):
        mapping = {i: SimpleNamespace(cluster_id=i, log_template_tokens=tuple(rng.choice(["a", "b", "中文", "<*>", "β", "123"], size=width))) for i in range(100)}
        default = Drain(); default.id_to_cluster = mapping
        for _ in range(100):
            ids = rng.permutation(105).tolist(); tokens = rng.choice(["a", "b", "中文", "<*>", "β", "123"], size=width).tolist()
            for include in (False, True):
                for threshold in (0., .4, .5, 1.):
                    expected = original(default, ids, tokens, threshold, include)
                    with drain_bitset.accelerated(): actual = default.fast_match(ids, tokens, threshold, include)
                    if expected is not actual: raise AssertionError("synthetic candidate/tie mismatch")
                    comparisons += 1
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    incident = json.loads((ROOT / "splits/re2_ss_kfold5.json").read_text())["fold_0"]["train"][0]
    raw = ROOT / state["jobs"]["re2_raw_re2_ss_" + incident.split(":")[1]]["output"]
    messages = list(dict.fromkeys(record[2] for record in log_records(raw)))
    results = []; durations = []
    def tree(node):
        return (tuple(node.cluster_ids), [(key, tree(child)) for key, child in node.key_to_child_node.items()])
    for enabled in (False, True):
        start = time.perf_counter()
        with drain_bitset.accelerated() if enabled else contextlib.nullcontext():
            miner = TemplateMiner(); ids = [miner.add_log_message(message)["cluster_id"] for message in messages]
            templates = [(c.cluster_id, c.log_template_tokens, c.size) for c in miner.drain.clusters]
            matches = []
            for i in np.linspace(0, len(messages)-1, 1000, dtype=int):
                match = miner.match(messages[i]); matches.append(match.cluster_id if match else None)
            structure = tree(miner.drain.root_node)
        results.append((ids, templates, matches, structure)); durations.append(time.perf_counter()-start)
    if results[0] != results[1]: raise AssertionError("actual full training-event outputs/tree differ")
    receipt = dict(status="passed_exact_bitset_witness", seed=2027, comparisons=comparisons, distinct_messages=len(messages),
        templates=len(results[0][1]), original_seconds=durations[0], accelerated_seconds=durations[1], speedup=durations[0]/durations[1],
        source_sha256=sha256(ROOT / "exp_revision/drain_bitset_compact.py"), spec_sha256=sha256(ROOT / "exp_revision/drain_bitset_compact_spec.json"),
        compared="every insertion ID, final tokens/sizes,1000matches, exact tree paths and candidate order")
    json_dump(output / "witness.json", receipt); print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--inside", action="store_true")
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.inside: witness(args.output)
    else: guarded(args.output.resolve(), [sys.executable, "-m", "exp_revision.drain_bitset_compact_witness", "--inside", "--output", str(args.output.resolve())], 180)


if __name__ == "__main__": main()
