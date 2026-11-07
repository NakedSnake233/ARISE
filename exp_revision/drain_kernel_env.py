"""Guarded build and seeded equivalence witness for the local Drain CPU kernel."""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
from types import SimpleNamespace

import numpy as np
from .common import ROOT, json_dump, sha256
from .disk_guard_local import DiskGuard, guarded_command
from . import drain_acceleration as acceleration


def witness(output):
    from drain3.drain import Drain
    from drain3 import TemplateMiner
    from .re2_data import log_records
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    rng = np.random.default_rng(2027); default = Drain(); compiled = acceleration.kernel()
    comparisons = 0
    vocabulary = ["alpha", "beta", "中文", "<*>", "", "123", "β", "a b"]
    for width in (0, 1, 3, 8, 20):
        mapping = {i: SimpleNamespace(cluster_id=i, log_template_tokens=tuple(rng.choice(vocabulary, size=width))) for i in range(100)}
        default.id_to_cluster = mapping
        for _ in range(100):
            ids = rng.permutation(105).tolist(); tokens = rng.choice(vocabulary, size=width).tolist()
            for include in (False, True):
                for threshold in (0., .4, .5, 1.):
                    old = default.fast_match(ids, tokens, threshold, include)
                    new = compiled.fast_match(ids, tokens, mapping, "<*>", threshold, include)
                    if old is not new: raise AssertionError("compiled candidate selection differs")
                    comparisons += 1
    # Fit the whole first training event in its original gzip record order.
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    incident = json.loads((ROOT / "splits/re2_ss_kfold5.json").read_text())["fold_0"]["train"][0]
    raw = ROOT / state["jobs"]["re2_raw_re2_ss_" + incident.split(":")[1]]["output"]
    messages = list(dict.fromkeys(record[2] for record in log_records(raw)))
    results = []; durations = []
    for use_kernel in (False, True):
        start = time.perf_counter()
        with acceleration.accelerated() if use_kernel else contextlib.nullcontext():
            miner = TemplateMiner()
            ids = [miner.add_log_message(message)["cluster_id"] for message in messages]
            templates = [(c.cluster_id, c.log_template_tokens, c.size) for c in miner.drain.clusters]
            matches = [miner.match(messages[i]).cluster_id if miner.match(messages[i]) else None
                for i in np.linspace(0, len(messages)-1, 1000, dtype=int)]
        results.append((ids, templates, matches)); durations.append(time.perf_counter()-start)
    if results[0] != results[1]: raise AssertionError("actual training-event parser outputs differ")
    receipt = dict(status="passed_exact_kernel_witness", seed=2027, synthetic_comparisons=comparisons,
        dataset="re2_ss", training_incident=incident, distinct_messages=len(messages), templates=len(results[0][1]),
        original_seconds=durations[0], accelerated_seconds=durations[1], speedup=durations[0]/durations[1],
        kernel_sha256=sha256(acceleration.artifact()), source_sha256=sha256(ROOT / "exp_revision/drain_kernel.c"),
        spec_sha256=sha256(ROOT / "exp_revision/drain_kernel_spec.json"),
        compared="insertion IDs, final template tokens/sizes, final matches, wildcard ties/candidate order")
    json_dump(output / "witness.json", receipt)
    print(json.dumps(receipt), flush=True)


def guarded(output, command, seconds, create_directory=False):
    key = "drain_kernel_" + output.name; log = ROOT / "logs" / (key + ".log")
    if output.exists() or log.exists(): raise FileExistsError("use a new guarded artifact directory")
    guard = DiskGuard(ROOT)
    env = {**os.environ, **guard.reserve(key, output, log), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    try:
        if create_directory: output.mkdir(parents=True)
        with log.open("x") as stream:
            proc = subprocess.Popen(guarded_command(ROOT, key, ["/usr/bin/timeout", str(seconds)+"s", *command]),
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            guard.attach(key, proc.pid); code = proc.wait()
        if code: raise RuntimeError(f"kernel command failed exit{code}; {log}")
        print("passed", output, flush=True)
    finally: guard.release(key)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--build", action="store_true")
    parser.add_argument("--witness", action="store_true"); parser.add_argument("--kernel-witness", action="store_true")
    parser.add_argument("--output", type=Path); args = parser.parse_args()
    if args.build:
        spec = json.loads((ROOT / "exp_revision/drain_kernel_spec.json").read_text())
        out = ROOT / spec["artifact_directory"]
        command = ["/usr/bin/cc", *spec["flags"], "-I" + sysconfig.get_paths()["include"],
            str(ROOT / spec["source"]), "-o", str(acceleration.artifact())]
        guarded(out, command, 60, create_directory=True)
    elif args.witness and args.output:
        guarded(args.output.resolve(), [sys.executable, "-m", "exp_revision.drain_kernel_env", "--kernel-witness",
            "--output", str(args.output.resolve())], 180)
    elif args.kernel_witness and args.output: witness(args.output)
    else: parser.error("--build or --witness --output required")


if __name__ == "__main__": main()
