"""Call the pinned official algorithms in their own runtimes (Python 3.8+).

NPZ transport contains only metric observations, times and a service catalogue.
The upstream package registry imports unrelated optional algorithms. Load its
original decorator and algorithm modules without executing that registry.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import importlib
import io
import json
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/rcaeval_259ea416"


def official_function(method):
    sys.path.insert(0, str(SOURCE))
    import RCAEval
    if "RCAEval.e2e" not in sys.modules:
        package = types.ModuleType("RCAEval.e2e")
        package.__path__ = [str(SOURCE / "RCAEval/e2e")]
        package.__package__ = "RCAEval.e2e"
        package.__file__ = str(SOURCE / "RCAEval/e2e/__init__.py")
        package.sys = sys
        parsed = ast.parse(Path(package.__file__).read_text())
        decorator = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "rca")
        code = ast.Module(body=[decorator], type_ignores=[])
        exec(compile(code, package.__file__, "exec"), package.__dict__)
        sys.modules[package.__name__] = package
        RCAEval.e2e = package
    name = {"BARO": "baro", "CIRCA": "circa", "RCD": "rcd"}[method]
    return getattr(importlib.import_module("RCAEval.e2e." + name), name)


def service_scores(ranks, columns, column_services, services):
    mapping = dict(zip(columns, column_services))
    best = {}
    for position, node in enumerate(ranks):
        service = mapping.get(str(node))
        if service in services and service not in best:
            best[service] = position
    # One service vote from its best metric; stable catalogue order breaks ties.
    return np.asarray([1. / (1. + best[s]) if s in best else 0. for s in services], dtype=float)


def infer(method, values, times, columns, column_services, services, inject_time, seed, dataset="generic"):
    np.random.seed(seed); random.seed(seed)
    frame = pd.DataFrame(np.asarray(values, dtype=float), columns=columns)
    frame.insert(0, "time", times)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    # No labels, selection on the original observed metric series only.
    valid = [c for c in columns if frame[c].notna().all() and frame[c].nunique() > 1]
    if not valid:
        return np.zeros(len(services)), dict(status="no_observed_varying_metrics", metric_ranks=[])
    frame = frame[["time"] + valid]
    stderr, stdout = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
        result = official_function(method)(frame.copy(), inject_time, dataset=dataset, seed=seed)
    diagnostic = stderr.getvalue()
    if "returning the unranked column order" in diagnostic:
        raise RuntimeError(diagnostic.strip()[-2000:])
    if not isinstance(result, dict) or "ranks" not in result:
        raise ValueError("official algorithm did not return metric ranks")
    ranks = [str(v) for v in result["ranks"]]
    if any(v not in valid for v in ranks):
        raise ValueError("official metric ranking contains an unknown metric")
    scores = service_scores(ranks, columns, column_services, services)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite service ranking")
    return scores, dict(status="completed_official", metric_ranks=ranks,
                       diagnostic=diagnostic[-2000:], stdout_tail=stdout.getvalue()[-1000:])


def witness(method, output):
    rng = np.random.default_rng(2027)
    n = 120
    x = rng.normal(size=(n, 6)); x[n // 2:, 0] += 5.
    columns = [s + "_" + c for s in ("service-a", "service-b", "service-c") for c in ("cpu", "mem")]
    services = ["service-a", "service-b", "service-c"]
    start = time.perf_counter()
    scores, receipt = infer(method, x, np.arange(n), columns,
                            [s for s in services for _ in range(2)], services, n // 2, 42)
    receipt.update(method=method, scores=scores.tolist(), elapsed_seconds=time.perf_counter()-start,
                   code_commit="259ea4167160256a74ad30004aa3d96a998c846e",
                   python=sys.version, localized_pc=method == "RCD", seed=42)
    if scores.shape != (3,) or not scores.any():
        raise ValueError("algorithm witness has no complete finite service output")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("BARO", "CIRCA", "RCD"), required=True)
    parser.add_argument("--witness", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.witness:
        witness(args.method, args.output)
    elif args.input:
        spec=json.loads(args.input.read_text()); services=spec["services"]; results=[]; cache={}
        for task in spec["tasks"]:
            clock=time.perf_counter()
            cache_key=(task["path"],tuple(task["metric_keep"]),task["inject_time"])
            try:
                if cache_key in cache:
                    receipt=dict(cache[cache_key]); receipt["reused_identical_metric_view"]=True
                else:
                    with np.load(task["path"]) as a:
                        columns=a["columns"].tolist(); column_services=a["column_services"].tolist()
                        times=a["times"].copy(); values=a["values"].copy()
                    keep=np.asarray(task["metric_keep"],dtype=bool)
                    selected=[i for i,s in enumerate(column_services) if s in services and keep[services.index(s)]]
                    scores,receipt=infer(args.method,values[:,selected],times,[columns[i] for i in selected],
                        [column_services[i] for i in selected],services,task["inject_time"],spec["seed"],spec["official_dataset"])
                    receipt.update(scores=scores.tolist(),failed=False)
                    cache[cache_key]=dict(receipt)
            except Exception as error:
                receipt=dict(scores=[0.]*len(services),failed=True,status="algorithm_failed",error=repr(error))
            receipt.update(incident_id=task["incident_id"],scenario=task["scenario"],elapsed_seconds=time.perf_counter()-clock)
            results.append(receipt)
            print(json.dumps({k:receipt[k] for k in ("incident_id","scenario","status","elapsed_seconds")}),flush=True)
        if args.output.exists():raise FileExistsError(args.output)
        args.output.write_text(json.dumps(results,indent=2)+"\n")
    else: parser.error("--input or --witness required")


if __name__ == "__main__": main()
