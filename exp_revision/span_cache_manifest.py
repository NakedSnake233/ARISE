"""Register bounded raw-span cache jobs in the local guarded queue."""
import argparse
import json
import subprocess
from pathlib import Path

from .common import ROOT, json_dump, sha256
from .fold_data import load_fold
from .remaining_queue import PRECHARGED, verified


def create(smoke=False):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = [ROOT / name for name in (
        "exp_revision/span_cache_manifest.py", "exp_revision/raw_spans_v2.py",
        "exp_revision/remaining_queue.py", "exp_revision/common.py", "exp_revision/fold_data.py",
        "exp_revision/disk_guard_local.py", "exp_revision/disk_guard.py",
        "scripts/care/trace_residual.py", "scripts/care/prepare_aiops_rpc.py", "scripts/care/prepare_aiops.py")]
    jobs = []
    for dataset in ("tt", "aiops"):
        manifest = ROOT / f"splits/{dataset}_unlabeled_events.json"
        rows = json.loads(manifest.read_text()); by_id = {r["incident_id"]: r for r in rows}
        ids = load_fold(dataset, 0).test_ids[:1] if smoke else list(by_id)
        if dataset == "tt": chunks = [ids[i:i + 10] for i in range(0, len(ids), 10)]
        else: chunks = [[key for key in ids if key.split(":")[0] == cloud] for cloud in sorted({key.split(":")[0] for key in ids})]
        metadata = ROOT / ("data/care/trainticket-prepared/metadata.json" if dataset == "tt"
                           else "data/care/aiops2022-paper128-prepared/metadata.json")
        services = json.loads(metadata.read_text())["services"]
        sources.extend([manifest, metadata, ROOT / ("artifacts/care/trace-residual-trainticket/features.npz"
            if dataset == "tt" else "artifacts/care/full-v3/aiops2022/rpc/features.npz")])
        for i, chunk in enumerate(chunks):
            if dataset == "tt": sources.extend(ROOT / Path(by_id[key]["source"]).parent / "traces.jsonl.gz" for key in chunk)
            else:
                sources.append(ROOT / "data/care/aiops2022-original/training_data_with_faults/training_data_with_faults/tar"
                               / chunk[0].split(":")[0] / "trace/all/trace_jaeger-span.csv")
            key = f"{'smoke' if smoke else 'full'}_rpc_cache_{dataset}_chunk{i}"
            jobs.append({"id": key, "source_commit": commit, "resource": "cpu", "cores": 1,
                "timeout_seconds": 600, "max_attempts": 2,
                "command": ["-m", "exp_revision.raw_spans_v2", "--dataset", dataset, "--incident-ids", json.dumps(chunk)],
                "expected_config": {"task": "raw_span_preprocessing", "dataset": dataset, "incident_ids": chunk},
                "expected_status": ["completed_span_cache"], "test_ids": chunk,
                "score_shape": [1, len(chunk), len(services)]})
    if not smoke:
        state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
        for dataset in ("tt", "aiops"):
            row = state["jobs"][f"smoke_rpc_cache_{dataset}_chunk0"]
            if row["status"] != "completed" or not verified(ROOT / row["output"], row["spec"]):
                raise ValueError("both raw-span S0 parity smoke jobs must pass")
    manifest = {"backend": "local", "phase": "smoke_span_cache" if smoke else "full_span_cache",
        "source_commit": commit, "precharged_hours": PRECHARGED,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in dict.fromkeys(sources)}, "jobs": jobs}
    path = ROOT / f"manifests/remaining/{'03_smoke_span_cache' if smoke else '04_full_span_cache'}.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest: raise FileExistsError("immutable span manifest differs")
    else: json_dump(path, manifest)
    return path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); print(create(a.smoke))


if __name__ == "__main__": main()
