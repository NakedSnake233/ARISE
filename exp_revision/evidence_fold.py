"""Build label-free, fold-aligned S0 evidence tensors for the original experts.

Contrasts use each fold's train-fitted telemetry tensors. TrainTicket raw KPI
contrasts and the existing per-event RPC summaries have no fitted statistics;
their original incident order is the frozen label-free event manifest order.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .common import ROOT, json_dump, sha256
from .fold_data import load_fold

sys.path.insert(0, str(ROOT / "scripts/care"))
from evidence_reranker import extract, raw_tt_metrics  # noqa: E402


RPC_PATHS = {
    "tt": ROOT / "artifacts/care/trace-residual-trainticket",
    "aiops": ROOT / "artifacts/care/full-v3/aiops2022/rpc",
}


def prepare_tt_raw_metric() -> Path:
    path = ROOT / "cache/raw_metric_tt.npz"
    if path.exists():
        with np.load(path) as archive:
            if set(archive.files) != {"metric", "incident_ids"}:
                raise ValueError("incomplete raw TrainTicket metric cache")
        return path
    rows = json.loads((ROOT / "splits/tt_unlabeled_events.json").read_text())
    metadata = json.loads((ROOT / "data/care/trainticket-prepared/metadata.json").read_text())
    bins = int((metadata["before"] + metadata["after"]) // metadata["step"])
    metric = raw_tt_metrics(rows, metadata["services"], metadata["metric_names"],
                            int(metadata["step"]), bins).astype(np.float32)
    if metric.shape[:2] != (len(rows), len(metadata["services"])):
        raise ValueError("raw TrainTicket metric cache shape mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, metric=metric,
                        incident_ids=np.asarray([row["incident_id"] for row in rows]))
    temporary.replace(path)
    return path


def prepare_fold(dataset: str, fold: int) -> Path:
    started = time.perf_counter()
    data = load_fold(dataset, fold, smoke=False)
    all_ids = data.train_ids + data.test_ids
    unlabeled_path = ROOT / f"splits/{dataset}_unlabeled_events.json"
    unlabeled = json.loads(unlabeled_path.read_text())
    original_ids = [row["incident_id"] for row in unlabeled]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("duplicate original incident ID")
    original_index = {incident_id: i for i, incident_id in enumerate(original_ids)}
    if set(all_ids) != set(original_ids):
        raise ValueError("fold events differ from the frozen label-free cohort")
    alignment_path = ROOT / f"cache/rpc_alignment/{dataset}.json"
    alignment = json.loads(alignment_path.read_text())
    if alignment["incident_ids"] != original_ids or alignment["services"] != data.services:
        raise ValueError("legacy RPC incident or service order differs from frozen fold")
    for relative, digest in alignment["input_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"RPC identity audit source changed: {relative}")
    rows = [{"incident_id": incident_id, "calls": data.metadata["calls"][incident_id]}
            for incident_id in all_ids]
    raw = None
    inputs = [unlabeled_path, alignment_path,
              ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}/features.npz",
              ROOT / f"cache/fold_preprocessing/{dataset}/fold{fold}/metadata.json",
              ROOT / "scripts/care/evidence_reranker.py",
              ROOT / "exp_revision/evidence_fold.py"]
    if dataset == "tt":
        raw_path = prepare_tt_raw_metric()
        with np.load(raw_path) as archive:
            if archive["incident_ids"].tolist() != original_ids:
                raise ValueError("raw TrainTicket metric order mismatch")
            raw = archive["metric"][[original_index[x] for x in all_ids]]
        inputs.append(raw_path)
    x, names = extract({"metrics": data.metrics, "traces": data.traces,
                        "logs": data.logs}, data.metadata, rows, raw)
    rpc_path = RPC_PATHS[dataset]
    rpc_names = json.loads((rpc_path / "feature_names.json").read_text())
    with np.load(rpc_path / "features.npz") as archive:
        rpc = archive["x"][[original_index[x] for x in all_ids]]
    if rpc.shape[:2] != x.shape[:2] or rpc.shape[-1] != len(rpc_names):
        raise ValueError("RPC evidence shape mismatch")
    inputs.extend((rpc_path / "feature_names.json", rpc_path / "features.npz"))
    features = np.concatenate([x, rpc], axis=-1).astype(np.float32)
    names += rpc_names
    if not np.isfinite(features).all():
        raise ValueError("nonfinite evidence feature")
    destination = ROOT / f"cache/fold_evidence_v2/{dataset}/fold{fold}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evidence fold: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.mkdir()
    np.savez_compressed(temporary / "features.npz", x=features,
                        incident_ids=np.asarray(all_ids))
    json_dump(temporary / "metadata.json", {
        "status": "S0_fold_aligned_evidence", "dataset": dataset, "fold": fold,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "train_ids": data.train_ids, "test_ids": data.test_ids,
        "feature_names": names, "shape": list(features.shape),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "limitations": ["RPC summaries are existing label-free per-event extracts; "
                        "masked-trace scenarios require a separate raw-span recomputation."],
    })
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    print(prepare_fold(args.dataset, args.fold))


if __name__ == "__main__":
    main()
