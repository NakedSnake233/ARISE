"""Identity-only audit of legacy RPC arrays before fold-aligned evidence use.

This preparation command accesses incident IDs and service order, never root
or fault-type fields. Training consumes only the resulting identity receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import ROOT, json_dump, sha256


def audit(dataset: str) -> Path:
    prepared = ROOT / ("data/care/trainticket-prepared" if dataset == "tt"
                       else "data/care/aiops2022-paper128-prepared")
    rpc = ROOT / ("artifacts/care/trace-residual-trainticket" if dataset == "tt"
                  else "artifacts/care/full-v3/aiops2022/rpc")
    source_rows = json.loads((prepared / "incidents.json").read_text())
    source_ids = [row["incident_id"] for row in source_rows]
    frozen_path = ROOT / f"splits/{dataset}_unlabeled_events.json"
    frozen_ids = [row["incident_id"] for row in json.loads(frozen_path.read_text())]
    services = json.loads((prepared / "metadata.json").read_text())["services"]
    with np.load(rpc / "features.npz") as archive:
        shape = list(archive["x"].shape)
    names = json.loads((rpc / "feature_names.json").read_text())
    if source_ids != frozen_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("legacy RPC incident order differs from the frozen label-free order")
    if shape != [len(source_ids), len(services), len(names)]:
        raise ValueError("legacy RPC service or feature dimension mismatch")
    inputs = [prepared / "incidents.json", prepared / "metadata.json",
              rpc / "features.npz", rpc / "feature_names.json", frozen_path]
    receipt = {"dataset": dataset, "incident_ids": source_ids,
               "services": services, "rpc_shape": shape,
               "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs}}
    destination = ROOT / f"cache/rpc_alignment/{dataset}.json"
    if destination.exists():
        if json.loads(destination.read_text()) != receipt:
            raise FileExistsError("RPC identity receipt differs from the existing audit")
    else:
        json_dump(destination, receipt)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    args = parser.parse_args()
    print(audit(args.dataset))


if __name__ == "__main__":
    main()
