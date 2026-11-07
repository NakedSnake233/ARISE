"""Collect completed local TT outputs using the reviewed aggregators."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from .common import ROOT, config_snapshot, json_dump
from .local_monitor import verify_sources


def collect(phase: str, config_path: Path, receipt: Path) -> dict:
    if receipt.exists():
        raise FileExistsError("immutable collection receipt already exists")
    config = json.loads(config_path.read_text())
    verify_sources(config)
    started = time.perf_counter()
    meta = config_snapshot(f"python -m exp_revision.local_collect --phase {phase}",
                           {"phase": phase, "dataset": "tt", "backend": "local",
                            "context_root": "results_v2/T1_context_local_full"})
    meta.update(start_time=dt.datetime.now(dt.timezone.utc).isoformat(), phase=phase)
    try:
        if phase == "context":
            from .aggregate_context import aggregate
            result = aggregate("tt", ROOT / "manifests/tt_context_full.json",
                               ROOT / "results_v2/T1_context_local_full")
            prediction_path = "results_v2/T1_context_local_full/tt/predictions.parquet"
        elif phase == "arise":
            from .aggregate_arise import aggregate
            result = aggregate("tt", ROOT / "manifests/tt_context_full.json",
                               ROOT / "manifests/tt_evidence_full.json",
                               ROOT / "results_v2/T1_context_local_full")
            prediction_path = "results_v2/T1_arise_s0/tt/predictions.parquet"
        else:
            raise ValueError("unknown collection phase")
        verify_sources(config)
        meta.update(status="completed", prediction_path=prediction_path,
                    predictions_sha256=result["predictions_sha256"], result=result)
    except Exception as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        meta.update(end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started)
        json_dump(receipt, meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("context", "arise"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.phase, args.config, args.receipt)))


if __name__ == "__main__":
    main()
