from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "care"
RESULT_ROOT = ROOT / "results_v2"
SEEDS = [42, 43, 44, 45, 46]
METRICS = ("PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5", "MRR")


def stable_seed(*parts: object) -> int:
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_snapshot(command: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    commit = os.environ.get("ARISE_SOURCE_COMMIT")
    if commit is None:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                             stderr=subprocess.DEVNULL).strip()
        except Exception:
            commit = "unavailable"
    dirty_override = os.environ.get("ARISE_SOURCE_DIRTY")
    if dirty_override is not None:
        dirty = dirty_override.lower() in ("1", "true", "yes")
    else:
        try:
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                                                 stderr=subprocess.DEVNULL).strip())
        except Exception:
            dirty = True
    package_versions = {}
    for package in ("numpy", "scikit-learn", "torch", "scipy", "pandas", "pyarrow", "drain3"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    cpu = platform.processor()
    if not cpu:
        try:
            cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        except (OSError, StopIteration):
            cpu = "unknown"
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "config": config or {},
        "config_sha256": hashlib.sha256(json.dumps(config or {}, sort_keys=True).encode()).hexdigest(),
        "python_version": sys.version,
        "packages": package_versions,
        "cpu_model": cpu,
        "gpu_model": "not queried (CPU-safe revision runner)",
        "command": command,
    }


@dataclass
class Bundle:
    dataset: str
    data_dir: Path
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    features: dict[str, np.ndarray]
    services: list[str]
    labels: np.ndarray
    before: int
    groups: np.ndarray
    availability: np.ndarray

    @property
    def n_services(self) -> int:
        return len(self.services)


def dataset_dir(dataset: str) -> Path:
    names = {"tt": "trainticket-prepared", "aiops": "aiops2022-paper128-prepared"}
    if dataset not in names:
        raise ValueError(f"unknown dataset: {dataset}")
    return DATA_ROOT / names[dataset]


def _availability(features: dict[str, np.ndarray]) -> np.ndarray:
    # Prepared arrays intentionally encode absent telemetry as all-zero slices.
    # The check is modality-local and does not use root/fault labels.
    metric = np.any(np.abs(np.nan_to_num(features["metrics"])) > 1e-12, axis=(2, 3))
    log = np.any(np.abs(np.nan_to_num(features["logs"])) > 1e-12, axis=(2, 3))
    trace = np.any(np.abs(np.nan_to_num(features["traces"])) > 1e-12, axis=(2, 3))
    return np.stack([metric, log, trace], axis=-1).astype(np.uint8)


def load_bundle(dataset: str) -> Bundle:
    data_dir = dataset_dir(dataset)
    rows = json.loads((data_dir / "incidents.json").read_text())
    metadata = json.loads((data_dir / "metadata.json").read_text())
    archive = np.load(data_dir / "features.npz")
    features = {key: np.asarray(archive[key]) for key in archive.files}
    labels = features["labels"].astype(int)
    before = int(round(float(metadata["before"]) / float(metadata["step"])))
    groups = np.asarray([str(r.get("overlap_group", r["incident_id"])) for r in rows])
    availability = _availability(features)
    if len(rows) != len(labels):
        raise ValueError("incident and label counts differ")
    return Bundle(dataset, data_dir, rows, metadata, features, list(metadata["services"]), labels, before, groups, availability)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write genuine parquet; never disguise JSONL with a parquet suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_parquet(path, index=False)


def read_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd
    return pd.read_parquet(path).to_dict("records")
