"""Load a preprocessed outer fold without exposing its held-out labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FoldData:
    dataset: str
    fold: int
    services: list[str]
    train_ids: list[str]
    test_ids: list[str]
    train_labels: np.ndarray
    train_fault_types: list[str]
    train_groups: list[object]
    metrics: np.ndarray
    logs: np.ndarray
    traces: np.ndarray
    availability: np.ndarray
    adjacency: np.ndarray
    metadata: dict

    @property
    def train_indices(self) -> np.ndarray:
        return np.arange(len(self.train_ids))

    @property
    def test_indices(self) -> np.ndarray:
        return np.arange(len(self.train_ids), len(self.train_ids) + len(self.test_ids))


def load_fold(dataset: str, fold: int, *, smoke: bool = False,
              cache_root: Path = Path("cache/fold_preprocessing"),
              splits_root: Path = Path("splits")) -> FoldData:
    if dataset not in ("tt", "aiops") or fold not in range(5):
        raise ValueError("unsupported dataset or fold")
    folder = cache_root / dataset / f"fold{fold}{'_smoke' if smoke else ''}"
    metadata = json.loads((folder / "metadata.json").read_text())
    manifest = json.loads((splits_root / f"{dataset}_kfold5.json").read_text())[f"fold_{fold}"]
    train_ids = list(metadata["train_ids"])
    test_ids = list(metadata["test_ids"])
    if train_ids != manifest["train"] or not test_ids or not set(test_ids).issubset(manifest["test"]):
        raise ValueError("preprocessed fold does not match the frozen split")
    if set(train_ids) & set(test_ids):
        raise ValueError("train/test event overlap")
    if not smoke and test_ids != manifest["test"]:
        raise ValueError("full fold cache is missing test incidents")
    training_rows = json.loads((splits_root / f"{dataset}_training_labels" / f"fold_{fold}.json").read_text())
    if [row["incident_id"] for row in training_rows] != train_ids:
        raise ValueError("training label manifest must contain training events only and in fold order")
    services = list(metadata["services"])
    service_to_index = {service: index for index, service in enumerate(services)}
    labels = np.asarray([service_to_index[row["root_service"]] for row in training_rows], dtype=np.int64)
    with np.load(folder / "features.npz") as archive:
        expected = {"metrics", "logs", "traces", "availability"}
        if set(archive.files) != expected:
            raise ValueError("fold features contain unexpected fields, possibly labels")
        arrays = {key: np.asarray(archive[key]) for key in expected}
    n = len(train_ids) + len(test_ids)
    for key in ("metrics", "logs", "traces"):
        array = arrays[key]
        if array.shape[:2] != (n, len(services)) or array.ndim != 4 or not np.isfinite(array).all():
            raise ValueError(f"invalid {key} fold tensor")
    availability = arrays["availability"]
    if availability.shape != (n, len(services), 3) or not np.isin(availability, (0, 1)).all():
        raise ValueError("invalid raw availability matrix")
    adjacency = np.eye(len(services), dtype=bool)
    for source, target in metadata["training_graph_edges"]:
        u, v = service_to_index[source], service_to_index[target]
        adjacency[u, v] = adjacency[v, u] = True
    return FoldData(dataset, fold, services, train_ids, test_ids, labels,
                    [row["fault_type"] for row in training_rows],
                    [row["overlap_group"] for row in training_rows],
                    arrays["metrics"], arrays["logs"], arrays["traces"], availability,
                    adjacency, metadata)
