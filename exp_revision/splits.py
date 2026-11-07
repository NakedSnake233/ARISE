from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .common import json_dump


UNLABELED_FIELDS = ("incident_id", "start", "end", "source", "overlap_group")


def write_unlabeled_manifest(bundle, output: Path) -> None:
    """Freeze event locations without exposing held-out labels to preprocessing."""
    rows = [{key: row[key] for key in UNLABELED_FIELDS} for row in bundle.rows]
    if len({row["incident_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate incident IDs")
    json_dump(output, rows)


def write_training_label_manifests(bundle, split: dict, output_dir: Path) -> None:
    """Expose only each fold's training labels to future model adapters."""
    lookup = {row["incident_id"]: row for row in bundle.rows}
    for fold_name, part in split.items():
        if set(part["train"]) & set(part["test"]):
            raise ValueError(f"overlapping train/test IDs in {fold_name}")
        rows = [
            {"incident_id": incident_id, "root_service": lookup[incident_id]["root_service"],
             "fault_type": lookup[incident_id]["fault_type"],
             "overlap_group": lookup[incident_id]["overlap_group"]}
            for incident_id in part["train"]
        ]
        json_dump(output_dir / f"{fold_name}.json", rows)


def make_kfold(bundle, output: Path, n_splits: int = 5, seed: int = 2027) -> dict:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(bundle.rows))
    y = np.asarray([r["fault_type"] for r in bundle.rows])
    result = {}
    for fold, (train, test) in enumerate(splitter.split(indices, y, bundle.groups)):
        result[f"fold_{fold}"] = {
            "train": [bundle.rows[i]["incident_id"] for i in train],
            "test": [bundle.rows[i]["incident_id"] for i in test],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def load_or_make(bundle, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return make_kfold(bundle, path)


def ids_to_indices(bundle, ids: list[str]) -> np.ndarray:
    mapping = {r["incident_id"]: i for i, r in enumerate(bundle.rows)}
    return np.asarray([mapping[x] for x in ids], dtype=int)


def check_splits(bundle, split: dict) -> None:
    all_tests = []
    for part in split.values():
        train, test = set(part["train"]), set(part["test"])
        if train & test:
            raise AssertionError("train/test overlap")
        all_tests.extend(test)
        groups = {r["incident_id"]: str(r.get("overlap_group", r["incident_id"])) for r in bundle.rows}
        if {groups[x] for x in train} & {groups[x] for x in test}:
            raise AssertionError("group crosses fold")
    if len(all_tests) != len(set(all_tests)) or set(all_tests) != {r["incident_id"] for r in bundle.rows}:
        raise AssertionError("test folds are not a partition")
