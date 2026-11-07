"""Freeze group-isolated TrainTicket checkpoint validation inside outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .common import json_dump


def build(splits_root: Path = Path("splits"), seed: int = 2027) -> dict:
    outer = json.loads((splits_root / "tt_kfold5.json").read_text())
    result = {}
    for fold_name, part in sorted(outer.items()):
        labels = json.loads((splits_root / "tt_training_labels" / f"{fold_name}.json").read_text())
        ids = [row["incident_id"] for row in labels]
        if ids != part["train"]:
            raise ValueError(f"training labels do not match {fold_name}")
        groups = np.asarray([str(row["overlap_group"]) for row in labels])
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        fit, validation = next(splitter.split(np.arange(len(ids)), groups=groups))
        if set(groups[fit]) & set(groups[validation]):
            raise AssertionError("inner validation group crosses fit split")
        result[fold_name] = {
            "fit": [ids[i] for i in fit],
            "validation": [ids[i] for i in validation],
        }
    return result


def freeze(path: Path = Path("splits/tt_inner_validation_kfold5.json")) -> dict:
    result = build(path.parent)
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError(f"frozen inner validation has changed: {path}")
    else:
        json_dump(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("splits/tt_inner_validation_kfold5.json"))
    args = parser.parse_args()
    result = freeze(args.output)
    print({key: {part: len(ids) for part, ids in value.items()} for key, value in result.items()})


if __name__ == "__main__":
    main()
