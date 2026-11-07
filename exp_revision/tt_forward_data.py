"""Frozen self-TT forward cohort with original train-only preprocessing.

One prespecified chronological cut: earliest 90 training events, latest 60
evaluation events. This remains posthoc validation within one collection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, dataset_dir, json_dump, sha256
from . import generalization_data as gd
from .splits import UNLABELED_FIELDS


TASK = "TFWD"
SPLIT = ROOT / "splits/tt_forward_90_60.json"
FOLDER = ROOT / "results_v2/TT_forward_correction"
PREPARE_JOB_ID = "forward_prepare"


def split_file(dataset, task):
    if (dataset, task) != ("tt", TASK):
        raise ValueError("forward adapter supports only self-collected TT/TFWD")
    return SPLIT


def training_file(dataset, task, fold):
    split_file(dataset, task)
    if fold != 0:
        raise ValueError("forward protocol has one fixed split, fold 0")
    return ROOT / "splits/tt_forward_training_labels/fold_0.json"


def make_cohorts(events, training_labels):
    """Only event times determine outer membership; labels group outer training."""
    if len(events) != 150 or len({row["incident_id"] for row in events}) != 150:
        raise ValueError("expected the 150 unique frozen self-TT events")
    if any(set(row) != set(UNLABELED_FIELDS) for row in events):
        raise ValueError("chronological split input must be the unlabeled event manifest")
    ordered = sorted(events, key=lambda row: (row["start"], row["end"], row["incident_id"]))
    train, test = ordered[:90], ordered[90:]
    if max(row["end"] for row in train) > min(row["start"] for row in test):
        raise ValueError("forward observation intervals cross the fixed train/test cut")
    if {str(row["overlap_group"]) for row in train} & {str(row["overlap_group"]) for row in test}:
        raise ValueError("original overlap group crosses the chronological cut")
    train_ids = [row["incident_id"] for row in train]
    if set(training_labels) != set(train_ids):
        raise ValueError("only the 90 outer-training labels may define inner groups")
    pairs = {key: (training_labels[key]["root_service"], training_labels[key]["fault_type"]) for key in train_ids}
    opaque = {pair: f"pair_{i:02d}" for i, pair in enumerate(sorted(set(pairs.values())))}
    groups = {key: opaque[pairs[key]] for key in train_ids}
    inner = gd.inner_partition([dict(incident_id=key, overlap_group=groups[key]) for key in train_ids],
                               .1, string_groups=True)
    return {"fold_0": dict(train=train_ids, test=[row["incident_id"] for row in test], purged=[],
                           split="chronological_90_60", context_inner=inner, baseline_inner=inner,
                           train_pair_groups=groups)}


def freeze():
    event_path = ROOT / "splits/tt_unlabeled_events.json"
    label_path = dataset_dir("tt") / "incidents.json"
    events = json.loads(event_path.read_text())
    train_ids = {row["incident_id"] for row in sorted(events, key=lambda row: (row["start"], row["end"], row["incident_id"]))[:90]}
    # Evaluation labels are not consulted for cut selection or grouping.
    labels = {row["incident_id"]: row for row in json.loads(label_path.read_text()) if row["incident_id"] in train_ids}
    parts = make_cohorts(events, labels)
    part = parts["fold_0"]
    training = [dict(incident_id=key, root_service=labels[key]["root_service"], fault_type=labels[key]["fault_type"],
                     overlap_group=part["train_pair_groups"][key]) for key in part["train"]]
    gd.immutable_json(SPLIT, parts)
    gd.immutable_json(training_file("tt", TASK, 0), training)
    lookup = {row["incident_id"]: row for row in events}
    receipt = dict(status="frozen_chronological_cohort", dataset="tt", task=TASK, fold=0,
                   order=["start", "end", "incident_id"], train=90, test=60,
                   train_last_end=max(lookup[key]["end"] for key in part["train"]),
                   test_first_start=min(lookup[key]["start"] for key in part["test"]),
                   inner_fit=len(part["context_inner"]["fit"]), inner_validation=len(part["context_inner"]["validation"]),
                   inner_rule="context/adapters: pair-grouped GroupShuffleSplit .1 random_state=2027; experts: unchanged training-only fault-type holdout beta selection",
                   input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (event_path, label_path)},
                   split_sha256=sha256(SPLIT), training_labels_sha256=sha256(training_file("tt", TASK, 0)),
                   interpretation="fixed posthoc forward validation within the same randomized collection; not outer pair isolation or an independent new test dataset")
    gd.immutable_json(FOLDER / "split_receipt.json", receipt)
    return receipt


def prepared_folder(dataset, task, fold, smoke=False):
    training_file(dataset, task, fold)
    if smoke:
        raise ValueError("forward protocol never truncates evaluation events")
    state = json.loads((FOLDER / "queue_state.json").read_text())
    row = state["jobs"][PREPARE_JOB_ID]
    if row["status"] != "completed":
        raise ValueError("forward preprocessing incomplete")
    return ROOT / row["output"]


def activate():
    gd.split_file = split_file
    gd.training_file = training_file
    gd.prepared_folder = prepared_folder


def prepare(fold, output):
    training_file("tt", TASK, fold)
    activate()
    gd.prepare("tt", TASK, fold, output, smoke=False)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["preprocessing_scope"] = "Drain/IDF/channel selection/scalers/graph fitted on the 90 chronological outer-training events only"
    json_dump(metadata_path, metadata)
    path = output / "run_meta.json"
    meta = json.loads(path.read_text())
    meta["metadata_sha256"] = sha256(metadata_path)
    meta["source_sha256"][str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    meta["protocol_note"] = "fixed earliest90/latest60 forward split; original preprocessing; no evaluation sample truncation"
    json_dump(path, meta)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument("--fold", type=int, choices=[0], default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare and args.output is None:
        parser.error("--prepare requires --output")
    print(json.dumps(freeze()) if args.freeze else prepare(args.fold, args.output))
