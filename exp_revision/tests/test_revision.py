import subprocess
import sys
import json
from types import SimpleNamespace

import numpy as np
import pytest

from exp_revision.common import load_bundle, stable_seed
from exp_revision.fold_data import load_fold
from exp_revision.metrics import ranking_metrics
from exp_revision.missingness import generate_masks, make_mask
from exp_revision.preprocess_tt import _templates_disjoint
from exp_revision.reproduction import verify_dataset
from exp_revision.splits import check_splits, load_or_make, write_training_label_manifests, write_unlabeled_manifest


def test_metrics():
    assert ranking_metrics(np.array([1]))["PR@1"] == 1
    assert ranking_metrics(np.array([1]))["MAP@3"] == 1
    assert ranking_metrics(np.array([2]))["MAP@3"] == .5
    assert ranking_metrics(np.array([6]))["MAP@5"] == 0
    assert ranking_metrics(np.array([np.inf]))["MRR"] == 0


def test_mask_rules():
    a0 = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=np.uint8)
    inc = {"dataset": "toy", "incident_id": "case-1"}
    k1, rate1 = make_mask(inc, a0, "S2", "0.75", 42)
    k2, rate2 = make_mask(inc, a0, "S2", "0.75", 42)
    assert np.array_equal(k1, k2) and rate1 == rate2
    assert np.all((a0 & k1).sum(axis=1) >= 1)
    ks, _ = make_mask(inc, a0, "S3", "all", 42, root=0)
    assert np.all(ks[0] == 0) and np.all(ks[1:] == 1)
    kc, _ = make_mask(inc, a0, "S3c", "trace", 42, root=0)
    assert np.all(kc[0] == 1) and kc[1, 2] == 0 and np.all(kc[2] == 1)


def test_stable_seed_across_processes():
    code = "from exp_revision.common import stable_seed; print(stable_seed('tt','x',42))"
    a = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    b = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert a == b == str(stable_seed("tt", "x", 42))


def test_non_root_masks_never_read_labels(tmp_path):
    class ForbiddenLabels:
        def __getitem__(self, _):
            raise AssertionError("test root label was read")
    class Bundle:
        dataset = "toy"
        rows = [{"incident_id": "case-1"}]
        availability = np.ones((1, 3, 3), dtype=np.uint8)
        labels = ForbiddenLabels()
    generate_masks(Bundle(), tmp_path, scenarios=[("S0", ""), ("S1", "log"), ("S2", "0.50")], seeds=[42])
    assert len(list(tmp_path.rglob("*.npz"))) == 3


def test_s2_recovery_expected_rate():
    # For one initially available modality, deletion is always undone.
    from exp_revision.audit_s2 import audit
    row = audit("tt")
    assert row["available_modality_count"][1] > 0
    assert abs(row["realized_rates"]["0.75"] - row["expected_rates_under_restoration"]["0.75"]) < .02
    assert abs(row["realized_rates"]["0.75"] - .75) > .02


def test_fold_manifests():
    from pathlib import Path
    for dataset in ("tt", "aiops"):
        bundle = load_bundle(dataset)
        split = load_or_make(bundle, Path("splits") / f"{dataset}_kfold5.json")
        check_splits(bundle, split)


def test_unlabeled_event_manifest_excludes_heldout_labels(tmp_path):
    bundle = SimpleNamespace(rows=[{
        "incident_id": "one", "start": 1, "end": 2, "source": "raw/one/incident.json",
        "overlap_group": 3, "root_service": "secret-root", "fault_type": "secret-fault",
    }])
    path = tmp_path / "events.json"
    write_unlabeled_manifest(bundle, path)
    text = path.read_text()
    assert "secret-root" not in text and "secret-fault" not in text
    assert set(json.loads(text)[0]) == {"incident_id", "start", "end", "source", "overlap_group"}


def test_training_label_manifest_excludes_outer_test(tmp_path):
    bundle = SimpleNamespace(rows=[
        {"incident_id": "train", "root_service": "train-root", "fault_type": "train-type", "overlap_group": 1},
        {"incident_id": "test", "root_service": "secret-test-root", "fault_type": "secret-test-type", "overlap_group": 2},
    ])
    write_training_label_manifests(bundle, {"fold_0": {"train": ["train"], "test": ["test"]}}, tmp_path)
    text = (tmp_path / "fold_0.json").read_text()
    assert "secret-test-root" not in text and "secret-test-type" not in text
    assert [row["incident_id"] for row in json.loads(text)] == ["train"]


def test_drain_fast_path_requires_unambiguous_templates():
    assert _templates_disjoint({1: "error in payment", 2: "error in cart"})
    assert not _templates_disjoint({1: "error in <*> ", 2: "error in cart"})


def test_fold_loader_exposes_only_train_labels(tmp_path):
    cache = tmp_path / "cache" / "tt" / "fold0_smoke"
    splits = tmp_path / "splits"
    cache.mkdir(parents=True)
    (splits / "tt_training_labels").mkdir(parents=True)
    (splits / "tt_kfold5.json").write_text(json.dumps({"fold_0": {"train": ["tr"], "test": ["te"]}}))
    (splits / "tt_training_labels" / "fold_0.json").write_text(json.dumps([
        {"incident_id": "tr", "root_service": "a", "fault_type": "cpu", "overlap_group": 1}]))
    (cache / "metadata.json").write_text(json.dumps({
        "train_ids": ["tr"], "test_ids": ["te"], "services": ["a", "b"], "training_graph_edges": []}))
    arrays = {key: np.zeros((2, 2, 2, 1), np.float32) for key in ("metrics", "logs", "traces")}
    arrays["availability"] = np.ones((2, 2, 3), np.uint8)
    np.savez_compressed(cache / "features.npz", **arrays)
    fold = load_fold("tt", 0, smoke=True, cache_root=tmp_path / "cache", splits_root=splits)
    assert fold.train_labels.tolist() == [0] and fold.test_ids == ["te"]
    assert not hasattr(fold, "test_labels")
    np.savez_compressed(cache / "features.npz", **arrays, labels=np.array([0, 1]))
    with pytest.raises(ValueError, match="unexpected fields"):
        load_fold("tt", 0, smoke=True, cache_root=tmp_path / "cache", splits_root=splits)


def test_saved_predictions_match_appendix_d():
    for dataset in ("tt", "aiops"):
        assert all(row["within_rounding_tolerance"] for row in verify_dataset(dataset))
