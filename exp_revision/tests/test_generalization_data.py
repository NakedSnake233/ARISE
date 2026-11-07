import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit

from exp_revision import generalization_data as data
from exp_revision.common import json_dump


def rows():
    result = [dict(incident_id=f"e{i}", root_service=f"svc{i//2:02d}", fault_type="a" if i % 2 == 0 else "b", overlap_group=i)
              for i in range(20)]
    result[3]["overlap_group"] = result[0]["overlap_group"]
    return result


@pytest.mark.parametrize("dataset,task", [(d, t) for d in ("tt", "aiops") for t in ("T5a", "T5b")])
def test_holdouts_purge_groups_and_partition_events(dataset, task):
    source = rows(); lookup = {r["incident_id"]: r for r in source}
    cohorts = data.make_cohorts(source, task, dataset)
    assert {key for part in cohorts.values() for key in part["test"]} == set(lookup)
    assert sum(len(part["test"]) for part in cohorts.values()) == len(source)
    for fold, part in enumerate(cohorts.values()):
        train_groups = {str(lookup[key]["overlap_group"]) for key in part["train"]}
        test_groups = {str(lookup[key]["overlap_group"]) for key in part["test"]}
        assert train_groups.isdisjoint(test_groups)
        assert set(part["train"] + part["test"] + part["purged"]) == set(lookup)
        if task == "T5a": assert all(lookup[key]["fault_type"] != part["held_fault_type"] for key in part["train"])
        else:
            expected = [f"svc{fold:02d}", f"svc{fold+5:02d}"]
            assert part["held_root_services"] == expected
            assert all(lookup[key]["root_service"] not in expected for key in part["train"])
        for inner in (part["context_inner"], part["baseline_inner"]):
            assert set(inner["fit"] + inner["validation"]) == set(part["train"])
            assert {lookup[key]["overlap_group"] for key in inner["fit"]}.isdisjoint({lookup[key]["overlap_group"] for key in inner["validation"]})


def test_ai_inner_validation_preserves_native_group_type():
    source = [dict(incident_id=f"e{i}", overlap_group=i) for i in range(20)]
    _, validation = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=2027).split(np.arange(20), groups=np.arange(20)))
    assert data.inner_partition(source, .2)["validation"] == [f"e{i}" for i in validation]
    assert data.inner_partition(source, .2)["validation"] != data.inner_partition(source, .2, string_groups=True)["validation"]


def test_prepare_fits_train_only_and_encodes_unseen_test_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "dataset_dir", lambda dataset: tmp_path / "data")
    monkeypatch.setattr(data, "config_snapshot", lambda command, config: {"command": command, "config": config})
    for name in ("generalization_data.py", "preprocess_tt.py", "preprocess_aiops.py"):
        path = tmp_path / "exp_revision" / name; path.parent.mkdir(exist_ok=True); path.write_text("fixture source")
    services = ["a", "b", "c"]
    source = [dict(incident_id=f"e{i}", start=0, end=4, source=f"raw/{i}/case.json", overlap_group=i) for i in range(3)]
    json_dump(tmp_path / "splits/tt_unlabeled_events.json", source)
    json_dump(tmp_path / "splits/tt_heldtype.json", {"fold_0": dict(train=["e0", "e1"], test=["e2"], split="heldtype_f")})
    json_dump(tmp_path / "data/metadata.json", dict(services=services, step=1, before=2, after=2))
    for i, row in enumerate(source):
        path = tmp_path / f"cache/raw_telemetry_tt/case-{i:03d}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
        identity = {key: row[key] for key in ("incident_id", "source", "start", "end")}
        identity.update(step=1, bins=4, services=services)
        calls = [dict(source="a" if i < 2 else "c", target="b" if i < 2 else "a", count=1)]
        np.savez_compressed(path, metric=np.full((3, 4, len(data.tt.METRICS)), [0, 2, 100][i], dtype=np.float32),
            trace=np.zeros((3, 4, 3), np.float32), metric_presence=np.ones(3, bool),
            calls_json=np.array(json.dumps(calls)), identity_json=np.array(json.dumps(identity)))
    text = {"e0": "known0", "e1": "known1", "e2": "test-only"}
    monkeypatch.setattr(data.tt, "_iter_logs", lambda row, *args: iter([(0, 0, text[row["incident_id"]])]))
    added = []
    class Miner:
        def __init__(self): self.drain = SimpleNamespace(clusters=[]); self.config = SimpleNamespace(depth=4)
        def add_log_message(self, message):
            added.append(message); key = len(self.drain.clusters) + 1
            self.drain.clusters.append(SimpleNamespace(cluster_id=key, get_template=lambda: message))
            return {"cluster_id": key}
        def match(self, message): return None
    monkeypatch.setitem(sys.modules, "drain3", SimpleNamespace(TemplateMiner=Miner))
    output = tmp_path / "output"; data.prepare("tt", "T5a", 0, output)
    metadata = json.loads((output / "metadata.json").read_text())
    assert added == ["known0", "known1"] and "test-only" not in metadata["log_templates"]
    assert metadata["training_graph_edges"] == [["a", "b"]]
    assert np.asarray(metadata["metric_scaler"]["mean"]).reshape(-1).tolist() == [1.] * (3 * len(data.tt.METRICS))
    with np.load(output / "features.npz") as archive:
        assert archive["logs"][2, 0, 0, 0] > 0
        assert archive["metrics"][2].min() == 20.
        assert set(archive.files) == {"metrics", "logs", "traces", "availability"}
