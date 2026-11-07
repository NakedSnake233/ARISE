import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exp_revision import variant_smoke_repair as repair
from exp_revision.common import json_dump, sha256


@pytest.mark.parametrize("task,index", [("T3", 3), ("T6", 4)])
def test_selection_calls_unchanged_trainer(monkeypatch, tmp_path, task, index):
    name = f"{task}_VARIANTS"
    original = getattr(repair.training, name)
    method = list(original)[index]
    def train(*args, **kwargs):
        assert args == ("tt", 0, 42, task, tmp_path)
        assert kwargs == {"smoke": True}
        selected = getattr(repair.training, name)
        assert list(selected) == [method] and selected[method] is original[method]
        return "trained"
    monkeypatch.setattr(repair.training, "train", train)
    assert repair.selected_variant(task, index, tmp_path) == "trained"
    assert getattr(repair.training, name) is original


def fixture_children(monkeypatch, root, task):
    monkeypatch.setattr(repair, "ROOT", root)
    monkeypatch.setattr(repair.q, "ROOT", root)
    monkeypatch.setattr(repair.q, "FOLDER", root / "queue")
    monkeypatch.setattr(repair, "config_snapshot", lambda command, config: {"config": config, "command": command})
    settings = getattr(repair.training, f"{task}_VARIANTS")
    scenarios = ["S0_none", "S1_trace"]
    ids = ["event0", "event1"]
    original = dict(expected_config=dict(task=task, methods=list(settings), variants=settings, scenarios=scenarios),
                    expected_status=["completed_variant_evidence"], test_ids=ids,
                    score_shape=[len(settings), len(scenarios), len(ids), 3])
    jobs = {repair.TARGETS[task]: {"spec": original, "status": "stuck", "attempts": [{"status": "interrupted"}]}}
    shared = root / "shared"; shared.write_text("existing frozen input/source/mask")
    for index, (method, setting) in enumerate(settings.items()):
        folder = root / "children" / str(index); (folder / "variant0").mkdir(parents=True)
        np.savez_compressed(folder / "scores.npz", scores=np.full((1, 2, 2, 3), index),
            method_names=[method], incident_ids=ids, scenario_keys=scenarios)
        np.savez_compressed(folder / "training_masks.npz", cohort0=np.ones((7, 4, 3, 3)))
        json_dump(folder / "training_mask_cohorts.json", {"cohort0": ["train0", "train1"]})
        selection = {"selected_beta": .25, "holdouts": ["fit-only"]}
        json_dump(folder / "variant0/inner_beta.json", selection)
        models = {}
        for name in ("specific.joblib", "generic.joblib"):
            path = folder / "variant0" / name; path.write_bytes(f"{index}-{name}".encode())
            models[str(path.relative_to(root))] = sha256(path)
        config = {**original["expected_config"], "methods": [method], "variants": {method: setting}}
        meta = dict(status="completed_variant_evidence", config=config, errors=[], start_time="start", end_time="end",
            elapsed_seconds=100 + index, beta_selections={method: selection}, model_sha256=models,
            scores_sha256=sha256(folder / "scores.npz"), training_masks_sha256=sha256(folder / "training_masks.npz"),
            training_mask_cohorts_sha256=sha256(folder / "training_mask_cohorts.json"))
        for group in ("input_sha256", "source_sha256", "mask_sha256"): meta[group] = {"shared": sha256(shared)}
        json_dump(folder / "run_meta.json", meta)
        jobs[f"{repair.PREFIX}{task}_variant{index}"] = dict(status="completed", output=str(folder.relative_to(root)),
            spec={**original, "expected_config": config, "score_shape": [1, 2, 2, 3]})
    json_dump(root / "queue/queue_state.json", {"jobs": jobs})
    return original, jobs


@pytest.mark.parametrize("task", ["T3", "T6"])
def test_composition_retains_masks_attempts_and_training_time(monkeypatch, tmp_path, task):
    original, jobs = fixture_children(monkeypatch, tmp_path, task)
    output = tmp_path / "assembled"
    repair.compose(task, output)
    assert repair.q.verified(output, original)
    with np.load(output / "scores.npz") as archive:
        for index in range(len(original["expected_config"]["methods"])): assert (archive["scores"][index] == index).all()
    meta = json.loads((output / "run_meta.json").read_text())
    assert meta["mask_sha256"] == {"shared": sha256(tmp_path / "shared")}
    assert meta["preserved_attempts"] == jobs[repair.TARGETS[task]]["attempts"]
    assert meta["training_elapsed_seconds"] == sum(meta["child_elapsed_seconds"].values())
    assert meta["training_elapsed_seconds"] > meta["elapsed_seconds"]


@pytest.mark.parametrize("change", ["model", "beta", "cohort_masks"])
def test_composition_rejects_changed_child(monkeypatch, tmp_path, change):
    fixture_children(monkeypatch, tmp_path, "T3")
    folder = tmp_path / "children/1"
    if change == "model": (folder / "variant0/specific.joblib").write_bytes(b"changed")
    elif change == "beta": json_dump(folder / "variant0/inner_beta.json", {"selected_beta": 1.0})
    else:
        np.savez_compressed(folder / "training_masks.npz", cohort0=np.zeros((7, 4, 3, 3)))
        meta = json.loads((folder / "run_meta.json").read_text())
        meta["training_masks_sha256"] = sha256(folder / "training_masks.npz")
        json_dump(folder / "run_meta.json", meta)
    with pytest.raises(ValueError, match="child artifact changed|beta selection differs|identical original cohort masks"):
        repair.compose("T3", tmp_path / "assembled")


def test_monitor_routes_unfinished_repair_before_normal_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(repair, "ROOT", tmp_path)
    monkeypatch.setattr(repair.q, "FOLDER", tmp_path / "queue")
    monkeypatch.setattr(repair, "DiskGuard", lambda root: SimpleNamespace(poll=lambda: {"safe_to_launch": True}))
    manifest = tmp_path / "manifests/remaining/14_split_tt_variant_smoke.json"
    json_dump(manifest, {})
    json_dump(tmp_path / "queue/queue_state.json", {"status": "stuck", "jobs": {"old": {"status": "stuck"}}})
    (tmp_path / "logs").mkdir()
    commands = []
    def launch(command, **kwargs):
        commands.append(command); return SimpleNamespace(pid=123)
    monkeypatch.setattr(repair.subprocess, "Popen", launch)
    assert repair.check()["action"] == "resume_variant_smoke_repair"
    assert commands[0][-1] == "--repair"
    json_dump(tmp_path / "results_v2/variant_smoke_repair/receipt.json", {"status": "completed"})
    monkeypatch.setattr(repair.q, "check", lambda: {"action": "normal_queue"})
    assert repair.check()["action"] == "normal_queue"
