"""Read immutable prediction inputs after validating their scientific receipts."""
import json
import numpy as np
from .common import ROOT, sha256
from . import remaining_queue as q


def check_hashes(entries, cache=None):
    cache = {} if cache is None else cache
    for name, digest in entries.items():
        if cache.get(name) == digest: continue
        if sha256(ROOT / name) != digest: raise ValueError(f"artifact changed: {name}")
        cache[name] = digest


def read_job(name, state=None, cache=None):
    if state is None: state = json.loads((q.FOLDER / "queue_state.json").read_text())
    row = state["jobs"][name]
    if row["status"] != "completed": raise ValueError(f"score prerequisite incomplete: {name}")
    folder = ROOT / row["output"]
    if not q.verified(folder, row["spec"]): raise ValueError(f"unverified score prerequisite: {name}")
    meta = json.loads((folder / "run_meta.json").read_text())
    for group in ("input_sha256", "source_sha256", "mask_sha256"):
        check_hashes(meta.get(group, {}), cache)
    models = meta.get("model_sha256", {})
    if isinstance(models, str):
        candidates = [folder / f for f in ("model.pt", "model.joblib") if (folder / f).is_file()]
        if len(candidates) != 1: raise ValueError("ambiguous model checkpoint receipt")
        path = candidates[0]
        if sha256(path) != models: raise ValueError("model checkpoint changed")
    else: check_hashes(models, cache)
    for file, field in (("training_masks.npz", "training_masks_sha256"), ("training_mask_cohorts.json", "training_mask_cohorts_sha256")):
        if field in meta and sha256(folder / file) != meta[field]: raise ValueError("training mask receipt changed")
    with np.load(folder / "scores.npz") as archive: arrays = {k: archive[k].copy() for k in archive.files}
    return arrays, meta, folder


def archive_axes(arrays, ids, scenarios, methods=None):
    if arrays["incident_ids"].tolist() != ids or arrays["scenario_keys"].tolist() != scenarios:
        raise ValueError("score event or scenario axis differs")
    if methods is not None and arrays["method_names"].tolist() != methods: raise ValueError("score method axis differs")


def fused(context, evidence):
    from .train_evidence import fuse_all
    if context.shape != evidence.shape or context.ndim != 3: raise ValueError("fusion axes differ")
    return np.stack([fuse_all((.25, c), (.75, e)) for c, e in zip(context, evidence)])
