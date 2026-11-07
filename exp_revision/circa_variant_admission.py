"""Independent smoke/full gates for the explicitly approved CIRCA input variant."""
from __future__ import annotations
import json

from .common import ROOT, sha256
from . import t8_admission as admission
from . import t8_manifest as original
from . import remaining_queue as q
from . import re2_data as data
from .extended_queue import isolatable_failure

METHOD = "CIRCA+rank-prune"


def create(dataset, smoke=True):
    approval = ROOT / "docs/circa_preprocessing_authorization.json"
    if not approval.exists() or json.loads(approval.read_text()).get("approved") is not True: return None
    path = q.MANIFESTS / f"{'55_pilot' if smoke else '50_full'}_official_{dataset}_CIRCAplusrankprune.json"
    if path.exists(): return None
    state = admission.state(); name = f"smoke_official_{dataset}_CIRCA_f0_s42"
    if name not in state["jobs"] or not isolatable_failure(state["jobs"][name]): return None
    original_spec = state["jobs"][name]["spec"]
    projection = None
    if not smoke:
        pilot_id = f"smoke_official_{dataset}_CIRCAplusrankprune_f0_s42"
        if not admission.verified_job(pilot_id, state): return None
        if dataset in data.DATASETS and not all(admission.verified_job(f"re2_fold_{dataset}_f{f}", state) for f in range(5)): return None
        pilot = state["jobs"][pilot_id]; elapsed = sum(a["elapsed_seconds"] for a in pilot["attempts"])
        parts = json.loads((ROOT / f"splits/{dataset}_kfold5.json").read_text())
        cases = sum(len(p["test"]) for p in parts.values()); cost = elapsed * cases * 5 / 3600
        projection = dict(kind="official", forecast_hours=dict(cpu=cost, gpu=0.),
            details=[dict(dataset=dataset, method=METHOD, smoke_seconds=elapsed, forecast_hours=cost, resource="cpu")],
            basis="successful author-approved variant smoke; full held-out cases, five seeds")
        admission.admit_forecast(projection, "official_" + dataset + "_CIRCAplusrankprune")
    jobs = []; parts = json.loads((ROOT / f"splits/{dataset}_kfold5.json").read_text())
    for fold in range(1 if smoke else 5):
        for seed in [42] if smoke else range(42, 47):
            ids = parts[f"fold_{fold}"]["test"][:1] if smoke else parts[f"fold_{fold}"]["test"]
            jobs.append({**original_spec, "id": f"{'smoke' if smoke else 'full'}_official_{dataset}_CIRCAplusrankprune_f{fold}_s{seed}",
                "source_commit": admission.commit(), "timeout_seconds": 900 if smoke else max(3600, int(elapsed * len(ids) * 2) + 60),
                "expected_config": {**original_spec["expected_config"], "method": METHOD, "fold": fold, "seed": seed, "smoke": smoke},
                "command": ["-m", "exp_revision.circa_rank_prune", "--dataset", dataset, "--fold", str(fold), "--seed", str(seed)] + (["--smoke"] if smoke else []),
                "test_ids": ids, "score_shape": [12, len(ids), original_spec["score_shape"][-1]]})
    sources = original.closure("official", [dataset])
    for name in ("exp_revision/circa_variant_admission.py", "exp_revision/circa_rank_prune.py",
            "exp_revision/rcaeval_rank_worker.py", "exp_revision/circa_rank_prune_policy.json", str(approval.relative_to(ROOT))):
        sources[name] = sha256(ROOT / name)
    return admission.write_manifest(path, ("pilot" if smoke else "full") + "_official_" + dataset + "_CIRCAplusrankprune", jobs, sources, projection)
