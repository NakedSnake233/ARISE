"""Admit a training-only input diagnostic, never the longer-window full matrix."""
from __future__ import annotations
import json

from .common import ROOT, dataset_dir, sha256
from . import remaining_queue as q
from . import t8_admission as admission
from . import t8_manifest
from .circa_window_assess import windows,overlap_groups
from .circa_window_probe import POLICY


def create():
    path=q.MANIFESTS/"57_smoke_circa_long_window_diagnostic.json"
    if path.exists():return None
    decision=ROOT/"docs/circa_long_window_execution_decision.json"
    if not decision.exists() or json.loads(decision.read_text()).get("run_long_window_smoke") is not True:return None
    policy=json.loads(POLICY.read_text());part=json.loads((ROOT/"splits/aiops_kfold5.json").read_text())["fold_0"]
    rows=json.loads((ROOT/"splits/aiops_unlabeled_events.json").read_text())
    groups=overlap_groups(windows(rows,policy["before_seconds"],policy["after_seconds"]))
    exclude={groups[key] for key in part["test"]};ids=[key for key in part["train"] if groups[key] not in exclude][:5]
    cfg=dict(task="T8",component="window_technical_diagnostic",dataset="aiops",fold=0,seed=42,smoke=True,protocol=policy["protocol"])
    job=dict(id="smoke_circa_long_window_training_diagnostic",resource="cpu",cores=1,max_attempts=2,
        timeout_seconds=600,source_commit=admission.commit(),expected_config=cfg,
        command=["-m","exp_revision.circa_window_probe"],test_ids=ids,score_shape=[2,len(ids),10],
        expected_status=["completed_window_diagnostic"])
    projection=dict(kind="input_diagnostic",forecast_hours=dict(cpu=610/3600,gpu=0.),
        details=[dict(dataset="aiops",component="window_technical_diagnostic",per_job_hours=610/3600)],
        basis="one600s-deadline training-only diagnostic; no full admission or test accuracy")
    admission.admit_forecast(projection,"long_window_diagnostic")
    sources=t8_manifest.closure("official",["aiops"])
    for name in ("exp_revision/circa_probe_admission.py","exp_revision/circa_window_probe.py","exp_revision/circa_window_assess.py",
            "exp_revision/circa_long_window_policy.json","exp_revision/rcaeval_rank_worker.py","exp_revision/circa_rank_prune_policy.json",
            "docs/circa_long_window_authorization.json","docs/circa_preprocessing_authorization.json",
            "splits/aiops_unlabeled_events.json",str((dataset_dir("aiops")/"metadata.json").relative_to(ROOT)),
            "exp_revision/drain_kernel_env.py","exp_revision/drain_acceleration.py","exp_revision/drain_kernel_spec.json"):
        sources[name]=sha256(ROOT/name)
    return admission.write_manifest(path,"pilot_long_window_diagnostic",[job],sources,projection)


if __name__=="__main__":print(create())
