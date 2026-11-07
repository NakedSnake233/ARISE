"""Apply an explicit author budget decision without editing frozen schedulers."""
from __future__ import annotations
import json

from .common import ROOT,sha256
from . import remaining_queue as q

APPROVAL=ROOT/"docs/existing_protocol_compute_authorization.json"
DEFAULT=dict(cpu=200.,gpu=48.)
PROPOSED=dict(cpu=260.,gpu=56.)


def apply():
    limits=DEFAULT;receipt=None
    if APPROVAL.exists():
        value=json.loads(APPROVAL.read_text())
        if value.get("approved") is True:
            if (value.get("limits")!=PROPOSED or value.get("scope")!="existing_protocol_only"
                    or value.get("source")!="explicit author reply in this conversation"):
                raise ValueError("unsupported or ambiguous compute authorization")
            limits=PROPOSED;receipt=sha256(APPROVAL)
    q.CPU_BUDGET=limits["cpu"];q.GPU_BUDGET=limits["gpu"]
    return dict(limits=limits,author_approval_sha256=receipt,
        policy_source_sha256=sha256(ROOT/"exp_revision/compute_budget_policy.py"),longer_window_authorized=False)


def bind_dump(original):
    def dump(path,value):
        if path.parent==q.MANIFESTS and isinstance(value,dict) and "jobs" in value:
            policy=apply();value={**value,"compute_budget_policy":policy}
            sources=dict(value["source_sha256"])
            sources["exp_revision/compute_budget_policy.py"]=policy["policy_source_sha256"]
            if policy["author_approval_sha256"] is not None:
                sources[str(APPROVAL.relative_to(ROOT))]=policy["author_approval_sha256"]
            value["source_sha256"]=sources
        return original(path,value)
    return dump
