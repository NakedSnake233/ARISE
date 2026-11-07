import copy

from exp_revision.priority_queue import prioritize


def test_only_order_changes_and_smoke_pilot_precede_full():
    state = {"jobs": {
        "full_long": {"spec": {"expected_config": {"smoke": False}, "resource": "cpu", "cores": 4}, "status": "running", "attempts": [{"pid": 123}]},
        "pilot": {"spec": {"expected_config": {"smoke": False}}, "status": "pending"},
        "prep": {"spec": {"expected_config": {"smoke": False, "component": "train_only_preprocessing"}}, "status": "pending"},
        "smoke": {"spec": {"expected_config": {"smoke": True}}, "status": "pending"},
    }}
    original = copy.deepcopy(state)
    prioritize(state, {"pilot"})
    assert list(state["jobs"]) == ["pilot", "smoke", "prep", "full_long"]
    assert state == original
