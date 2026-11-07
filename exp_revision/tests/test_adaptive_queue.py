import copy
import numpy as np

from exp_revision import adaptive_queue as a
from exp_revision import variant_timeout_recovery as r
from exp_revision import train_variant_evidence as training
from exp_revision.common import ROOT
from exp_revision.fold_data import load_fold
from exp_revision.masked_evidence import MaskedEvidence


def test_effective_deadline_preserves_old_attempt_and_full_accounting():
    job = dict(resource="cpu", cores=4, timeout_seconds=1200,
        expected_config=dict(task="T3", smoke=False, variants={}))
    assert a.deadline(job) == 2400
    assert a.deadline(job, {}) == 1200
    attempt = dict(runtime_timeout_seconds=2400)
    state = dict(jobs={"x": dict(spec=job, attempts=[dict(elapsed_seconds=1202), attempt])})
    assert np.isclose(a.charged(state, "cpu"), 40 + 1202 * 4 / 3600)
    assert np.isclose(a.charged(state, "cpu", True), 40 + (1202 + 2410) * 4 / 3600)
    attempt["elapsed_seconds"] = 2403
    assert np.isclose(a.charged(state, "cpu"), 40 + (1202 + 2403) * 4 / 3600)
    assert job["timeout_seconds"] == 1200


def test_original_cohort_masks_rebuild_bit_identically():
    # A completed full pilot is the independent witness for recovery's mask rebuild.
    import json
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    row = state["jobs"]["full_t3_evidence_tt_fold0_seed42"]
    folder = ROOT / row["output"]
    data = load_fold("tt", 0)
    class Evidence:
        pass
    evidence = Evidence(); evidence.data = data
    trainer = training.VariantTrainer(evidence, 42)
    index = {name: i for i, name in enumerate(data.train_ids)}
    holds = r.expected_holdouts(trainer)
    selection = json.loads((folder / "variant0/inner_beta.json").read_text())
    assert holds == selection["holdouts"]
    for item in holds:
        trainer.views(np.asarray([index[name] for name in item["fit_ids"]], dtype=int))
    trainer.views(data.train_indices)
    with np.load(folder / "training_masks.npz") as archive:
        for indices, masks in trainer.view_masks.items():
            assert np.array_equal(np.stack(masks).astype(np.uint8), archive[trainer.cohorts[indices]])
    cohorts = json.loads((folder / "training_mask_cohorts.json").read_text())
    assert cohorts == {trainer.cohorts[k]: [data.train_ids[i] for i in k] for k in trainer.view_masks}


def test_only_exhausted_deadline_attempts_eligible():
    row = dict(status="stuck", spec=dict(timeout_seconds=1200, expected_config=dict(task="T3", smoke=False, variants={})),
        attempts=[dict(status="interrupted", elapsed_seconds=1202), dict(status="interrupted", elapsed_seconds=1204, exit_code=124)])
    assert r.eligible(row)
    row["attempts"][-1]["status"] = "failed"
    assert not r.eligible(row)
    row["attempts"][-1].update(status="interrupted", elapsed_seconds=30)
    assert not r.eligible(row)
    row["attempts"][-1].update(elapsed_seconds=1500, runtime_timeout_seconds=2400)
    assert not r.eligible(row)
