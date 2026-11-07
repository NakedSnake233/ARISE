from types import SimpleNamespace

import numpy as np
import pytest

from exp_revision import generalization_models as models
from exp_revision.train_evidence import _fit_one, fit, model_for
from exp_revision.masked_evidence import transform


def toy():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8, 3, 4)).astype(np.float32)
    names = ["metric:x:positive_z", "trace:x:positive_z", "log:template_js", "callers:degree"]
    a0 = np.ones((8, 3, 3), np.uint8); a0[1, 1, 2] = 0
    data = SimpleNamespace(train_ids=[f"t{i}" for i in range(6)], services=["a", "b", "c"],
        train_labels=np.array([0, 1, 2, 0, 1, 2]), availability=a0)
    return SimpleNamespace(data=data, cached=x, names=names)


def same_trees(actual, expected):
    assert actual.get_params() == expected.get_params()
    for left, right in zip(actual.estimators_, expected.estimators_):
        for field in ("threshold", "feature", "value"):
            np.testing.assert_array_equal(getattr(left.tree_, field), getattr(right.tree_, field))


def test_augmented_cohort_training_is_original_t1():
    evidence = toy(); trainer = models.CachedTrainer(evidence, 43); indices = np.array([0, 1, 3, 4])
    actual = trainer.fit_pair(indices, dict(augment=True))
    for model, generic in zip(actual, (False, True)):
        expected = _fit_one(evidence.cached, evidence.names, evidence.data.train_labels, evidence.data.availability,
            indices, 43, type_agnostic=generic)[0]
        same_trees(model, expected)
    assert len(trainer.views(indices)) == 7


def test_remove_augmentation_preserves_family_gate_and_flags():
    evidence = toy(); trainer = models.CachedTrainer(evidence, 44); indices = np.array([0, 1, 3, 4])
    actual = trainer.fit_pair(indices, dict(augment=False))
    for model, generic in zip(actual, (False, True)):
        x = transform(evidence.cached[indices], evidence.names, evidence.data.availability[indices], "family_gated", generic=generic)
        expected = fit(model_for(None).set_params(random_state=44), x, evidence.data.train_labels[indices])
        same_trees(model, expected)
        assert model.n_features_in_ == x.shape[-1]
        np.testing.assert_array_equal(x[:, :, -3:], evidence.data.availability[indices])


def test_beta_cache_cannot_score_missing_test_views():
    evidence = toy(); trainer = models.CachedTrainer(evidence, 43)
    with pytest.raises(ValueError, match="complete views"):
        trainer.score_pair(None, np.array([6, 7]), np.zeros((2, 3, 3)), dict(augment=True))
