from types import SimpleNamespace
import numpy as np
import pytest

from exp_revision.train_variant_evidence import VariantTrainer
from robust_evidence import _augmentation_availabilities


def test_variants_share_original_masks_and_reject_test_indices():
    data = SimpleNamespace(train_ids=["t0", "t1", "t2"], services=["a", "b", "c"])
    trainer = VariantTrainer(SimpleNamespace(data=data), 43)
    indices = np.array([0, 2])
    views = trainer.views(indices)
    expected = _augmentation_availabilities((2, 3, 196), 43, (.25, .5, .75))
    assert len(views) == 7 and trainer.views(indices) is views
    for actual, original in zip(views, expected): np.testing.assert_array_equal(actual, original)
    with pytest.raises(ValueError, match="training events only"):
        trainer.views(np.array([0, 3]))
