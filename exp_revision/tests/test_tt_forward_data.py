"""Integrity checks for a new chronological split, without model training."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from exp_revision.common import ROOT, dataset_dir
from exp_revision.tt_forward_data import make_cohorts
from exp_revision.tt_forward_models import neural_scores, gm


class ForwardChecks(unittest.TestCase):
    def setUp(self):
        self.events = json.loads((ROOT / "splits/tt_unlabeled_events.json").read_text())
        ordered = sorted(self.events, key=lambda row: (row["start"], row["end"], row["incident_id"]))
        self.train_ids = [row["incident_id"] for row in ordered[:90]]
        self.test_ids = [row["incident_id"] for row in ordered[90:]]
        self.labels = {row["incident_id"]: row for row in json.loads((dataset_dir("tt") / "incidents.json").read_text())
                       if row["incident_id"] in self.train_ids}

    def test_fixed_chronological_cut_and_pair_isolated_inner(self):
        part = make_cohorts(self.events, self.labels)["fold_0"]
        self.assertEqual(part["train"], self.train_ids)
        self.assertEqual(part["test"], self.test_ids)
        self.assertEqual(len(part["train"]), 90)
        self.assertEqual(len(part["test"]), 60)
        pair = lambda key: (self.labels[key]["root_service"], self.labels[key]["fault_type"])
        for name in ("context_inner", "baseline_inner"):
            inner = part[name]
            self.assertEqual(set(inner["fit"]) | set(inner["validation"]), set(self.train_ids))
            self.assertFalse(set(inner["fit"]) & set(inner["validation"]))
            self.assertFalse(set(map(pair, inner["fit"])) & set(map(pair, inner["validation"])))
        self.assertEqual(make_cohorts(list(reversed(self.events)), self.labels), {"fold_0": part})

    def test_test_labels_rejected_and_overlapping_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, "only the 90 outer-training labels"):
            make_cohorts(self.events, {**self.labels, self.test_ids[0]: {"root_service": "hidden"}})
        events = [dict(row) for row in self.events]
        lookup = {row["incident_id"]: row for row in events}
        lookup[self.train_ids[-1]]["end"] = lookup[self.test_ids[0]]["start"] + 1
        with self.assertRaisesRegex(ValueError, "intervals cross"):
            make_cohorts(events, self.labels)

    def test_eadro_md_inference_keeps_graph(self):
        import torch
        torch.manual_seed(42)
        widths = {name: 4 for name in ("metrics", "logs", "traces")}
        model = gm.baseline.EadroAdapter(widths, 8, 2, 0.)
        model.eval()
        data = SimpleNamespace(metadata={"before": 60, "step": 5}, test_ids=["a", "b"])
        prepared = {name: np.ones((2, 3, 4), np.float32) for name in widths}
        with patch.object(gm.baseline, "descriptors", return_value=(prepared, None)):
            expected = neural_scores(model, data, {}, None, np.eye(3, dtype=np.float32), "Eadro", {})
            actual = neural_scores(model, data, {}, None, np.eye(3, dtype=np.float32), "Eadro+MD", {})
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(actual.shape, (2, 3))
        np.testing.assert_allclose(actual.sum(1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
