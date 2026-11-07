"""Original fixed model settings on the self-TT chronological cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, json_dump, sha256
from . import tt_forward_data as fd
from . import generalization_models as gm
from . import generalization_data as gd
from .s3v2_infer import shared_masks, SCENES


ORIGINAL_NEURAL_SCORES = gm.neural_scores
BASELINES = ("Eadro", "Eadro+MD", "MULAN")


def neural_scores(model, data, features, scaler, adjacency, method, cfg):
    # MD changes training augmentation, while the Eadro architecture still needs its graph.
    return ORIGINAL_NEURAL_SCORES(model, data, features, scaler, adjacency, method.removesuffix("+MD"), cfg)


def activate():
    fd.activate()
    gm.load_cohort = gd.load_cohort
    gm.split_file = fd.split_file
    gm.training_file = fd.training_file
    gm.neural_scores = neural_scores
    gm.scenarios = lambda task: list(SCENES)

    def masks(data, task, seed):
        values, hashes = shared_masks(data, seed)
        return dict(zip(SCENES, values)), hashes

    gm.get_masks = masks


def run(component, fold, seed, output, method=None):
    fd.training_file("tt", fd.TASK, fold)
    if seed not in (42, 43, 44, 45, 46):
        raise ValueError("forward validation freezes seeds 42 through 46")
    if component not in ("context", "experts", "baseline"):
        raise ValueError(component)
    if component == "baseline" and method not in BASELINES:
        raise ValueError("unexpected adapted baseline")
    if component != "baseline" and method is not None:
        raise ValueError("method is specified only for baseline components")
    activate()
    args = ("tt", fd.TASK, fold, seed, output)
    if component == "context":
        gm.train_context(*args, smoke=False)
    elif component == "experts":
        gm.train_experts(*args, smoke=False)
    else:
        gm.train_baseline(*args, method=method, smoke=False)
    path = output / "run_meta.json"
    meta = json.loads(path.read_text())
    meta["source_sha256"].update({str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "exp_revision/tt_forward_data.py")})
    meta["protocol_note"] = "earliest90/latest60 forward cohort; all60 evaluation events; original settings; outer-training preprocessing fixed during inner model selection"
    meta["inner_selection_note"] = "context/adapters: pair-grouped GroupShuffleSplit .1 seed2027; experts: original training-only fault-type holdout beta selection"
    meta["evaluation_scope"] = "complete_fixed_chronological_test_cohort"
    meta["baseline_implementation_note"] = "Eadro and MULAN are local adapted implementations, not strict reproductions of original paper pipelines"
    json_dump(path, meta)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("context", "experts", "baseline"), required=True)
    parser.add_argument("--fold", type=int, choices=[0], default=0)
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--method", choices=BASELINES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.component, args.fold, args.seed, args.output, args.method))
