"""Command-line entry points for the verified part of the revision protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_bundle
from .reproduction import write_reproduction
from .splits import check_splits, load_or_make, write_training_label_manifests, write_unlabeled_manifest


BLOCKED = {
    "T2": "Raw, fold-specific availability and preprocessing must be implemented before missingness predictions are valid.",
    "T3": "T3 requires the original two-expert evidence pipeline and recomputation from masked raw telemetry.",
    "T5": "Generalization depends on the same fold-specific preprocessing and original model adapters.",
    "T6": "Ablations require the original context, evidence, and fusion components.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="ARISE experiment revision")
    parser.add_argument("--task", required=True, choices=["T0", "T1", "T2", "T3", "T4", "T5", "T6"])
    parser.add_argument("--dataset", choices=["tt", "aiops"], default="tt")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-fold", type=int, choices=range(5), metavar="FOLD")
    args = parser.parse_args()
    if args.task == "T0":
        rows = write_reproduction(args.dataset)
        print(f"T0 verified {len(rows)} methods on {args.dataset}; see results_v2/T0_reproduction.csv")
    elif args.task == "T1":
        if args.prepare_fold is not None:
            path = Path("splits") / f"{args.dataset}_kfold5.json"
            unlabeled = Path("splits") / f"{args.dataset}_unlabeled_events.json"
            if not path.exists() or not unlabeled.exists():
                parser.error("freeze the T1 fold and label-free event manifests first")
            if args.dataset == "tt":
                from .preprocess_tt import prepare_fold
            else:
                from .preprocess_aiops import prepare_fold
            print(f"T1 raw fold preprocessing: {prepare_fold(args.prepare_fold, args.smoke)}")
            print("T1 model evaluation remains pending")
            return
        bundle = load_bundle(args.dataset)
        path = Path("splits") / f"{args.dataset}_kfold5.json"
        split = load_or_make(bundle, path)
        check_splits(bundle, split)
        unlabeled = Path("splits") / f"{args.dataset}_unlabeled_events.json"
        write_unlabeled_manifest(bundle, unlabeled)
        write_training_label_manifests(bundle, split, Path("splits") / f"{args.dataset}_training_labels")
        print(f"T1 fold manifest verified: {path}; label-free event manifest: {unlabeled}")
        print("T1 model evaluation remains pending")
    elif args.task == "T4":
        from .sanity_baselines import run
        print(run(args.dataset, args.smoke))
    else:
        parser.error(BLOCKED[args.task])


if __name__ == "__main__":
    main()
