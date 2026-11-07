"""Post-protocol isolated ablation: disable augmentation but retain flags.

The frozen main run contains a legacy no-augmentation/no-flag comparator and
an isolated no-flag comparator.  This supplementary run closes the remaining
factorial cell so the augmentation contribution is independently identifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robust_evidence import (
    MODALITIES,
    all_available,
    complete_modality_availability,
    fit_robust_ranker,
    fuse_all,
    local_missing_availability,
)
from run_review_experiments import BasePredictor, base_directory
from train_base import ranking_metrics


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/care/review-response-v1.json")
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    output = ROOT / config["output"]
    status = json.loads((output / "status.json").read_text())
    if status["state"] != "complete":
        raise RuntimeError("The frozen parent experiment must complete first")
    protocol_sha = digest(output / "protocol.json")
    ranker_cfg = {
        "partial_rates": config["evidence"]["partial_training_rates"],
        "trees": config["evidence"]["trees"],
        "min_samples_leaf": config["evidence"]["min_samples_leaf"],
    }
    results = {
        "parent_protocol_sha256": protocol_sha,
        "source_sha256": digest(Path(__file__)),
        "factor": "missing-modality augmentation x availability flags",
        "changed": "augment=False with append_flags in {True, False}",
        "held_constant": "base checkpoint, missing masks, generic weight, and fusion settings inherited",
        "datasets": {},
    }
    for dataset, dataset_cfg in config["datasets"].items():
        data = ROOT / dataset_cfg["data"]
        rows = json.loads((data / "incidents.json").read_text())
        loaded_features = np.load(data / "features.npz")
        labels = loaded_features["labels"]
        feature_arrays = {
            name: loaded_features[name] for name in ("metrics", "traces", "logs")
        }
        train_indices = np.asarray(
            [i for i, row in enumerate(rows) if row["split"] == "train"], dtype=int
        )
        test_indices = np.asarray(
            [i for i, row in enumerate(rows) if row["split"] == "test"], dtype=int
        )
        evidence_dir = ROOT / dataset_cfg["evidence"]
        x = np.load(evidence_dir / "full.npz")["x"]
        names = json.loads((evidence_dir / "full-names.json").read_text())
        availability = all_available(x.shape)
        dataset_records = []
        for seed in config["seeds"]:
            parent = json.loads(
                (output / dataset / f"seed-{seed}" / "result.json").read_text()
            )
            parent_predictions = np.load(
                output / dataset / f"seed-{seed}" / "predictions.npz"
            )
            base = parent_predictions["base"]
            generic_weight = float(parent["main"]["selected_generic_weight"])
            with_flags_raw = fit_robust_ranker(
                x,
                names,
                labels,
                train_indices,
                seed=seed,
                augment=False,
                append_flags=True,
                **ranker_cfg,
            )
            with_flags_generic = fit_robust_ranker(
                x,
                names,
                labels,
                train_indices,
                seed=seed,
                augment=False,
                append_flags=True,
                type_agnostic=True,
                **ranker_cfg,
            )
            without_flags_raw = fit_robust_ranker(
                x,
                names,
                labels,
                train_indices,
                seed=seed,
                augment=False,
                append_flags=False,
                **ranker_cfg,
            )
            without_flags_generic = fit_robust_ranker(
                x,
                names,
                labels,
                train_indices,
                seed=seed,
                augment=False,
                append_flags=False,
                type_agnostic=True,
                **ranker_cfg,
            )
            predictor = BasePredictor(
                base_directory(output, dataset, seed),
                dataset_cfg["device"],
                config["base"]["batch_size"],
            )

            def evaluate(
                raw_ranker,
                generic_ranker,
                observed: np.ndarray,
                base_scores: np.ndarray,
            ) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
                raw_scores = raw_ranker.predict(
                    x[test_indices], observed[test_indices]
                )
                generic_scores = generic_ranker.predict(
                    x[test_indices], observed[test_indices]
                )
                correction = fuse_all(
                    (1 - generic_weight, raw_scores),
                    (generic_weight, generic_scores),
                )
                correction_metrics = ranking_metrics(
                    correction, labels[test_indices]
                )
                # Use the condition-specific base prediction supplied by the
                # caller.  Complete observations reuse the sealed parent
                # prediction; missing views are replayed from the checkpoint.
                method_scores = fuse_all(
                    (config["evidence"]["base_weight"], base_scores),
                    (config["evidence"]["evidence_weight"], correction),
                )
                method_metrics = ranking_metrics(method_scores, labels[test_indices])
                return correction_metrics, method_metrics, method_scores

            full_correction, full, full_scores = evaluate(
                with_flags_raw, with_flags_generic, availability, base
            )
            no_flags_full_correction, no_flags_full, no_flags_full_scores = evaluate(
                without_flags_raw, without_flags_generic, availability, base
            )
            predictions = {
                "indices": test_indices,
                "labels": labels[test_indices],
                "full_with_flags": full_scores,
                "full_without_flags": no_flags_full_scores,
            }
            complete_missing = {}
            for modality in MODALITIES:
                observed = complete_modality_availability(x.shape, modality)
                masked_base = predictor.predict(
                    feature_arrays, observed, test_indices
                )
                with_correction, with_method, with_scores = evaluate(
                    with_flags_raw,
                    with_flags_generic,
                    observed,
                    masked_base,
                )
                without_correction, without_method, without_scores = evaluate(
                    without_flags_raw,
                    without_flags_generic,
                    observed,
                    masked_base,
                )
                complete_missing[modality] = {
                    "with_flags": {
                        "correction_only": with_correction,
                        "method": with_method,
                    },
                    "without_flags": {
                        "correction_only": without_correction,
                        "method": without_method,
                    },
                }
                predictions[f"complete_{modality}_with_flags"] = with_scores
                predictions[f"complete_{modality}_without_flags"] = without_scores
            local_missing = []
            for modality in (*MODALITIES, None):
                for rate in config["missingness"]["rates"]:
                    for mask_seed in config["missingness"]["mask_seeds"]:
                        observed = local_missing_availability(
                            x.shape, rate, mask_seed, modality
                        )
                        masked_base = predictor.predict(
                            feature_arrays, observed, test_indices
                        )
                        with_correction, with_method, with_scores = evaluate(
                            with_flags_raw,
                            with_flags_generic,
                            observed,
                            masked_base,
                        )
                        without_correction, without_method, without_scores = evaluate(
                            without_flags_raw,
                            without_flags_generic,
                            observed,
                            masked_base,
                        )
                        condition = {
                            "modality": modality or "mixed",
                            "missing_rate": rate,
                            "mask_seed": mask_seed,
                            "realized_missing_rate": float(
                                1 - observed[test_indices].mean()
                            ),
                            "with_flags": {
                                "correction_only": with_correction,
                                "method": with_method,
                            },
                            "without_flags": {
                                "correction_only": without_correction,
                                "method": without_method,
                            },
                        }
                        local_missing.append(condition)
                        key = f"local_{modality or 'mixed'}_{rate}_{mask_seed}"
                        predictions[f"{key}_with_flags"] = with_scores
                        predictions[f"{key}_without_flags"] = without_scores
            record = {
                "seed": seed,
                "selected_generic_weight_from_parent": generic_weight,
                "correction_only": full_correction,
                "full": full,
                "without_flags_full": {
                    "correction_only": no_flags_full_correction,
                    "method": no_flags_full,
                },
                "complete_missing": complete_missing,
                "local_missing": local_missing,
                "evaluation_scope": (
                    "Same missing masks and condition-specific parent base "
                    "checkpoint; augmentation disabled; availability flags "
                    "retained or removed factorially."
                ),
            }
            atomic_json(
                output
                / "supplementary-isolated-missing-ablation"
                / dataset
                / f"seed-{seed}.json",
                record,
            )
            np.savez_compressed(
                output
                / "supplementary-isolated-missing-ablation"
                / dataset
                / f"seed-{seed}-predictions.npz",
                **predictions,
            )
            predictor.close()
            dataset_records.append(record)
        results["datasets"][dataset] = dataset_records
    atomic_json(
        output / "supplementary-isolated-missing-ablation" / "summary.json",
        results,
    )
    print(output / "supplementary-isolated-missing-ablation" / "summary.json")


if __name__ == "__main__":
    main()
