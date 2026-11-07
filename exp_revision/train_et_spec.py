"""Train the frozen channel-specific ExtraTrees sanity baseline for S0.

No missing-view augmentation, availability gate, indicators, generic expert,
or context fusion is applied. Held-out root labels are never opened here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from .common import ROOT, config_snapshot, json_dump, sha256
from .fold_data import load_fold

sys.path.insert(0, str(ROOT / "scripts/care"))
from robust_evidence import fit_robust_ranker  # noqa: E402


def train(dataset: str, fold: int, seed: int, *, smoke: bool = False,
          output: Path | None = None) -> Path:
    started = time.perf_counter()
    start_time = dt.datetime.now(dt.timezone.utc).isoformat()
    data = load_fold(dataset, fold, smoke=False)
    evidence = ROOT / f"cache/fold_evidence_v2/{dataset}/fold{fold}"
    metadata = json.loads((evidence / "metadata.json").read_text())
    with np.load(evidence / "features.npz") as archive:
        x = archive["x"].astype(np.float32)
        ids = archive["incident_ids"].tolist()
    if (ids != data.train_ids + data.test_ids
            or metadata["train_ids"] != data.train_ids
            or metadata["test_ids"] != data.test_ids
            or x.shape[:2] != (len(ids), len(data.services))
            or x.shape[2] != len(metadata["feature_names"])
            or not np.isfinite(x).all()):
        raise ValueError("ET-spec evidence does not match the frozen outer fold")
    for relative, digest in metadata["input_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"ET-spec evidence source changed: {relative}")
    train_count = len(data.train_ids)
    ranker = fit_robust_ranker(
        x[:train_count], metadata["feature_names"], data.train_labels,
        np.arange(train_count), seed=seed, augment=False, append_flags=False,
        type_agnostic=False, trees=256, min_samples_leaf=3,
    )
    test_ids = data.test_ids[:5] if smoke else data.test_ids
    score = ranker.predict(x[train_count:train_count + len(test_ids)])
    if (score.shape != (len(test_ids), len(data.services))
            or not np.isfinite(score).all() or (score < 0).any()
            or not np.allclose(score.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("invalid full-service ET-spec scores")
    if output is None:
        stage = "T4_et_spec_smoke" if smoke else "T4_et_spec_full"
        output = ROOT / "results_v2" / stage / dataset / f"fold{fold}_seed{seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite ET-spec output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "scores.npz", incident_ids=np.asarray(test_ids),
                        probabilities=score)
    joblib.dump(ranker.model, output / "model.joblib")
    config = {"method": "ET-spec", "dataset": dataset, "fold": fold,
              "seed": seed, "smoke": smoke, "trees": 256,
              "min_samples_leaf": 3, "max_features": 0.7,
              "class_weight": "balanced", "n_jobs": 4,
              "augment": False, "append_availability_flags": False,
              "availability_gate": False, "context_fusion": False}
    meta = config_snapshot(
        f"python -m exp_revision.train_et_spec --dataset {dataset} --fold {fold} --seed {seed}",
        config)
    sources = [ROOT / "exp_revision/train_et_spec.py",
               ROOT / "scripts/care/robust_evidence.py",
               ROOT / "scripts/care/evidence_reranker.py",
               ROOT / f"splits/{dataset}_kfold5.json",
               ROOT / f"splits/{dataset}_training_labels/fold_{fold}.json",
               evidence / "features.npz", evidence / "metadata.json"]
    sources.extend(ROOT / relative for relative in metadata["input_sha256"])
    sources = list(dict.fromkeys(sources))
    meta.update(status="partial_T4_ET_spec_S0", start_time=start_time,
                end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                elapsed_seconds=time.perf_counter() - started,
                fit_count=train_count, test_count=len(test_ids),
                input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sources},
                scores_sha256=sha256(output / "scores.npz"),
                model_sha256=sha256(output / "model.joblib"))
    json_dump(output / "run_meta.json", meta)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("tt", "aiops"))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(train(args.dataset, args.fold, args.seed, smoke=args.smoke,
                output=args.output))


if __name__ == "__main__":
    main()
