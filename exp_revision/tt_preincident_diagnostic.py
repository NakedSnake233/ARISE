"""Fixed posthoc acquisition-confound probes on self-collected TrainTicket.

Both frozen split protocols are evaluated without choosing splits or parameters
from scores. Predictions read only each fold's training-label manifest; held-out
labels are opened afterwards for scoring and descriptive acquisition statistics.
This diagnostic is not a main RCA baseline and is not a model-selection signal.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import ROOT, dataset_dir, json_dump, sha256
from .preprocess_tt import METRICS


PROTOCOLS = {
    "ordinary_grouped_5fold": ("tt_kfold5.json", "tt_training_labels"),
    "pair_holdout_5fold": ("tt_pairholdout_kfold5.json", "tt_pairholdout_training_labels"),
}


def utc(stamp):
    return dt.datetime.fromtimestamp(stamp, dt.timezone.utc).isoformat()


def predict(features, timestamps, ids, split, training_labels):
    """Prediction boundary: no held-out labels or label-bearing metadata."""
    mapping = {key: i for i, key in enumerate(ids)}
    train = np.asarray([mapping[key] for key in split["train"]])
    test = np.asarray([mapping[key] for key in split["test"]])
    if set(split["train"]) & set(split["test"]):
        raise ValueError("train/test incident overlap")
    if set(training_labels) != set(split["train"]):
        raise ValueError("training labels differ from frozen training cohort")
    finite = np.isfinite(features[train])
    counts = finite.sum(0)
    keep = counts > 0
    means = np.nansum(features[train][:, keep], axis=0) / counts[keep]
    x = features[:, keep]
    x = np.where(np.isfinite(x), x, means)
    scale = x[train].std(0)
    scale[scale < 1e-12] = 1.
    x = (x - means) / scale
    rows = []
    for index in test:
        nearest = {
            "timestamp_only": train[np.argmin(abs(timestamps[train] - timestamps[index]))],
            "preincident_metrics_1nn": train[np.argmin(((x[train] - x[index])**2).sum(1))],
        }
        for probe, source in nearest.items():
            rows.append(dict(probe=probe, incident_id=ids[index], nearest_train_id=ids[source],
                             prediction=training_labels[ids[source]],
                             timestamp_distance_seconds=float(abs(timestamps[source] - timestamps[index]))))
    return rows, dict(train=len(train), test=len(test), retained_features=int(keep.sum()),
                      dropped_all_missing_features=int((~keep).sum()))


def run(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = utc(time.time())
    clock = time.perf_counter()
    cpu = time.process_time()
    plan = dict(scope="posthoc acquisition-confound probe, no target accuracy or parameter/split selection",
                probes=["nearest training observation-start timestamp", "pre-injection metric mean/std Euclidean 1-NN"],
                protocols=PROTOCOLS, metric_window="first 12 five-second bins (60 seconds), verified strictly before injection request",
                preprocessing="drop all-missing training features; training-only mean imputation and standardization",
                ties="first incident in frozen training manifest order", labels="training manifests for prediction; held-out labels only after predictions",
                chronology="descriptive UTC dates/gaps and declared campaign metadata; no inferred acquisition-session boundaries")
    json_dump(output / "plan.json", plan)
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        return json.loads(path.read_text())

    events = read(ROOT / "splits/tt_unlabeled_events.json")
    metadata = read(dataset_dir("tt") / "metadata.json")
    if len(events) != 150 or len(metadata["services"]) != 41 or metadata["step"] != 5 or metadata["before"] != 60:
        raise ValueError("unexpected self-collected TT configuration")
    ids = [row["incident_id"] for row in events]
    timestamps = np.asarray([row["start"] for row in events])
    features, acquisition = [], []
    margins = []
    finite_channels = np.zeros(len(METRICS), dtype=bool)
    for i, row in enumerate(events):
        path = ROOT / f"cache/raw_telemetry_tt/case-{i:03d}.npz"
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        expected = dict(incident_id=row["incident_id"], source=row["source"], start=row["start"], end=row["end"],
                        step=5, bins=48, services=metadata["services"])
        raw_metadata = read(ROOT / row["source"])
        margin = raw_metadata["injection_requested_at"] - (row["start"] + 60)
        if margin <= 0:
            raise ValueError("nominal baseline window reaches injection request")
        margins.append(margin)
        with np.load(path) as archive:
            if json.loads(str(archive["identity_json"].item())) != expected:
                raise ValueError("raw cache identity differs from frozen event manifest")
            before = archive["metric"][:, :12].astype(float)
        if before.shape != (41, 12, 7):
            raise ValueError("unexpected raw metric tensor shape")
        finite_channels |= np.isfinite(before).any((0, 1))
        count = np.isfinite(before).sum(1)
        mean = np.divide(np.nansum(before, axis=1), count, out=np.full(count.shape, np.nan), where=count > 0)
        variance = np.divide(np.nansum((before - mean[:, None])**2, axis=1), count,
                             out=np.full(count.shape, np.nan), where=count > 0)
        features.append(np.concatenate([mean.ravel(), np.sqrt(variance).ravel()]))
        acquisition.append(dict(incident_id=row["incident_id"], start=row["start"], end=row["end"],
                                start_utc=utc(row["start"]), date_utc=utc(row["start"])[:10],
                                injection_requested_at=raw_metadata["injection_requested_at"],
                                injection_confirmed_at=raw_metadata["injection_confirmed_at"],
                                collector_version=raw_metadata.get("collector_version"),
                                declared_session_fields={k: raw_metadata[k] for k in ("session_id", "campaign_id", "batch_id") if k in raw_metadata}))
    features = np.asarray(features)
    rows, cohorts = [], {}
    for protocol, (split_name, training_folder) in PROTOCOLS.items():
        splits = read(ROOT / "splits" / split_name)
        tests = [key for part in splits.values() for key in part["test"]]
        if collections.Counter(tests) != collections.Counter(ids):
            raise ValueError("frozen folds do not partition the 150 evaluation incidents")
        for fold, part in splits.items():
            train_manifest = read(ROOT / "splits" / training_folder / f"{fold}.json")
            train_labels = {item["incident_id"]: item["root_service"] for item in train_manifest}
            predictions, sizes = predict(features, timestamps, ids, part, train_labels)
            rows.extend(dict(protocol=protocol, fold=fold, **item) for item in predictions)
            cohorts[f"{protocol}/{fold}"] = sizes
    # No held-out labels have entered either prediction function above.
    labels = {row["incident_id"]: row for row in read(dataset_dir("tt") / "incidents.json")}
    frame = pd.DataFrame(rows)
    frame["root_service"] = frame.incident_id.map(lambda key: labels[key]["root_service"])
    frame["correct"] = frame.prediction == frame.root_service
    frame.to_csv(output / "predictions.csv", index=False)
    summary = frame.groupby(["protocol", "probe"]).correct.agg(["mean", "sum", "count"]).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    frame.groupby(["protocol", "probe", "fold"]).correct.agg(["mean", "sum", "count"]).to_csv(output / "by_fold.csv")
    for item in acquisition:
        item.update(root_service=labels[item["incident_id"]]["root_service"], fault_type=labels[item["incident_id"]]["fault_type"])
    chronology = pd.DataFrame(acquisition).sort_values("start").reset_index(drop=True)
    chronology["gap_from_previous_end_seconds"] = chronology.start - chronology.end.shift()
    chronology["same_root_as_previous"] = chronology.root_service == chronology.root_service.shift()
    chronology["root_run"] = chronology.root_service.ne(chronology.root_service.shift()).cumsum()
    chronology.to_csv(output / "chronology.csv", index=False)
    chronology.groupby(["date_utc", "root_service"]).size().rename("events").to_csv(output / "date_root_counts.csv")
    pairs = chronology.groupby(["root_service", "fault_type"]).agg(events=("incident_id", "size"), dates=("date_utc", "nunique"),
                              first_start=("start", "min"), last_start=("start", "max"))
    pairs.to_csv(output / "pair_date_span.csv")
    campaign = read(ROOT / "data/care/trainticket-collected/campaign-status.json")
    schedule = read(ROOT / "data/care/trainticket-collected/schedule.json")
    source_collector = ROOT / "scripts/care/collect_trainticket.py"
    inputs[str(source_collector.relative_to(ROOT))] = sha256(source_collector)
    chronology_summary = dict(start_utc=utc(chronology.start.min()), end_utc=utc(chronology.end.max()),
        date_counts={str(k): int(v) for k, v in chronology.date_utc.value_counts().sort_index().items()},
        pairs_spanning_more_than_one_utc_date=int((pairs.dates > 1).sum()), total_pairs=len(pairs),
        adjacent_same_root=int(chronology.same_root_as_previous.iloc[1:].sum()), adjacency_pairs=len(chronology)-1,
        root_run_lengths={str(k): int(v) for k, v in chronology.groupby("root_run").size().value_counts().sort_index().items()},
        gaps_seconds={"min": float(chronology.gap_from_previous_end_seconds.min()), "median": float(chronology.gap_from_previous_end_seconds.median()),
                     "max": float(chronology.gap_from_previous_end_seconds.max())},
        largest_gaps=chronology.nlargest(5, "gap_from_previous_end_seconds")[["incident_id", "start_utc", "gap_from_previous_end_seconds"]].to_dict("records"),
        declared_session_ids_present=any(item["declared_session_fields"] for item in acquisition),
        latest_campaign_record={k: campaign[k] for k in ("state", "started_at", "updated_at") if k in campaign},
        events_before_latest_campaign_start=int((chronology.start < campaign["started_at"]).sum()),
        schedule_entries=len(schedule), schedule_matches_all_incident_ids=set(item["incident_id"] for item in schedule) == set(ids),
        caveat="No per-event acquisition-session ID exists. UTC dates, the latest campaign process timestamp, and observed gaps are descriptions, not validated independent sessions. Collector shuffles one fixed schedule; a later process may resume it.")
    json_dump(output / "chronology_summary.json", chronology_summary)
    receipt = dict(status="completed", scope=plan["scope"], dataset="self-collected TT", started_at=started, completed_at=utc(time.time()),
                   elapsed_seconds=time.perf_counter()-clock, process_cpu_seconds=time.process_time()-cpu,
                   input_sha256=inputs, source_sha256=sha256(Path(__file__)), rows=len(frame), cohorts=cohorts,
                   metric_channels_present=[name for name, present in zip(METRICS, finite_channels) if present],
                   all_missing_metric_channels=[name for name, present in zip(METRICS, finite_channels) if not present],
                   baseline_bins=12, baseline_injection_request_margin_min_seconds=float(min(margins)),
                   limitations=["Timestamp is a diagnostic only; actual models are not shown to ingest it.",
                                "Nonzero pre-injection predictability does not identify its causal source or establish direct target leakage.",
                                "Low probe accuracy does not rule out other nonlinear or cross-modal acquisition shortcuts.",
                                "Only one randomized fixed-load/fixed-injection collection campaign is documented; no genuine cross-campaign validation can be inferred."])
    json_dump(output / "receipt.json", receipt)
    report = ["# Self-collected TT pre-injection diagnostic", "", "Fixed probes; all predictions use training labels only. No tuning against the scores below.", "",
              "| Protocol | Probe | Top1 | Correct / events |", "|---|---|---:|---:|"]
    for row in summary.to_dict("records"):
        report.append(f"| {row['protocol']} | {row['probe']} | {100*row['mean']:.2f}% | {row['sum']}/{row['count']} |")
    retained = sorted({item["retained_features"] for item in cohorts.values()})
    report += ["", f"Metrics: 41 service candidates × 5 observed resource channels; mean/std of 12 strictly pre-injection bins, {retained} usable features per fold after dropping all-missing training features. Minimum gap from baseline end to injection request: {min(margins):.6f} s.",
               "", "The balanced ten-root cohort gives a 10% expected accuracy for random selection among the ten historically observed roots; uniform selection over all 41 service candidates is 2.44%. The nearest-neighbor probes always reuse a historical training root, so 10% is the relevant descriptive prior reference.",
               "", "Acquisition chronology is in chronology_summary.json. Date boundaries and process restarts are not validated independent environments. No new split was selected from these diagnostic scores.", "",
               "Limitations: " + " ".join(receipt["limitations"]), ""]
    (output / "REPORT.md").write_text("\n".join(report))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "review/TT_PREINCIDENT_DIAGNOSTIC_20261003")
    args = parser.parse_args()
    print(run(args.output).to_string(index=False))
