"""Posthoc RE2-TT acquisition-confound probes, never a main RCA baseline.

Fixed before inspecting probe scores: nearest timestamp and Euclidean 1-NN
on pre-injection metric mean/std, with imputation and scaling fitted on each
outer training fold only. No test labels enter either prediction function.
This does not establish that timestamps or identities enter the actual models.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, json_dump, sha256
from .re2_data import raw_locations


def run(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    split_path = ROOT / 'splits/re2_tt_kfold5.json'
    events_path = ROOT / 'splits/re2_tt_unlabeled_events.json'
    labels_path = ROOT / 'splits/re2_tt_evaluation_labels.json'
    splits = json.loads(split_path.read_text())
    events = json.loads(events_path.read_text())
    labels = {r['incident_id']: r for r in json.loads(labels_path.read_text())}
    ids = [r['incident_id'] for r in events]
    timestamps = np.asarray([r['start'] for r in events])
    mapping = {key: i for i, key in enumerate(ids)}
    locations = raw_locations('re2_tt')
    features = []
    for key in ids:
        with np.load(locations[key] / 'raw.npz') as archive:
            before = archive['metrics'][:, :48].astype(float)
        count = np.isfinite(before).sum(1)
        mean = np.divide(np.nansum(before, axis=1), count,
                         out=np.full(count.shape, np.nan), where=count > 0)
        variance = np.divide(np.nansum((before - mean[:, None])**2, axis=1), count,
                             out=np.full(count.shape, np.nan), where=count > 0)
        features.append(np.concatenate([mean.ravel(), np.sqrt(variance).ravel()]))
    features = np.asarray(features)
    rows = []
    for fold, part in splits.items():
        train = np.asarray([mapping[k] for k in part['train']])
        test = np.asarray([mapping[k] for k in part['test']])
        finite = np.isfinite(features[train])
        count = finite.sum(0)
        keep = count > 0
        train_mean = np.nansum(features[train][:, keep], axis=0) / count[keep]
        x = features[:, keep]
        x = np.where(np.isfinite(x), x, train_mean)
        scale = x[train].std(0)
        scale[scale < 1e-12] = 1.
        x = (x - train_mean) / scale
        for index in test:
            nearest = {
                'timestamp_only': train[np.argmin(abs(timestamps[train] - timestamps[index]))],
                'preincident_metrics_1nn': train[np.argmin(((x[train] - x[index])**2).sum(1))],
            }
            for probe, source in nearest.items():
                prediction = labels[ids[source]]['root_service']
                rows.append(dict(probe=probe, fold=fold, incident_id=ids[index],
                                 nearest_train_id=ids[source], prediction=prediction,
                                 correct=prediction == labels[ids[index]]['root_service']))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'predictions.csv', index=False)
    summary = frame.groupby('probe').correct.agg(['mean', 'sum', 'count']).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    json_dump(output / 'receipt.json', dict(
        scope='posthoc acquisition-confound probe; not a main model or tuning signal',
        data='RE2-TT', metrics_interval='48 bins strictly before injection; 15 seconds per bin',
        training='fixed 1-NN, training-only imputation and feature scaling; no hyperparameter search',
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (split_path, events_path, labels_path)},
        source_sha256=sha256(Path(__file__)), rows=len(frame)))
    return summary


if __name__ == '__main__':
    print(run(ROOT / 'review/RE2_PREINCIDENT_DIAGNOSTIC_20261003').to_string(index=False))
