"""Run the unchanged RE2 preprocessing with progress and deadline provenance.

The first SS/TT attempts exceeded 900 seconds while fitting large log corpora.
This wrapper only observes phases; Drain settings, message order and tensors
remain those of the frozen implementation. Recovery uses a new attempt folder.
"""
from __future__ import annotations
import argparse
import json
import time

from .common import ROOT, json_dump, sha256
from . import re2_data as data


def prepare(dataset, fold, output, original_id=None):
    import drain3
    original_miner = drain3.TemplateMiner
    original_disjoint = data._templates_disjoint
    original_normalize = data._normalize
    original_dump = data.json_dump
    started = time.perf_counter()

    def progress(phase, **values):
        print(json.dumps(dict(phase=phase, dataset=dataset, fold=fold,
            seconds=round(time.perf_counter()-started, 3), **values)), flush=True)

    class ObservedMiner(original_miner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fit_calls = self.match_calls = 0
            self.last_progress = time.perf_counter()

        def observe(self, phase):
            if time.perf_counter()-self.last_progress >= 20:
                progress(phase, unique_training_messages=self.fit_calls,
                    matches=self.match_calls, clusters=len(self.drain.clusters))
                self.last_progress = time.perf_counter()

        def add_log_message(self, *args, **kwargs):
            result = super().add_log_message(*args, **kwargs)
            self.fit_calls += 1
            self.observe("drain_fit")
            return result

        def match(self, *args, **kwargs):
            result = super().match(*args, **kwargs)
            self.match_calls += 1
            self.observe("drain_match")
            return result

    def disjoint(templates):
        progress("template_overlap_check_start", templates=len(templates))
        result = original_disjoint(templates)
        progress("template_overlap_check_end", disjoint=result)
        return result

    def normalize(*args, **kwargs):
        progress("normalize_start", shape=list(args[0].shape))
        result = original_normalize(*args, **kwargs)
        progress("normalize_end")
        return result

    def dump(path, value):
        # The queue must never accept a recovered artifact before its final
        # provenance is durable, including a timeout between these two writes.
        if original_id and path == output / "run_meta.json":
            value = {**value, "status": "pending_recovery_receipt"}
        return original_dump(path, value)

    drain3.TemplateMiner = ObservedMiner
    data._templates_disjoint = disjoint
    data._normalize = normalize
    data.json_dump = dump
    try:
        progress("preprocessing_start")
        data.prepare_fold(dataset, fold, output)
    finally:
        drain3.TemplateMiner = original_miner
        data._templates_disjoint = original_disjoint
        data._normalize = original_normalize
        data.json_dump = original_dump
    if original_id:
        path = output / "run_meta.json"
        meta = json.loads(path.read_text())
        meta.update(status="completed_re2_fold", completed_recovery_receipt=True,
            recovery=dict(original_id=original_id, algorithm_changed=False,
                reason="900-second preprocessing deadline exceeded; original corpus/order/settings retained",
                runtime_source_sha256=sha256(ROOT / "exp_revision/re2_preprocessing_runtime.py")))
        json_dump(path, meta)
    progress("preprocessing_end")


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=data.DATASETS, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--original-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.fold, args.output, args.original_id)


if __name__ == "__main__": main()
