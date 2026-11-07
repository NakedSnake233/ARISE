"""Measure complete TT suites before admitting their full CPU matrices.

The frozen original manifest generator produces staged full manifests. Two
ordinary full jobs become pilots and are reused when those manifests join.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from .common import ROOT, json_dump, sha256
from . import variant_manifest as original
from .remaining_queue import PRECHARGED, verified


STAGED = ROOT / "manifests/staged_remaining"
PILOT = ROOT / "manifests/remaining/17_pilot_evidence_variants.json"


def stage():
    STAGED.mkdir(parents=True, exist_ok=True)
    if all((STAGED / f"{n:02d}_full_{task.lower()}_evidence.json").exists() for task, n in (("T3", 8), ("T6", 10))): return
    with tempfile.TemporaryDirectory(prefix="arise-formal-pilots-", dir="/tmp") as temp:
        folder = Path(temp); (folder / "manifests/remaining").mkdir(parents=True)
        for name in (".git", "exp_revision", "cache", "results_v2", "scripts", "configs", "splits", "third_party", "data", "artifacts"):
            (folder / name).symlink_to(ROOT / name, target_is_directory=True)
        parent = ROOT / "manifests/remaining/06_full_evidence_masks.json"
        (folder / "manifests/remaining" / parent.name).symlink_to(parent)
        previous = original.ROOT; original.ROOT = folder
        try:
            for task in ("T3", "T6"):
                path = original.create(task)
                target = STAGED / path.name
                if target.exists():
                    if target.read_bytes() != path.read_bytes(): raise FileExistsError("immutable staged matrix differs")
                else: shutil.copyfile(path, target)
        finally: original.ROOT = previous


def create():
    stage(); jobs, sources, commits = [], {}, set()
    for path in sorted(STAGED.glob("*_full_*_evidence.json")):
        manifest = json.loads(path.read_text()); sources.update(manifest["source_sha256"]); commits.add(manifest["source_commit"])
        sources[str(path.relative_to(ROOT))] = sha256(path)
        jobs += [job for job in manifest["jobs"] if job["id"].endswith("tt_fold0_seed42")]
    if len(jobs) != 2 or len(commits) != 1: raise ValueError("expected the two unchanged full TT pilots")
    sources["exp_revision/variant_pilot_manifest.py"] = sha256(Path(__file__))
    manifest = dict(backend="local", phase="formal_TT_evidence_pilots", source_commit=commits.pop(),
        precharged_hours=PRECHARGED, source_sha256=sources, jobs=jobs)
    if PILOT.exists():
        if json.loads(PILOT.read_text()) != manifest: raise FileExistsError("immutable pilot manifest differs")
    else: json_dump(PILOT, manifest)
    return PILOT


def promote():
    manifest = json.loads(PILOT.read_text())
    state = json.loads((ROOT / "results_v2/remaining_queue/queue_state.json").read_text())
    for job in manifest["jobs"]:
        row = state["jobs"][job["id"]]
        if row["status"] != "completed" or not verified(ROOT / row["output"], job): raise ValueError("formal CPU pilot incomplete")
    paths = []
    for path in sorted(STAGED.glob("*_full_*_evidence.json")):
        full = json.loads(path.read_text())
        for name, digest in full["source_sha256"].items():
            if sha256(ROOT / name) != digest: raise ValueError("staged matrix source changed")
        target = ROOT / "manifests/remaining" / path.name
        if target.exists():
            if target.read_bytes() != path.read_bytes(): raise FileExistsError("immutable formal matrix differs")
        else: json_dump(target, full)
        paths.append(str(target))
    return paths


def main():
    p = argparse.ArgumentParser(); p.add_argument("--promote", action="store_true"); a = p.parse_args()
    print(promote() if a.promote else create())


if __name__ == "__main__": main()
