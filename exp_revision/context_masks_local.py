"""Use the completed uniform local TT matrix without changing its trainer."""
import argparse
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from .common import ROOT
from .infer_context_masks import infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("tt", "aiops"), required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    print(infer(a.dataset, a.fold, a.seed, smoke=a.smoke, output=a.output,
                model_root=ROOT / ("results_v2/T1_context_local_full" if a.dataset == "tt" else "results_v2/T1_context_full"),
                device_name="cuda" if a.dataset == "tt" else "cpu"))


if __name__ == "__main__":
    main()
