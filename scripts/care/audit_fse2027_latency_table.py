"""Fail if any displayed RCA latency differs from the frozen measurements."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fse2027_baseline_common import ROOT


SUMMARY = ROOT / "artifacts/care/fse2027-latency-20260927/summary.json"
TABLE = ROOT / "Paper/fse2027_revision_20260926/figures/table_latency.tex"
METHODS = (
    "MicroScope", "MicroRCA", "SBLD", "LogFaultFlagger", "TraceAnomaly",
    "MicroRank", "Nezha", "Eadro", "MULAN", "CARE", "CARE+GPT", "ReAct", "mABC",
)


def main():
    data = json.loads(SUMMARY.read_text())["methods"]
    lines = TABLE.read_text().splitlines()
    checked = 0
    for method in METHODS:
        matching = [line for line in lines if line.startswith(method + " ") or line.startswith(method + "$^")]
        if len(matching) != 1:
            raise AssertionError((method, "expected one table row", len(matching)))
        numbers = [float(value) for value in re.findall(r"\d+\.\d+", matching[0])]
        if len(numbers) != 4:
            raise AssertionError((method, numbers))
        scale = 1000 if method in ("CARE+GPT", "ReAct", "mABC") else 1
        expected = [
            data["trainticket"][method]["mean_ms"] / scale,
            data["trainticket"][method]["p95_ms"] / scale,
            data["aiops2022"][method]["mean_ms"] / scale,
            data["aiops2022"][method]["p95_ms"] / scale,
        ]
        for displayed, value in zip(numbers, expected):
            if f"{value:.2f}" != f"{displayed:.2f}":
                raise AssertionError((method, displayed, value))
            checked += 1
    print(f"Verified {checked} displayed latency values against {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
