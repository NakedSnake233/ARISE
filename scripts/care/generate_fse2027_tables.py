"""Generate FSE manuscript tables only from retained machine-readable results."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "artifacts/care/fse2027-baselines/main_results.csv"
OUT = (
    ROOT / "Paper/fse2027_revision_20260926/figures"
    if (ROOT / "Paper/fse2027_revision_20260926/main.tex").is_file()
    else ROOT / "generated"
)
METRICS = ("PR@1", "PR@3", "PR@5", "MAP@3", "MAP@5")
ORDER = (
    "MicroScope", "MicroRCA", "SBLD", "LogFaultFlagger", "TraceAnomaly",
    "MicroRank", "Nezha", "Eadro", "MULAN", "ReAct", "mABC", "CARE+GPT", "CARE",
)


def number(value: str) -> float | None:
    return None if value in ("", None) else float(value)


def cell(row: dict[str, str], metric: str, maximum: float) -> str:
    mean = 100 * float(row[f"{metric}_mean"])
    sd = number(row[f"{metric}_std"])
    low = 100 * float(row[f"{metric}_ci_low"])
    high = 100 * float(row[f"{metric}_ci_high"])
    estimate = f"{mean:.2f}" if sd is None else f"{mean:.2f}\\,$\\pm$\\,{100 * sd:.2f}"
    if abs(mean - maximum) < 5e-7:
        estimate = "\\textbf{" + estimate + "}"
    return f"\\shortstack{{{estimate}\\\\{{\\scriptsize [{low:.2f},{high:.2f}]}}}}"


def main() -> None:
    global CSV, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=CSV.parent)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    CSV = args.results_dir.resolve() / 'main_results.csv'
    OUT = args.output_dir.resolve()
    rows = list(csv.DictReader(CSV.open()))
    by_dataset = {dataset: {row["method"]: row for row in rows if row["dataset"] == dataset}
                  for dataset in ("trainticket", "aiops2022")}
    lines = []
    for dataset, label in (("trainticket", "TrainTicket ($n=30$)"), ("aiops2022", "AIOps2022 ($n=26$)")):
        current = by_dataset[dataset]
        maxima = {metric: max(100 * float(row[f"{metric}_mean"]) for row in current.values()) for metric in METRICS}
        tag = "tab:main" if dataset == "trainticket" else "tab:main-aiops"
        lines.extend([
            r"\begin{table}[t]", r"\centering",
            rf"\caption{{Matched {label} service ranking (\%). Brackets are 95\% incident-bootstrap intervals; $\pm$ is run SD (five model seeds or three API runs), omitted for deterministic methods. $\dagger$ marks a material adapter. Bold is the numerical maximum. Comparisons are exploratory.}}",
            rf"\label{{{tag}}}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
            r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
            r"Method & PR@1 & PR@3 & PR@5 & MAP@3 & MAP@5 \\", r"\midrule",
        ])
        for method in ORDER:
            if method not in current:
                if method in ("ReAct", "mABC", "CARE+GPT"):
                    lines.append(rf"{method}$^{{\dagger}}$ & \multicolumn{{5}}{{c}}{{\textit{{Incomplete API execution}}}} \\")
                continue
            row = current[method]
            mark = "" if method == "CARE" else r"$^{\dagger}$"
            name = r"\textbf{CARE}" if method == "CARE" else method
            values = " & ".join(cell(row, metric, maxima[metric]) for metric in METRICS)
            lines.append(f"{name}{mark} & {values} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_main_results.tex").write_text("\n".join(lines))
    generate_cost(rows)


def generate_cost(rows: list[dict[str, str]]) -> None:
    """Keep scoring and HTTP timings distinct; never fill incomplete API rows."""
    accounting_path = CSV.parent / "llm_accounting.json"
    study_path = ROOT / "artifacts/care/review-response-v1/report/aggregate.json"
    if not accounting_path.is_file() or not study_path.is_file():
        return
    accounting = json.loads(accounting_path.read_text())
    study = json.loads(study_path.read_text())
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Measured scoring and API cost per incident (per incident--run decision for API treatments). CARE time is loaded base plus evidence-branch scoring; API time is summed successful HTTP wall time. These are different timing scopes and both exclude preparation. Incomplete API treatments are omitted.}",
        r"\label{tab:cost}", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule", r"Method & Dataset & PR@1 (\%) & Time (s) & Calls & Tokens \\",
        r"\midrule",
    ]
    for dataset, label in (("trainticket", "TrainTicket"), ("aiops2022", "AIOps2022")):
        care = next(row for row in rows if row["dataset"] == dataset and row["method"] == "CARE")
        timing = study["datasets"][dataset]["timing"]["loaded_single_incident_latency_ms"]["total"]["mean_ms"]
        lines.append(f"CARE & {label} & {100 * float(care['PR@1_mean']):.2f} & {timing/1000:.3f} & 0 & 0 \\\\")
        for method, value in accounting["datasets"][dataset]["methods"].items():
            if value["status"] != "complete":
                continue
            row = next(row for row in rows if row["dataset"] == dataset and row["method"] == method)
            lines.append(f"{method}$^{{\\dagger}}$ & {label} & {100 * float(row['PR@1_mean']):.2f} & {value['successful_call_sum_seconds_per_incident']:.2f} & {value['calls_per_incident']:.2f} & {value['tokens_per_incident']:,.0f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "table_cost.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
