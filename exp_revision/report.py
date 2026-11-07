"""Event-paired statistics and versioned ARISE experiment tables/figures."""
from __future__ import annotations
import argparse
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from .common import ROOT, METRICS, SEEDS, config_snapshot, json_dump, sha256
from . import remaining_queue as q
from .score_receipts import read_job, check_hashes

TASKS = ("T1", "T2", "T3", "T5a", "T5b", "T6")
COLLECTION = dict(T1="main", T2="main", T3="T3", T5a="T5", T5b="T5", T6="T6")


def metric_values(ranks, metric):
    ranks = np.asarray(ranks, dtype=float)
    valid = np.isfinite(ranks) & (ranks >= 1)
    reciprocal = np.divide(1., ranks, out=np.zeros_like(ranks), where=valid)
    if metric == "MRR": return reciprocal
    k = int(metric.split("@")[1]); hit = valid & (ranks <= k)
    return hit.astype(float) if metric.startswith("PR") else np.where(hit, reciprocal, 0.)


@lru_cache(maxsize=32)
def bootstrap_indices(n):
    if n < 1: raise ValueError("empty paired event cohort")
    return np.random.default_rng(2027).integers(0, n, size=(10_000, n), dtype=np.int32)


def paired(method, reference):
    difference = np.asarray(method) - np.asarray(reference)
    if difference.ndim != 1 or not len(difference) or not np.isfinite(difference).all(): raise ValueError("invalid paired event metrics")
    draws = bootstrap_indices(len(difference)); means = []
    for chunk in np.array_split(draws, 40): means.append(difference[chunk].mean(axis=1))
    ci = np.quantile(np.concatenate(means), [.025, .975])
    p = 1. if np.all(difference == 0) else float(wilcoxon(difference, alternative="two-sided").pvalue)
    return float(difference.mean()), float(ci[0]), float(ci[1]), p


def statistics(frame, reference="ARISE", metrics=METRICS, seeds=SEEDS):
    keys = ["dataset", "scenario", "scenario_param"]
    if frame.duplicated(keys + ["method", "incident_id", "seed"]).any(): raise ValueError("duplicate statistical samples")
    records = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        methods = sorted(group.method.unique())
        if reference not in methods: raise ValueError("missing paired reference method")
        for metric in metrics:
            values = group.assign(value=metric_values(group.rank_of_root, metric))
            event_means = {}; seed_means = {}
            for method in methods:
                current = values[values.method == method]
                # Average seeds for each event before bootstrap/Wilcoxon.
                by_event = current.groupby("incident_id", sort=True).value.agg(["mean", "count"])
                counts = by_event["count"]
                if (counts != len(seeds)).any() or set(current.seed) != set(seeds): raise ValueError("incomplete paired seed cohort")
                event_means[method] = by_event["mean"]
                seed_means[method] = current.groupby("seed").value.mean().reindex(seeds).to_numpy()
            baseline = event_means[reference]
            indices = []
            for method in methods:
                if not event_means[method].index.equals(baseline.index): raise ValueError("paired methods have different event denominators")
                per_seed = seed_means[method]
                delta, lo, hi, raw_p = paired(event_means[method].to_numpy(), baseline.to_numpy())
                record = dict(dataset=group_key[0], scenario=group_key[1], scenario_param=group_key[2], method=method, metric=metric,
                    mean=100 * per_seed.mean(), sd=100 * per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan, delta=100 * delta, ci_low=100 * lo, ci_high=100 * hi,
                    p_raw=raw_p, p_holm=1., events=len(baseline), seeds=len(seeds), reference=reference,
                    invalid_by_design=bool(group[group.method == method].invalid_by_design.any()))
                records.append(record)
                if method != reference: indices.append(len(records) - 1)
            if indices:
                adjusted = multipletests([records[i]["p_raw"] for i in indices], method="holm")[1]
                for index, value in zip(indices, adjusted): records[index]["p_holm"] = float(value)
    return pd.DataFrame(records)


def realized_x(rates, dataset, seed, scenario, param):
    if scenario == "S0": return 0.
    current = rates[(rates.dataset == dataset) & (rates.seed == seed) & (rates.scenario == scenario) & (rates.scenario_param == param)]
    denominator = current.original_available.sum()
    if not len(current) or denominator <= 0: raise ValueError("missing realized-rate cohort")
    return float(current.deleted.sum() / denominator)


def degradation(frame, rates, seeds=SEEDS):
    points, auc_rows = [], []
    scenes = [("S0", ""), ("S2", "0.10"), ("S2", "0.25"), ("S2", "0.50"), ("S2", "0.75")]
    for dataset in sorted(frame.dataset.unique()):
        for method in sorted(frame.method.unique()):
            for metric in ("PR@1", "MAP@5"):
                aucs, xs, ys = [], [], []
                for seed in seeds:
                    x, y = [], []
                    for scenario, param in scenes:
                        current = frame[(frame.dataset == dataset) & (frame.method == method) & (frame.seed == seed)
                            & (frame.scenario == scenario) & (frame.scenario_param == param)]
                        if not len(current): raise ValueError("incomplete degradation scenario")
                        x.append(realized_x(rates, dataset, seed, scenario, param)); y.append(100 * metric_values(current.rank_of_root, metric).mean())
                    x, y = np.asarray(x), np.asarray(y); order = np.argsort(x, kind="stable")
                    if x.max() <= x.min(): raise ValueError("zero degradation x span")
                    aucs.append(float(np.trapz(y[order], x[order]) / (x.max() - x.min()))); xs.append(x); ys.append(y)
                for i, (scenario, param) in enumerate(scenes):
                    points.append(dict(dataset=dataset, method=method, metric=metric, scenario=scenario, scenario_param=param,
                        realized_rate=float(np.mean(xs, axis=0)[i]), mean=float(np.mean(ys, axis=0)[i]), sd=float(np.std(ys, axis=0, ddof=1)[i]) if len(seeds)>1 else np.nan))
                auc_rows.append(dict(dataset=dataset, method=method, metric=metric, mean=float(np.mean(aucs)),
                    sd=float(np.std(aucs, ddof=1)) if len(seeds)>1 else np.nan, x_span=float(np.mean([np.ptp(x) for x in xs])), normalization="per-seed trapezoid / actual x span"))
    return pd.DataFrame(points), pd.DataFrame(auc_rows)


def text_cell(value, column):
    if isinstance(value, (float, np.floating)):
        if np.isnan(value): return "NA"
        return f"{value:.4g}" if column.startswith("p_") else f"{value:.2f}"
    return str(value)


def latex_escape(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(mapping.get(c, c) for c in text)


def table(folder, name, frame):
    frame.to_csv(folder / f"{name}.csv", index=False)
    columns = list(frame.columns); cells = [[text_cell(value, col) for col, value in zip(columns, row)] for row in frame.itertuples(index=False, name=None)]
    md = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    md.extend("| " + " | ".join(v.replace("|", r"\|") for v in row) + " |" for row in cells)
    (folder / f"{name}.md").write_text("\n".join(md) + "\n")
    latex = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule", " & ".join(latex_escape(c) for c in columns) + r" \\", r"\midrule"]
    latex.extend(" & ".join(latex_escape(v) for v in row) + r" \\" for row in cells)
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (folder / "latex").mkdir(exist_ok=True); (folder / "latex" / f"{name}.tex").write_text("\n".join(latex) + "\n")


def display_stats(stats):
    frame = stats.copy(); frame["mean_sd"] = [f"{a:.2f} ± {text_cell(b, 'sd')}" for a, b in zip(frame["mean"], frame.sd)]
    frame["CI95"] = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(frame.ci_low, frame.ci_high)]
    return frame[["dataset", "scenario", "scenario_param", "method", "metric", "mean_sd", "delta", "CI95", "p_holm", "events", "reference", "invalid_by_design"]]


def per_fault(frame, seeds=SEEDS):
    result = []
    for dataset, current in frame.groupby("dataset"):
        ranks = current.assign(hit=metric_values(current.rank_of_root, "PR@1"), map5=metric_values(current.rank_of_root, "MAP@5"))
        order = ranks[ranks.method != "ARISE"].groupby("method")[["hit", "map5"]].mean().reset_index().sort_values(["hit", "map5", "method"], ascending=[False, False, True])
        strongest = order.method.iloc[:2].tolist()
        for (method, fault), group in ranks[ranks.method.isin(["ARISE", *strongest])].groupby(["method", "fault_type"]):
            values = group.groupby("seed").hit.mean().reindex(seeds).to_numpy()
            result.append(dict(dataset=dataset, method=method, fault_type=fault, mean=100 * values.mean(), sd=100 * values.std(ddof=1) if len(values)>1 else np.nan,
                events=group.incident_id.nunique(), baseline_selection="top two S0 PR@1; ties MAP@5 then name; reporting only"))
    return pd.DataFrame(result)


def make_figures(folder, points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    for dataset, current in points.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        for ax, metric in zip(axes, ("PR@1", "MAP@5")):
            for i, (method, values) in enumerate(current[current.metric == metric].groupby("method", sort=True)):
                values = values.sort_values("realized_rate")
                ax.plot(100 * values.realized_rate, values["mean"], marker="o", markersize=3, linewidth=2 if method == "ARISE" else 1,
                    linestyle="-" if i < 10 else "--", color=plt.get_cmap("tab20")(i), label=method)
            ax.set_xlabel("Actual deletion rate (%)"); ax.set_ylabel(f"{metric} (%)"); ax.grid(alpha=.2); ax.set_ylim(0, 100)
        axes[-1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=7)
        fig.suptitle("TrainTicket" if dataset == "tt" else "AIOps2022")
        fig.savefig(folder / f"fig_degradation_{dataset}.pdf", bbox_inches="tight"); plt.close(fig)


def generate(output=None, publish=True, smoke=False, tasks=None):
    began, clock = q.now(), time.perf_counter(); stamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    if output is None: output = ROOT / "reports" / "archive" / ("smoke_" if smoke else "") / stamp
    if output.exists() and any(output.iterdir()): raise FileExistsError("immutable report archive exists")
    output.mkdir(parents=True, exist_ok=True); inputs, frames, missing = {}, {}, []
    state = json.loads((q.FOLDER / "queue_state.json").read_text()); seen = set()
    selected = TASKS if tasks is None else tuple(tasks)
    if not set(selected).issubset(TASKS): raise ValueError("unknown report task")
    missing = [task for task in TASKS if task not in selected]
    for task in selected:
        folder = ROOT / "results_v2" / task; path = folder / "predictions.parquet"
        if not path.exists(): missing.append(task); continue
        collection = COLLECTION[task]
        if collection not in seen:
            _, cm, _ = read_job(f"collect_{collection}", state); seen.add(collection)
        else:
            row = state["jobs"][f"collect_{collection}"]; cm = json.loads((ROOT / row["output"] / "run_meta.json").read_text())
        receipt = folder / "evaluation_meta.json"; meta = json.loads(receipt.read_text())
        if sha256(path) != meta["predictions_sha256"] or cm["products"][task]["predictions_sha256"] != meta["predictions_sha256"]:
            raise ValueError("report input prediction receipt changed")
        frame = pd.read_parquet(path)
        if (frame[frame.status == "failed"].rank_of_root != np.inf).any(): raise ValueError("failed event has non-infinite rank")
        if smoke:
            frame = frame[(frame.fold == 0) & (frame.seed == 42)]
            ids = {dataset: sorted(group.incident_id.unique())[:5] for dataset, group in frame.groupby("dataset")}
            frame = frame[[row.incident_id in ids[row.dataset] for row in frame.itertuples(index=False)]]
        frames[task] = frame; inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in (path, receipt)})
    if not {"T1", "T2"}.issubset(frames): raise ValueError("main and missingness matrices must be collected before reporting")
    seeds = [42] if smoke else SEEDS
    stats = {task: statistics(frame, "ARISE[V-c]" if task == "T3" else "ARISE", seeds=seeds) for task, frame in frames.items()}
    # S0 appears in T1 and T2 files for reproducibility; the missingness table excludes it.
    table(output, "tab_main", display_stats(stats["T1"]))
    table(output, "tab_missing", display_stats(stats["T2"][(stats["T2"].scenario != "S0") & stats["T2"].metric.isin(["PR@1", "MAP@5"])]))
    table(output, "tab_per_fault_type", per_fault(frames["T1"], seeds))
    rates_path = ROOT / "results_v2/T2/realized_rates.csv"; rates = pd.read_csv(rates_path, keep_default_na=False)
    main_row = state["jobs"]["collect_main"]
    provenance_path = ROOT / main_row["output"] / "products/run_meta.json"
    provenance = json.loads(provenance_path.read_text())
    check_hashes(provenance["source_sha256"])
    inputs[str(provenance_path.relative_to(ROOT))] = sha256(provenance_path)
    expected_rate_sha = provenance["input_sha256"][str(rates_path.relative_to(ROOT))]
    if sha256(rates_path) != expected_rate_sha: raise ValueError("actual deletion-rate source changed since main collection")
    if smoke:
        ids = {dataset: set(group.incident_id.unique()) for dataset, group in frames["T2"].groupby("dataset")}
        rates = rates[[row.incident_id in ids[row.dataset] for row in rates.itertuples(index=False)]]
    points, auc = degradation(frames["T2"], rates, seeds)
    point_table = points.assign(actual_rate_percent=100 * points.realized_rate)
    table(output, "fig_degradation", point_table); table(output, "tab_degradation_auc", auc); make_figures(output, points)
    for task, name in (("T3", "tab_leakage"), ("T6", "tab_ablation_matrix")):
        if task in stats: table(output, name, display_stats(stats[task][stats[task].metric.isin(["PR@1", "MAP@5"])]))
        else: (output / f"{name}.md").write_text(f"Pending {task}; no completed matrix is available.\n")
    generalization = []
    for task in ("T5a", "T5b"):
        if task in stats:
            current = display_stats(stats[task]); current.insert(0, "task", task); generalization.append(current)
    if generalization: table(output, "tab_generalization", pd.concat(generalization, ignore_index=True))
    else: (output / "tab_generalization.md").write_text("Pending T5a/T5b; no completed matrix is available.\n")
    (output / "tab_timing.md").write_text("T11 has not run; training wall times are not end-to-end inference latency.\n")
    failure_rows = []
    for task, frame in frames.items():
        for (dataset, method, scenario, param), group in frame.groupby(["dataset", "method", "scenario", "scenario_param"], dropna=False):
            failure_rows.append(dict(task=task, dataset=dataset, method=method, scenario=scenario, scenario_param=param,
                rows=len(group), failed=int((group.status == "failed").sum()), fallback=int((group.status == "fallback").sum())))
    table(output, "tab_failures", pd.DataFrame(failure_rows))
    pd.concat([value.assign(task=task) for task, value in stats.items()], ignore_index=True).to_csv(output / "statistics.csv", index=False)
    deviations_path = ROOT / "docs/deviations.md"; deviations = deviations_path.read_text()
    snapshot = output / "deviations_snapshot.md"; snapshot.write_text(deviations); inputs[str(snapshot.relative_to(ROOT))] = sha256(snapshot)
    summary = ["# ARISE experiment results", "", f"Generated {began}; {'SMOKE validation only' if smoke else 'full available matrices'}.", "",
        f"All metrics are percentages. {'Smoke uses fold0, seed42, at most5 events/scenario; SD is unavailable.' if smoke else 'Mean and sample SD use five model seeds.'} Paired tests first average the available seeds for each incident; 10,000 incident bootstraps use seed2027. Wilcoxon tests are two-sided and Holm correction covers all comparators within each dataset/scenario/metric. Delta and CI mean method minus the reference, in percentage points.", "",
        "S2 keeps the restore-one rule; curve coordinates are total deleted / total originally available cells. AUC is computed for each seed with trapezoids and normalized by its actual horizontal span.", "",
        f"Completed matrices: {', '.join(frames)}. Pending matrices: {', '.join(missing) or 'none'}. T0 is saved-prediction verification, not a new historical training run. T6 has no author preregistration (D4). V-a, when present, is deliberately invalid. D1/D2/D3/D6-dependent extensions and a fresh locked test set remain unexecuted. No paper source was edited.", "",
        "S0 results (PR@1 and MAP@5):", "", "| Dataset | Method | PR@1 | MAP@5 |", "| --- | --- | --- | --- |"]
    for dataset in ("tt", "aiops"):
        for method in ("ARISE", "context-only", "ET-spec", "Eadro", "MULAN"):
            current = stats["T1"][(stats["T1"].dataset == dataset) & (stats["T1"].method == method)]
            values = [current[current.metric == metric].iloc[0] for metric in ("PR@1", "MAP@5")]
            summary.append(f"| {dataset} | {method} | {values[0]['mean']:.2f} ± {text_cell(values[0]['sd'], 'sd')} | {values[1]['mean']:.2f} ± {text_cell(values[1]['sd'], 'sd')} |")
    summary.extend(["", "Failed and fallback counts are in tab_failures.csv; denominator includes all events, and failed ranks contribute zero. T1 and T2 share S0 records and are not pooled as independent samples.", "", f"Local queue: {state['counts']}; charged/reserved historical computation: {state['charged_hours']}.", "", "## Recorded deviations", "", deviations])
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    inputs[str(rates_path.relative_to(ROOT))] = sha256(rates_path)
    meta = config_snapshot("python -m exp_revision.report", dict(task="T10", smoke=smoke, completed_tasks=list(frames), pending_tasks=missing,
        bootstrap=10_000, bootstrap_seed=2027, seeds=seeds, pairing_unit=f"incident after mean over {len(seeds)} seeds", delta="method minus reference"))
    meta.update(status="completed_report", start_time=began, end_time=q.now(), elapsed_seconds=time.perf_counter()-clock,
        input_sha256=inputs, source_sha256={"exp_revision/report.py": sha256(Path(__file__))},
        report_packages={name: importlib.metadata.version(name) for name in ("statsmodels", "patsy", "matplotlib")},
        artifact_sha256={str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()})
    json_dump(output / "run_meta.json", meta)
    if publish and not smoke: publish_report(output, meta)
    return meta


def publish_report(output, meta):
    for name, digest in meta["artifact_sha256"].items():
        if sha256(output / name) != digest: raise ValueError("report artifact changed before publication")
    destination = ROOT / "reports"; destination.mkdir(exist_ok=True); (destination / "latex").mkdir(exist_ok=True)
    for path in output.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(output); target.parent.mkdir(exist_ok=True, parents=True)
            if target.resolve() != path.resolve():
                temp = target.with_suffix(target.suffix + ".new"); shutil.copyfile(path, temp); temp.replace(target)
    json_dump(destination / "latest.json", dict(time=q.now(), archive=str(output.relative_to(ROOT)),
        completed_tasks=meta["config"]["completed_tasks"], input_sha256=meta["input_sha256"]))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path); p.add_argument("--no-publish", action="store_true"); p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.output is None and not a.no_publish:
        from . import report_queue, adaptive_queue
        path = report_queue.create(a.smoke); monitor = adaptive_queue.check()
        print(json.dumps(dict(manifest=str(path.relative_to(ROOT)) if path else None, action=monitor["action"]))); return
    meta = generate(a.output, not a.no_publish, a.smoke); print(json.dumps({k: meta[k] for k in ("status", "elapsed_seconds", "config")}))


if __name__ == "__main__": main()
