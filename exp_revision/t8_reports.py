"""Report complete extension groups with event-paired five/three-repeat statistics."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .common import ROOT,config_snapshot,json_dump,sha256
from . import remaining_queue as q
from . import t8_admission as admission
from .report import metric_values,paired

METRICS=("PR@1","PR@3","PR@5","MAP@3","MAP@5","MRR")
LLMS={"ReAct","mABC"}


def statistics(frame):
    keys=["dataset","scenario","scenario_param"]
    if frame.duplicated(keys+["method","incident_id","seed"]).any():raise ValueError("duplicate report predictions")
    records=[]
    for key,group in frame.groupby(keys,sort=True,dropna=False):
        for metric in METRICS:
            means={};runs={}
            for method,current in group.groupby("method",sort=True):
                seeds=list(range(42,45 if method in LLMS else 47))
                values=current.assign(value=metric_values(current.root_rank,metric))
                event=values.groupby("incident_id",sort=True).value.agg(["mean","count"])
                if set(current.seed)!=set(seeds) or (event["count"]!=len(seeds)).any():
                    raise ValueError("incomplete method seed/repeat cohort")
                means[method]=event["mean"];runs[method]=values.groupby("seed").value.mean().reindex(seeds)
            reference=means.get("ARISE");indices=[]
            for method,event in means.items():
                if reference is not None and not event.index.equals(reference.index):raise ValueError("paired event denominator differs")
                scores=runs[method].to_numpy();delta=lo=hi=p=float("nan")
                if reference is not None:delta,lo,hi,p=paired(event.to_numpy(),reference.to_numpy())
                records.append(dict(dataset=key[0],scenario=key[1],scenario_param=key[2],method=method,metric=metric,
                    mean=100*scores.mean(),sd=100*scores.std(ddof=1),events=len(event),repeats=len(scores),
                    delta_vs_ARISE=100*delta,ci_low=100*lo,ci_high=100*hi,p_raw=p,p_holm=p,
                    relative_change_percent=100*delta/float(reference.mean()) if reference is not None and reference.mean()!=0 else float("nan")))
                if reference is not None and method!="ARISE":indices.append(len(records)-1)
            if indices:
                adjusted=multipletests([records[i]["p_raw"] for i in indices],method="holm")[1]
                for i,value in zip(indices,adjusted):records[i]["p_holm"]=float(value)
    return pd.DataFrame(records)


def table(frame,path):
    frame.to_csv(path.with_suffix(".csv"),index=False)
    def cell(value):
        return "NA" if pd.isna(value) else f"{value:.2f}" if isinstance(value,(float,np.floating)) else str(value)
    rows=[[cell(v) for v in row] for row in frame.itertuples(index=False,name=None)]
    header=list(frame.columns)
    path.with_suffix(".md").write_text("| "+" | ".join(header)+" |\n| "+" | ".join(["---"]*len(header))+" |\n"+
        "".join("| "+" | ".join(row)+" |\n" for row in rows))
    def tex(value):
        for old,new in (("\\","\\textbackslash{}"),("_","\\_"),("%","\\%"),("&","\\&"),("#","\\#")):
            value=value.replace(old,new)
        return value
    path.with_suffix(".tex").write_text("\\begin{tabular}{"+"l"*len(header)+"}\n\\toprule\n"+
        " & ".join(map(tex,header))+" \\\\\n\\midrule\n"+"".join(" & ".join(map(tex,row))+" \\\\\n" for row in rows)+"\\bottomrule\n\\end{tabular}\n")


def inputs():
    current=admission.state();collected=[]
    for name,row in sorted(current["jobs"].items()):
        if row["spec"]["expected_config"].get("component")!="extension_collection" or row["status"]!="completed":continue
        folder=ROOT/row["output"]
        if not q.verified(folder,row["spec"]):raise ValueError("unverified extension collection")
        meta=json.loads((folder/"run_meta.json").read_text())
        for relative,digest in meta["artifact_sha256"].items():
            if sha256(ROOT/relative)!=digest:raise ValueError("collection product changed")
        collected.append(dict(job=name,folder=row["output"]))
    return collected


def create():
    collected=inputs()
    if not collected:return None
    sources={}
    for name in ("exp_revision/t8_reports.py","exp_revision/report.py","exp_revision/common.py","exp_revision/remaining_queue.py",
            "exp_revision/t8_admission.py","exp_revision/extended_queue.py","exp_revision/disk_guard.py","exp_revision/disk_guard_local.py",
            "results_v2/T2/predictions.parquet","results_v2/T2/realized_rates.csv","results_v2/T8_inputs/realized_rates.csv"):
        sources[name]=sha256(ROOT/name)
    for item in collected:
        folder=ROOT/item["folder"]
        for path in (folder/"run_meta.json",folder/"scores.npz",folder/"products/predictions.parquet",folder/"products/evaluation_meta.json"):
            sources[str(path.relative_to(ROOT))]=sha256(path)
    fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:16]
    current=admission.state();pilot_id="smoke_t8_report_"+fingerprint
    smoke=not admission.verified_job(pilot_id,current)
    path=q.MANIFESTS/f"61_{'smoke' if smoke else 'full'}_t8_report_{fingerprint}.json"
    if path.exists():return None
    cfg=dict(task="T10",component="extension_report",fingerprint=fingerprint,smoke=smoke)
    job=dict(id=("smoke_" if smoke else "full_")+"t8_report_"+fingerprint,resource="cpu",cores=1,max_attempts=2,
        timeout_seconds=300,source_commit=admission.commit(),expected_config=cfg,
        command=["-m","exp_revision.t8_reports","--manifest",str(path.relative_to(ROOT))],test_ids=["report_tables"],
        score_shape=[1,1,1],expected_status=["completed_t8_report"])
    projection=dict(kind="report",forecast_hours=dict(cpu=310/3600,gpu=0.),
        details=[dict(component="extension_report",dataset=None,per_job_hours=310/3600)],basis="guarded300s report job")
    admission.admit_forecast(projection,"report_"+fingerprint)
    json_dump(path,dict(backend="local",phase="pilot_t8_report" if smoke else "full_t8_report",source_commit=admission.commit(),
        precharged_hours=q.PRECHARGED,source_sha256=sources,jobs=[job],forecast=projection,report_inputs=collected))
    return path


def run(manifest_path,output):
    started=dt.datetime.now(dt.timezone.utc).isoformat();clock=time.perf_counter()
    manifest=json.loads(manifest_path.read_text());cfg=manifest["jobs"][0]["expected_config"]
    for relative,digest in manifest["source_sha256"].items():
        if sha256(ROOT/relative)!=digest:raise ValueError("report input changed: "+relative)
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True);products=output/"products";products.mkdir()
    old=pd.read_parquet(ROOT/"results_v2/T2/predictions.parquet")
    old=old.rename(columns={"rank_of_root":"root_rank"});old["scenario_param"]=old.scenario_param.fillna("")
    old=old[old.scenario!="S4"]
    parts=[old]
    for item in manifest["report_inputs"]:
        parts.append(pd.read_parquet(ROOT/item["folder"]/"products/predictions.parquet"))
    frame=pd.concat(parts,ignore_index=True);frame["scenario_param"]=frame.scenario_param.fillna("")
    if cfg["smoke"]:
        selected={ds:set(sorted(g.incident_id.unique())[:5]) for ds,g in frame.groupby("dataset")}
        frame=frame[frame.apply(lambda r:r.incident_id in selected[r.dataset],axis=1)]
    stats=statistics(frame)
    table(stats[stats.scenario=="S0"],products/"tab_main");table(stats,products/"tab_missing")
    failure=frame.groupby(["dataset","method","scenario","scenario_param"],dropna=False).agg(
        predictions=("status","size"),failed=("status",lambda x:int((x=="failed").sum()))).reset_index()
    table(failure,products/"failed_counts")
    rates=pd.concat([pd.read_csv(ROOT/"results_v2/T2/realized_rates.csv",keep_default_na=False,dtype={"scenario_param":str}),
        pd.read_csv(ROOT/"results_v2/T8_inputs/realized_rates.csv",keep_default_na=False,dtype={"scenario_param":str}).rename(
            columns={"available_cells":"original_available","deleted_cells":"deleted"})],ignore_index=True)
    curves=[];auc=[]
    for (ds,method),group in frame.groupby(["dataset","method"]):
        if method in LLMS:continue
        scenes=[("S0",""),("S2","0.10"),("S2","0.25"),("S2","0.50"),("S2","0.75")]
        if any(not ((group.scenario==s)&(group.scenario_param==p)).any() for s,p in scenes):continue
        for metric in ("PR@1","MAP@5"):
            values=[]
            for seed in range(42,47):
                xs=[];ys=[]
                for scenario,param in scenes:
                    cohort=group[(group.seed==seed)&(group.scenario==scenario)&(group.scenario_param==param)]
                    r=rates[(rates.dataset==ds)&(rates.seed==seed)&(rates.scenario==scenario)&(rates.scenario_param==param)&rates.incident_id.isin(cohort.incident_id)]
                    if not len(r) or r.original_available.sum()<=0:raise ValueError("missing realized deletion cohort")
                    x=0. if scenario=="S0" else float(r.deleted.sum()/r.original_available.sum())
                    y=100*float(metric_values(cohort.root_rank,metric).mean());xs.append(x);ys.append(y)
                    curves.append(dict(dataset=ds,method=method,metric=metric,seed=seed,scenario=scenario,param=param,realized_rate=x,value=y))
                order=np.argsort(xs);x=np.asarray(xs)[order];y=np.asarray(ys)[order]
                values.append(float(np.trapz(y,x)/(x[-1]-x[0])))
            auc.append(dict(dataset=ds,method=method,metric=metric,mean=float(np.mean(values)),sd=float(np.std(values,ddof=1))))
    points=pd.DataFrame(curves);table(points,products/"degradation_points");table(pd.DataFrame(auc),products/"degradation_auc")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ds,g in points.groupby("dataset"):
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
        for ax,metric in zip(axes,("PR@1","MAP@5")):
            for method,c in g[g.metric==metric].groupby("method"):
                line=c.groupby(["scenario","param"]).agg(x=("realized_rate","mean"),y=("value","mean")).sort_values("x")
                ax.plot(line.x,line.y,marker="o",markersize=3,label=method)
            ax.set(xlabel="Realized deletion rate",ylabel=metric+" (%)",title=ds);ax.grid(alpha=.2)
        axes[-1].legend(fontsize=6,loc="upper left",bbox_to_anchor=(1,1))
        fig.savefig(products/f"fig_degradation_{ds}.pdf");plt.close(fig)
    observed=sorted(frame.dataset.unique());missing_ref=sorted(set(frame.dataset.unique())-set(frame[frame.method=="ARISE"].dataset))
    (products/"summary.md").write_text("# ARISE extension report\n\n"+("SMOKE: five events per dataset; not full results.\n\n" if cfg["smoke"] else "")+
        "Tables use complete collected method groups plus original T2 results. Failure rows remain in denominators.\n\n"+
        "Model/official methods: five seeds; ReAct/mABC: three independent repeats. Means and sample SD are percentages.\n\n"+
        "Paired tests first average each method's own repeats per event, then use10000event bootstrap draws,seed2027,two-sided Wilcoxon and Holm correction within dataset/scenario/metric. Differences are method minus ARISE.\n\n"+
        "Deletion curves use deleted/available cells pooled over evaluated events, not nominal request p.\n\n"+
        "Datasets: "+", ".join(observed)+". Missing complete ARISE reference: "+(", ".join(missing_ref) or "none")+".\n\n"+
        "Incomplete/paused methods are not full comparison entries. See docs/questions.md and queue scientific_blocks for CIRCA failure gates and RCD smoke timeouts. Longer-window execution is deferred by the author.\n")
    np.savez_compressed(output/"scores.npz",scores=np.asarray([[[float(len(stats))]]]),incident_ids=["report_tables"])
    receipt=config_snapshot("python -m exp_revision.t8_reports",cfg)
    receipt.update(status="completed_t8_report",errors=[],start_time=started,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-clock,scores_sha256=sha256(output/"scores.npz"),rows=len(frame),
        artifact_sha256={str(p.relative_to(ROOT)):sha256(p) for p in products.iterdir() if p.is_file()},
        witness_type="report table row count; not RCA score",bootstrap_seed=2027,bootstrap_draws=10000)
    json_dump(output/"run_meta.json",receipt)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.manifest,args.output)


if __name__=="__main__":main()
