"""Verify score batches before reading held-out labels; retain failed rows."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import ROOT, json_dump, load_bundle, sha256
from .fold_data import load_fold
from .missingness import SCENARIOS
from .remaining_queue import verified


def replay_baseline(job, output):
    from . import baseline_fold as b
    meta=json.loads((output/"run_meta.json").read_text())
    cfg=meta["config"];method=cfg["method"];base=method.removesuffix("+MD")
    data=load_fold(cfg["dataset"],cfg["fold"]);count=meta["test_count"]
    config=cfg["hyperparameters"]
    if base in ("Eadro","MULAN","TraceAnomaly"):
        saved=torch.load(output/"model.pt",map_location="cpu",weights_only=False)
        if base=="TraceAnomaly":
            model=b.TraceVAE(data.traces.shape[-1],config["hidden_dim"],config["latent_dim"])
            model.load_state_dict(saved["state_dict"]);model.eval()
            post=(data.traces[len(data.train_ids):len(data.train_ids)+count,:,int(data.metadata["before"]//data.metadata["step"]):]-saved["center"])/saved["scale"]
            flat=torch.as_tensor(post.reshape(-1,post.shape[-1]),dtype=torch.float32)
            with torch.no_grad():
                reconstruction,_,_=model(flat)
                error=(reconstruction-flat).square().mean(-1).numpy().reshape(post.shape[:3])
            scores=error.mean(-1) if config["aggregation"]=="mean" else error.max(-1) if config["aggregation"]=="max" else np.quantile(error,.9,axis=-1)
        else:
            raw={name:getattr(data,name)[len(data.train_ids):len(data.train_ids)+count] for name in b.MODALITIES}
            values,_=b.descriptors(raw,int(data.metadata["before"]//data.metadata["step"]),saved["scaler"])
            widths={name:x.shape[-1] for name,x in values.items()}
            model=b.EadroAdapter(widths,config["hidden_dim"],config["attention_heads"],config["dropout"]) if base=="Eadro" else b.MulanAdapter(widths,config["hidden_dim"])
            model.load_state_dict(saved["state_dict"]);model.eval()
            torch.set_num_threads(1)
            with torch.no_grad():
                inputs={name:torch.as_tensor(x) for name,x in values.items()}
                logits=model(inputs,torch.as_tensor(saved["adjacency"])[None].expand(count,-1,-1)) if base=="Eadro" else model(inputs)[0]
                scores=logits.softmax(-1).numpy()
            if base=="MULAN":
                propagated=np.stack([b.random_walk(saved["adjacency"],row,damping=.85,reverse=True) for row in scores])
                scores=(1-config["graph_mix"])*b.row_unit_interval(scores)+config["graph_mix"]*b.row_unit_interval(propagated)
    else:
        import joblib
        saved=joblib.load(output/"model.joblib")
        bundle=b.make_bundle(data,data.train_indices,np.array([],dtype=int))
        scores=b.classical_scores(bundle,method,config,saved["model"],saved["adjacency"])[len(data.train_ids):len(data.train_ids)+count]
    with np.load(output/"scores.npz") as archive:
        expected=archive["scores"][0]
    difference=float(np.max(np.abs(scores-expected)))
    if difference>1e-6:raise ValueError(f"saved checkpoint replay differs for {job['id']}: {difference}")
    return difference


def evaluate(manifest_path:Path, output:Path):
    manifest=json.loads(manifest_path.read_text())
    for relative,digest in manifest["source_sha256"].items():
        if sha256(ROOT/relative)!=digest:raise ValueError(f"input changed: {relative}")
    state=json.loads((ROOT/"results_v2/remaining_queue/queue_state.json").read_text())
    matrices=[];replays={};failures={}
    for job in manifest["jobs"]:
        row=state["jobs"][job["id"]]
        terminal_failure=row["status"] in ("stuck","failed")
        if row["status"]!="completed" and not terminal_failure:
            raise ValueError(f"incomplete score batch: {job['id']}")
        output_relative=row.get("output") or (row["attempts"][-1]["output"] if row.get("attempts") else None)
        folder=ROOT/output_relative if output_relative else None
        meta_path=folder/"run_meta.json" if folder else None
        meta=json.loads(meta_path.read_text()) if meta_path and meta_path.exists() else {}
        if not terminal_failure and not verified(folder,job):
            raise ValueError(f"unverified successful score batch: {job['id']}")
        if meta and any(meta["config"].get(k)!=v for k,v in job["expected_config"].items()):
            raise ValueError(f"score configuration changed: {job['id']}")
        for group in ("source_sha256","input_sha256","mask_sha256"):
            for relative,digest in meta.get(group,{}).items():
                if sha256(ROOT/relative)!=digest:raise ValueError(f"score input changed: {relative}")
        score_path=folder/"scores.npz" if folder else None
        scores=np.full(job["score_shape"],np.nan,dtype=np.float32)
        digest=None
        if score_path and score_path.exists():
            digest=sha256(score_path)
            if meta.get("scores_sha256") and meta["scores_sha256"]!=digest:
                raise ValueError(f"score archive changed: {job['id']}")
            with np.load(score_path) as archive:
                keys=archive["scenario_keys"].tolist()
                if keys!=[f"{s}_{p or 'none'}" for s,p in SCENARIOS]:raise ValueError("scenario order changed")
                if archive["incident_ids"].tolist()!=job["test_ids"]:raise ValueError("incident order changed")
                scores=archive["scores" if "scores" in archive else "probabilities"]
                if scores.shape!=tuple(job["score_shape"]):raise ValueError("score shape changed")
        if terminal_failure:
            failures[job["id"]]={"output":output_relative,"errors":meta.get("errors",[]),"error_tail":row.get("error_tail","")}
        cfg=job["expected_config"]
        if manifest["phase"]=="smoke" and "method" in cfg and not terminal_failure:
            replays[job["id"]]=replay_baseline(job,folder)
        matrices.append((job,scores,digest))
    # Every score matrix/checkpoint is verified before this first read of
    # held-out root/fault-type labels. These fields are used only for metrics.
    labels={}
    for dataset in {job["expected_config"]["dataset"] for job,_,_ in matrices}:
        bundle=load_bundle(dataset)
        labels[dataset]=(bundle,{r["incident_id"]:(int(bundle.labels[i]),r["fault_type"]) for i,r in enumerate(bundle.rows)})
    with (ROOT/"results_v2/T2/realized_rates.csv").open(newline="") as stream:
        rates={(r["dataset"],r["incident_id"],int(r["seed"]),r["scenario"],r["scenario_param"]):float(r["realized_rate"]) for r in csv.DictReader(stream)}
    predictions=[]
    for job,scores,_ in matrices:
        cfg=job["expected_config"];dataset=cfg["dataset"]
        bundle,truth=labels[dataset]
        method=cfg.get("method","context-only")
        for j,(scenario,param) in enumerate(SCENARIOS):
            for i,key in enumerate(job["test_ids"]):
                root,fault=truth[key]
                valid=bool(np.isfinite(scores[j,i]).all())
                order=np.argsort(-scores[j,i],kind="stable") if valid else np.arange(len(bundle.services))
                predictions.append({"task":"T1" if scenario=="S0" else "T2","method":method,"dataset":dataset,
                    "split":"kfold5","fold":cfg["fold"],"seed":cfg["seed"],"scenario":scenario,"scenario_param":param,
                    "realized_rate":rates[(dataset,key,cfg["seed"],scenario,param)],"incident_id":key,
                    "fault_type":fault,"root_true":bundle.services[root],
                    "rank_of_root":float(np.flatnonzero(order==root)[0]+1) if valid else float("inf"),
                    "top5":json.dumps([bundle.services[int(k)] for k in order[:5]] if valid else [],ensure_ascii=False),
                    "status":"ok" if valid else "failed","invalid_by_design":False,"runtime_ms":None})
    frame=pd.DataFrame(predictions)
    if frame.duplicated(["dataset","method","incident_id","seed","scenario","scenario_param"]).any():raise ValueError("duplicate predictions")
    output.mkdir(parents=True,exist_ok=True)
    path=output/"predictions.parquet"
    if path.exists():
        if not pd.read_parquet(path).equals(frame):raise FileExistsError("immutable predictions differ")
    else:frame.to_parquet(path,index=False)
    summary={"time":dt.datetime.now(dt.timezone.utc).isoformat(),"phase":manifest["phase"],"rows":len(frame),
             "failed":int((frame.status!="ok").sum()),"predictions_sha256":sha256(path),
             "manifest_sha256":sha256(manifest_path),"checkpoint_replay_max_difference":max(replays.values(),default=0.),
             "checkpoint_replay":replays,"terminal_failures":failures,
             "coverage":frame.groupby(["dataset","method"]).size().to_dict()}
    summary["coverage"]={"|".join(k):int(v) for k,v in summary["coverage"].items()}
    groups=frame.assign(failed=frame.status!="ok").groupby(["dataset","method","scenario","scenario_param"],dropna=False)
    summary["failure_rates"]={"|".join(k):float(g.failed.mean()) for k,g in groups}
    summary["failure_threshold_exceeded"]=[k for k,v in summary["failure_rates"].items() if v>.1]
    receipt=output/"evaluation_meta.json"
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old["predictions_sha256"]!=summary["predictions_sha256"]:raise FileExistsError("immutable evaluation differs")
    else:json_dump(receipt,summary)
    if summary["failure_threshold_exceeded"]:
        raise ValueError("prediction failure rate exceeds 10% for "+", ".join(summary["failure_threshold_exceeded"]))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(json.dumps(evaluate(a.manifest,a.output),ensure_ascii=False))


if __name__=="__main__":main()
