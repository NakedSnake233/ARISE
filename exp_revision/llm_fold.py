"""Fold-isolated ReAct/mABC with original workflows and larger token ceilings."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

import numpy as np

from .common import ROOT,config_snapshot,json_dump,sha256
from .fold_data import load_fold as legacy_fold
from .baseline_fold import inner_indices as legacy_inner,make_bundle
from . import re2_data as re2

sys.path.insert(0,str(ROOT/"scripts/care"))
import fse2027_llm_baselines as workflow
import fse2027_llm_client as client


def run(dataset,fold,repeat,method,output,smoke=False,expected_model=None):
    start=dt.datetime.now(dt.timezone.utc).isoformat();clock=time.perf_counter()
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    data=re2.load_fold(dataset,fold) if dataset in re2.DATASETS else legacy_fold(dataset,fold)
    fit_ids,validation=re2.inner_indices(data) if dataset in re2.DATASETS else legacy_inner(data)
    bundle=make_bundle(data,fit_ids,validation)
    bundle.test=data.test_indices[:1] if smoke else data.test_indices
    builder=workflow.TelemetryBuilder(bundle)
    prompts=json.loads((ROOT/"scripts/care/fse2027_llm_prompts_budget_v2.json").read_text())
    prompts["react"]["maximum_output_tokens"]=8192
    prompts["mabc"]["maximum_role_output_tokens"]=8192
    prompts["mabc"]["maximum_aggregation_output_tokens"]=8192
    json_dump(output/"prompts.json",prompts)
    calls=[];returned_models=set()

    def call(path,messages,max_tokens):
        if path.exists():record=json.loads(path.read_text())
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                record=client.request_chat(messages,max_tokens=max_tokens,timeout_seconds=90,retries=1,run_seed=42+repeat)
            except client.LLMClientError as error:
                record=error.record
                record["call_date_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
                json_dump(path,record)
                # Do not silently replace infrastructure failures with a local
                # heuristic ranking, or continue billing an unavailable API.
                raise
            record["call_date_utc"]=dt.datetime.now(dt.timezone.utc).isoformat();json_dump(path,record)
        response=record.get("response") or {}; raw=record.get("provider_response") or {}
        returned=raw.get("model") or response.get("model")
        if returned:returned_models.add(returned)
        if expected_model is not None and returned!=expected_model:
            raise RuntimeError("provider model differs from the observed campaign snapshot")
        if len(returned_models)>1:raise RuntimeError("provider model changed within the frozen fold run")
        calls.append(dict(path=str(path.relative_to(ROOT)),date=record.get("call_date_utc"),returned_model=returned,
            status=raw.get("status"),truncated=raw.get("status")!="completed" or response.get("choices",[{}])[0].get("finish_reason")=="length",
            attempts=record.get("attempts"),usage=raw.get("usage") or response.get("usage"),wall_seconds=record.get("wall_seconds")))
        return record

    def content(record):
        if record.get("error"):return None,"provider_error"
        response=record.get("response") or {};raw=record.get("provider_response") or {}
        if raw.get("status")!="completed":return None,"generation_truncated_or_incomplete"
        try:return str(response["choices"][0]["message"]["content"]),None
        except (KeyError,IndexError,TypeError):return None,"missing_response_content"

    old_call,old_content=workflow.call_cached,workflow.response_content
    workflow.call_cached,workflow.response_content=call,content
    values=[];failed=[];audits=[]
    try:
        for index in bundle.test:
            function=workflow.run_react_incident if method=="ReAct" else workflow.run_mabc_incident
            ranking,audit=function(builder,int(index),output,prompts)
            order=[data.services.index(s) for s in ranking]
            scores=workflow.ranking_scores(order,len(data.services));values.append(scores)
            failed.append(not audit["valid"]);audit["incident_id"]=(data.train_ids+data.test_ids)[int(index)];audits.append(audit)
            print(json.dumps(dict(method=method,incident_id=audit["incident_id"],valid=audit["valid"],calls=len(calls))),flush=True)
    finally:workflow.call_cached,workflow.response_content=old_call,old_content
    ids=data.test_ids[:len(bundle.test)]
    np.savez_compressed(output/"scores.npz",scores=np.asarray(values)[None],failed=np.asarray(failed)[None],incident_ids=ids,scenario_keys=["S0_none"])
    json_dump(output/"llm_audit.json",dict(incidents=audits,calls=calls))
    truncation_rate=sum(c["truncated"] for c in calls)/max(len(calls),1);failure_rate=float(np.mean(failed))
    meta=config_snapshot("python -m exp_revision.llm_fold",dict(task="T7",component="llm_baseline",dataset=dataset,fold=fold,repeat=repeat,method=method,smoke=smoke,
        requested_model=client.MODEL,max_output_tokens=8192,scenarios=["S0_none"],expected_model=expected_model))
    good=truncation_rate<.05 and failure_rate<=.1
    meta.update(status="completed_llm_scores" if good else "blocked_llm_quality_gate",errors=[] if good else [dict(error="truncation or invalid ranking gate")],
        start_time=start,end_time=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-clock,
        scores_sha256=sha256(output/"scores.npz"),requested_model=client.MODEL,returned_models=sorted(returned_models),
        public_model_reference="https://developers.openai.com/api/docs/models/gpt-5.6-luna",call_date_utc=start,
        truncation_rate=truncation_rate,failure_rate=failure_rate,call_count=len(calls),seed_sent=False,
        history_fit_ids=[data.train_ids[int(i)] for i in fit_ids],prompt_amendment="8192 output tokens only; original tools, seven roles and workflows unchanged")
    json_dump(output/"run_meta.json",meta)
    if not good:raise RuntimeError("LLM quality gate; preserve actual failed predictions and call records")


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--fold",type=int,required=True)
    p.add_argument("--repeat",type=int,choices=range(3),required=True);p.add_argument("--method",choices=("ReAct","mABC"),required=True)
    p.add_argument("--smoke",action="store_true");p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected-model")
    a=p.parse_args();run(a.dataset,a.fold,a.repeat,a.method,a.output,a.smoke,a.expected_model)


if __name__=="__main__":main()
