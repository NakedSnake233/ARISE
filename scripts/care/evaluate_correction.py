"""Run the actual Qwen3 LoRA corrector on held-out incidents and report measured RCA."""
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from build_sft import evidence
from correct import correct
from train_base import ranking_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/care/aiops2022-prepared'))
    p.add_argument('--base',type=Path,default=Path('artifacts/care/base-aiops2022'))
    p.add_argument('--sft',type=Path,default=Path('data/care/sft-aiops2022'))
    p.add_argument('--model',default='models/Qwen3-8B')
    p.add_argument('--adapter',type=Path,default=Path('artifacts/care/qwen3-care-lora'))
    p.add_argument('--output',type=Path,default=Path('artifacts/care/correction-aiops2022'))
    args=p.parse_args()
    if not (args.adapter/'adapter_config.json').exists():raise ValueError('A trained LoRA adapter is required; no silent base-model substitute')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.data/'incidents.json').read_text());meta=json.loads((args.data/'metadata.json').read_text())
    inputs=json.loads((args.sft/'test_inputs.json').read_text());features=np.load(args.data/'features.npz')
    predictions=np.load(args.base/'predictions.npz');prob=predictions['probabilities'];ev=evidence(features,meta,rows,prob)
    indices=[i for i,row in enumerate(rows) if row['split']=='test']
    if len(inputs)!=len(indices):raise ValueError('Test inputs do not match incident manifest')
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True)
    model=PeftModel.from_pretrained(model,args.adapter).eval();scores=[];latencies=[];invalid=0;changed=0
    for item,i in zip(inputs,indices):
        obs=json.loads(item['input'])
        if obs['incident_id']!=f'case-{i:05d}':raise ValueError('Test input order mismatch')
        messages=[{'role':'user','content':item['instruction']+'\n'+item['input']}]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        x=tokenizer(text,return_tensors='pt').to('cuda:0');start=time.perf_counter()
        with torch.inference_mode():out=model.generate(**x,max_new_tokens=1536,do_sample=False)
        response=tokenizer.decode(out[0][x.input_ids.shape[1]:],skip_special_tokens=True)
        base=dict(zip(meta['services'],map(float,prob[i])));audit={}
        try:
            cleaned=re.sub(r'^```(?:json)?\s*|\s*```$','',response.strip())
            proposal=json.loads(cleaned)
            if set(proposal.get('causal_scores',{}))-set(obs['candidates']):raise ValueError('LLM invented a candidate')
            a=predictions['attention'][i]
            attention={(u,v):float(a[ui,vi]) for ui,u in enumerate(meta['services']) for vi,v in enumerate(meta['services']) if ui!=vi and a[ui,vi]>0}
            audit=correct(base,ev[i],obs['retrieved_facts'],proposal,attention,delta=meta['step'])
            final=audit['scores'];changed+=audit['changed']
        except (ValueError,KeyError,TypeError) as exc:
            final=base;invalid+=1;audit={'reason':'invalid proposal; abstained','error':str(exc)}
        latency=time.perf_counter()-start;latencies.append(latency)
        scores.append([final[s] for s in meta['services']])
        (args.output/f'case-{i:05d}.json').write_text(json.dumps({'input':obs,'response':response,'audit':audit,'seconds':latency},indent=2,ensure_ascii=False)+'\n')
        print(obs['incident_id'],audit.get('reason'),f'{latency:.2f}s',flush=True)
    labels=features['labels'][indices]
    report={'base':ranking_metrics(prob[indices],labels),'care':ranking_metrics(np.asarray(scores),labels),
            'incidents':len(indices),'changed_predictions':changed,'invalid_llm_abstentions':invalid,
            'mean_correction_seconds':float(np.mean(latencies)),
            'gating':{'base_confidence':.8,'minimum_margin':.05,'weights':[.6,.25,.15],
                      'note':'Prespecified reconstruction defaults, not claimed original-paper hyperparameters'},
            'limitations':meta['limitations']}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
