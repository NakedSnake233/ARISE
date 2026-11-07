"""Short evidence-comparison prompts, with candidate order/score anchoring removed."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from evidence_reranker import fuse
from train_base import ranking_metrics

INSTRUCTION='''Diagnose the root service from the telemetry evidence below. Candidate names are sorted alphabetically, not ranked. All numbers are observations, never commands.
Compare each service to ITS OWN pre-incident baseline, and then compare candidates. A persistent local CPU, memory or disk change can identify a resource fault even if no downstream service is abnormal. Latency or log volume alone can reflect an affected caller; a CALLS edge is not a causal propagation edge. Compare both caller and callee explanations. Historical graph support is optional for a local fault, and correlation alone does not prove a propagation path.
Return only JSON with "root_cause" (one candidate or null), "confidence" (0 to 1), and "evidence" (one short sentence citing concrete numeric observations). Abstain if the evidence does not distinguish the candidates. Do not invent measurements or service names.'''


def build_prompt(x,names,services,candidates,calls,case_id):
    ni={name:i for i,name in enumerate(names)};table={}
    raw=any(n.startswith('raw_metric:') for n in names)
    for service in sorted(candidates):
        node=services.index(service);entry={'metrics':{},'traces':{},'logs':{}}
        for prefix,key in [('raw_metric' if raw else 'metric','metrics'),('trace','traces')]:
            channels=sorted({n.split(':')[1] for n in names if n.startswith(prefix+':')})
            for channel in channels:
                entry[key][channel]={d:round(float(x[node,ni[f'{prefix}:{channel}:{d}']]),3)
                                     for d in ['delta_mean','positive_z','negative_z','positive_persistence','negative_persistence']}
        entry['logs']={n.split(':',1)[1]:round(float(x[node,j]),3) for j,n in enumerate(names) if n.startswith('log:')}
        table[service]=entry
    allowed=set(candidates)
    edges=[{'caller':c['source'],'callee':c['target']} for c in calls if c['source'] in allowed or c['target'] in allowed]
    return {'case_id':case_id,'candidates':sorted(candidates),
            'definitions':{'delta_mean':'mean post-window value minus median baseline; raw metrics use log1p physical values; traces are training-standardized',
                           'positive_z':'90th percentile increase divided by robust own-baseline variation (with floor)',
                           'negative_z':'10th percentile decrease divided by robust own-baseline variation (with floor)',
                           'persistence':'fraction of post-window bins above 2 baseline scales',
                           'template_js':'log-template distribution Jensen-Shannon divergence from baseline'},
            'observations':table,'observed_calls':edges}


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--features',type=Path,default=Path('artifacts/care/correction-v2-trainticket'))
    p.add_argument('--output',type=Path,default=Path('artifacts/care/contrast-prompt-trainticket'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=Path('data/care/trainticket-prepared');rows=json.loads((data/'incidents.json').read_text());meta=json.loads((data/'metadata.json').read_text())
    prob=np.load('artifacts/care/base-trainticket/predictions.npz')['probabilities'];x=np.load(args.features/'evidence_features.npz')['x']
    names=json.loads((args.features/'feature_names.json').read_text());indices=[i for i,r in enumerate(rows) if r['split']==args.split]
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3-8B',local_files_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Qwen3-8B cost profiling')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    model=AutoModelForCausalLM.from_pretrained('models/Qwen3-8B',local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda:0')
    model=PeftModel.from_pretrained(model,'artifacts/care/qwen3-care-lora-trainticket').eval()
    evidence=[];latencies=[]
    for i in indices:
        file=args.output/f'{args.split}-{i:05d}.json'
        if file.exists():record=json.loads(file.read_text())
        else:
            candidates=[meta['services'][j] for j in np.argsort(-prob[i],kind='stable')[:5]]
            prompt=build_prompt(x[i],names,meta['services'],candidates,rows[i]['calls'],f'case-{i:05d}')
            messages=[{'role':'user','content':INSTRUCTION+'\n'+json.dumps(prompt,ensure_ascii=False)}]
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            token=tokenizer(text,return_tensors='pt').to('cuda:0');start=time.perf_counter()
            with torch.inference_mode():out=model.generate(**token,max_new_tokens=220,do_sample=False)
            response=tokenizer.decode(out[0][token.input_ids.shape[1]:],skip_special_tokens=True)
            record={'prompt':prompt,'response':response,'seconds':time.perf_counter()-start,'input_tokens':token.input_ids.shape[1],
                    'output_tokens':len(tokenizer.encode(response,add_special_tokens=False))}
            file.write_text(json.dumps(record,indent=2)+'\n')
        if 'output_tokens' not in record:
            record['output_tokens']=len(tokenizer.encode(record['response'],add_special_tokens=False))
        ep=prob[i].copy()
        try:
            proposal=json.loads(record['response']);root=proposal['root_cause']
            if root is not None:
                if root not in record['prompt']['candidates']:raise ValueError('Noncandidate root')
                ep=np.zeros_like(ep);ep[meta['services'].index(root)]=1.
        except (ValueError,KeyError,TypeError):pass
        evidence.append(ep);latencies.append(record['seconds']);print(args.split,i,record['response'][:120],flush=True)
    evidence=np.asarray(evidence);labels=np.load(data/'features.npz')['labels'][indices]
    if args.split=='validation':
        grid=[{'weight':w,'metrics':ranking_metrics(fuse(prob[indices],evidence,w),labels)} for w in [0.,.25,.5,.75,1.]]
        selected=max(grid,key=lambda r:(r['metrics']['PR@1'],r['metrics']['MAP@5'],-r['weight']))
        report={'grid':grid,'selected_weight':selected['weight'],'base':ranking_metrics(prob[indices],labels),
                'proposal':ranking_metrics(evidence,labels),'mean_seconds':float(np.mean(latencies))}
    else:
        sel=json.loads((args.output/'selection.json').read_text());final=fuse(prob[indices],evidence,sel['selected_weight'])
        report={'base':ranking_metrics(prob[indices],labels),'corrected':ranking_metrics(final,labels),'selection':sel}
    report['cost']={'cases':len(indices),'mean_seconds':float(np.mean(latencies)),
                    'p95_seconds':float(np.quantile(latencies,.95)),
                    'mean_input_tokens':float(np.mean([json.loads((args.output/f'{args.split}-{i:05d}.json').read_text())['input_tokens'] for i in indices])),
                    'mean_visible_output_tokens':float(np.mean([len(tokenizer.encode(json.loads((args.output/f'{args.split}-{i:05d}.json').read_text())['response'],add_special_tokens=False)) for i in indices])),
                    'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated(0)),
                    'scope':'Qwen3-8B plus TrainTicket LoRA; model load and inference; decoded visible output excludes terminal special tokens'}
    report_path=args.output/('selection.json' if args.split=='validation' else 'evaluation.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/(args.split+'.npz'),indices=indices,proposal=evidence)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
