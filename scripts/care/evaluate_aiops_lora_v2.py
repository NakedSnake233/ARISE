"""Matched letter likelihood evaluation; labels are opened only after inference."""
import json
from pathlib import Path
import time
import numpy as np
from evidence_reranker import fuse, paired_statistics
from train_base import ranking_metrics


def metrics(prob, labels):
    ranks=np.argsort(-prob,axis=1,kind='stable')
    pos=(ranks==labels[:,None]).argmax(1)+1
    return {'n':len(labels),'correct':int((pos==1).sum()),**ranking_metrics(prob,labels)}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    cfg=json.loads(Path('configs/care/aiops-paper128-lora-v2.json').read_text());out=Path(cfg['output'])
    if (out/'result.json').exists():raise ValueError('Refuse overwrite of scored experiment')
    inputs=json.loads((Path(cfg['sft'])/'test_inputs.json').read_text())
    services=json.loads((Path(cfg['data'])/'metadata.json').read_text())['services']
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True)
    model.eval();torch.set_num_threads(4)
    ids=[tokenizer.encode(c,add_special_tokens=False)[0] for c in 'ABCDEFGHIJ']
    assert all(len(tokenizer.encode(c,add_special_tokens=False))==1 for c in 'ABCDEFGHIJ')
    scores={};records={}
    for mode in ['unadapted','lora']:
        if mode=='lora':model=PeftModel.from_pretrained(model,str(out/'adapter'),is_trainable=False);model.eval()
        pred=[];details=[]
        with torch.inference_mode():
            for j,row in enumerate(inputs):
                text=row['instruction']+'\n'+row['input']
                encoded=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to('cuda')
                assert encoded.shape[1]<=cfg['max_input_tokens']
                start=time.monotonic()
                logits=model(input_ids=encoded,attention_mask=torch.ones_like(encoded),use_cache=False,logits_to_keep=1).logits[0,-1].float()
                letter_logits=logits[ids];p=letter_logits.softmax(-1).cpu().numpy();canonical=np.zeros(len(services))
                for k,letter in enumerate('ABCDEFGHIJ'):canonical[services.index(row['mapping'][letter])]=p[k]
                pred.append(canonical)
                details.append({'index':row['index'],'order_seed':row['order_seed'],'input_tokens':encoded.shape[1],
                                'letter_logits':letter_logits.cpu().tolist(),'valid_letter_mass':float((torch.logsumexp(letter_logits,0)-torch.logsumexp(logits,0)).exp()),
                                'unconstrained_first_token_id':int(logits.argmax()),'seconds':time.monotonic()-start})
                if (j+1)%10==0:print(mode,j+1,'/',len(inputs),flush=True)
        scores[mode]=np.asarray(pred);records[mode]=details
        np.savez_compressed(out/(mode+'-predictions.npz'),indices=[r['index'] for r in inputs],order_seeds=[r['order_seed'] for r in inputs],probabilities=scores[mode])
        (out/(mode+'-inference.json')).write_text(json.dumps(details,indent=2)+'\n')
    # Scoring labels cannot affect model inference above.
    rows=json.loads((Path(cfg['data'])/'incidents.json').read_text())
    y=np.array([services.index(r['root_service']) for r in rows]);te=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    base=np.load(Path(cfg['base'])/'predictions.npz')['probabilities'][te]
    old=np.load(Path(cfg['reference_correction'])/'predictions.npz');assert np.array_equal(old['indices'],te)
    tree=old['evidence'];result={'protocol':cfg,'base':metrics(base,y[te]),'evidence_only':metrics(tree,y[te]),'orders':{}}
    all_predictions={}
    for seed in cfg['inference_order_seeds']:
        ix=[j for j,r in enumerate(inputs) if r['order_seed']==seed];assert np.array_equal([inputs[j]['index'] for j in ix],te)
        zero=scores['unadapted'][ix];lora=scores['lora'][ix]
        pred={'unadapted':zero,'lora':lora,'base_plus_lora':fuse(base,lora,cfg['base_fusion_weight'],cfg['base_fusion_top_k']),
              'evidence_plus_lora':fuse(tree,lora,cfg['evidence_fusion_weight'],len(services))}
        r={name:metrics(p,y[te]) for name,p in pred.items()}
        r['paired']={'lora_vs_unadapted':paired_statistics(zero,lora,y[te]),'lora_vs_base':paired_statistics(base,lora,y[te]),
                     'lora_vs_evidence':paired_statistics(tree,lora,y[te]),'base_fusion_vs_base':paired_statistics(base,pred['base_plus_lora'],y[te]),
                     'evidence_fusion_vs_evidence':paired_statistics(tree,pred['evidence_plus_lora'],y[te])}
        r['valid_letter_mass']={mode:float(np.mean([records[mode][j]['valid_letter_mass'] for j in ix])) for mode in scores}
        result['orders'][str(seed)]=r
        for name,p in pred.items():all_predictions[f'{name}_{seed}']=p
    np.savez_compressed(out/'comparison.npz',indices=te,labels=y[te],base=base,evidence=tree,**all_predictions)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
