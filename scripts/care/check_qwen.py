"""Verify local model loading and short generation on the available GPU."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='models/Qwen3-8B')
    p.add_argument('--output',type=Path,default=Path('artifacts/care/qwen-load-check.json'))
    args=p.parse_args();start=time.time()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                                             device_map='cuda:0',local_files_only=True)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':'Return only JSON: {"status":"ready"}'}],
                                        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    x=tokenizer(prompt,return_tensors='pt').to('cuda:0')
    with torch.inference_mode():out=model.generate(**x,max_new_tokens=32,do_sample=False)
    result={'model':'Qwen/Qwen3-8B','gpu':torch.cuda.get_device_name(),'seconds':time.time()-start,
            'response':tokenizer.decode(out[0][x.input_ids.shape[1]:],skip_special_tokens=True),
            'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
