"""Rebuild raw features and execute a frozen, paper-sized base-model protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'configs/care/aiops2022-paper-base.json')
    args=p.parse_args();config=json.loads(args.config.read_text())
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    data=ROOT/config['data'];manifest=ROOT/config['split_manifest']
    protocol={'config':config,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'split_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    frozen=output/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=protocol:
        raise ValueError('Protocol changed; use a new output directory')
    frozen.write_text(json.dumps(protocol,indent=2)+'\n')
    state={'pid':os.getpid(),'state':'preparing','completed_seeds':[]}
    def save():
        state['updated_at']=time.time()
        tmp=output/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(output/'status.json')
    def run(command,log):
        with (output/log).open('w') as stream:
            subprocess.run([sys.executable,str(ROOT/'scripts/care'/command[0])]+command[1:],
                           cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        meta=data/'metadata.json'
        if not meta.exists():
            run(['prepare_aiops.py','--split-manifest',str(manifest),'--output',str(data)],'prepare.log')
        if json.loads(meta.read_text())['split_manifest_sha256']!=protocol['split_sha256']:
            raise ValueError('Prepared features use a different split')
        results=[]
        for seed in config['model_seeds']:
            state.update(state='training',seed=seed);save()
            folder=output/f'seed-{seed}'
            if not (folder/'evaluation.json').exists():
                run(['train_base.py','--data',str(data),'--output',str(folder),'--seed',str(seed),
                     '--epochs',str(config['epochs']),'--checkpoint-selection',config['checkpoint_selection'],
                     '--lr',str(config['learning_rate']),'--batch-size',str(config['batch_size']),
                     '--contrast-weight',str(config['contrast_weight']),
                     '--metric-patch-size',str(config['metric_patch_size']),
                     '--metric-patch-stride',str(config['metric_patch_stride']),
                     '--device',config['device'],'--threads',str(config['threads'])],f'seed-{seed}.log')
            report=json.loads((folder/'evaluation.json').read_text())
            results.append({'seed':seed,**report['test']});state['completed_seeds'].append(seed);save()
        metrics=[k for k in results[0] if k!='seed']
        summary={'protocol':protocol,'counts':json.loads(meta.read_text())['counts'],
                 'primary':next(r for r in results if r['seed']==config['primary_seed']),
                 'per_seed':results,
                 'mean':{k:float(np.mean([r[k] for r in results])) for k in metrics},
                 'sample_std':{k:float(np.std([r[k] for r in results],ddof=1)) for k in metrics},
                 'paper_base_only':{'PR@1':.56,'PR@3':.76,'PR@5':.82,'MAP@3':.66,'MAP@5':.69},
                 'comparison_note':'Descriptive only: original incident IDs and unspecified settings are unavailable; not a paired reproduction.'}
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        state['state']='complete';save();print(json.dumps(summary,indent=2))
    except BaseException as exc:
        state.update(state='failed',error=str(exc));save();raise


if __name__=='__main__':main()
