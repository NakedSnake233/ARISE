"""Frozen fresh-adapter training followed by matched constrained diagnosis scoring."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    os.chdir(ROOT)
    config_path = Path('configs/care/aiops-paper128-lora-v2.json')
    cfg = json.loads(config_path.read_text()); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'result.json').exists(): raise ValueError('Completed run is immutable')
    adapter = (out/'adapter').resolve()
    if adapter.exists(): raise ValueError('Refuse implicit resume or overwriting an adapter')
    sft = Path(cfg['sft']); manifest = json.loads((sft/'manifest.json').read_text())
    assert sha(config_path) == manifest['config_sha256']
    assert sha(Path('scripts/care/build_aiops_lora_v2.py')) == manifest['source_sha256']
    for name, digest in manifest['files_sha256'].items(): assert sha(sft/name) == digest
    config = yaml.safe_load(Path('configs/care/qwen3_lora.yaml').read_text()); config.pop('eval_dataset', None)
    config.update(model_name_or_path=str(Path(cfg['model']).resolve()), dataset_dir=str(sft.resolve()),
                  output_dir=str(adapter), eval_strategy='no', save_total_limit=1,
                  lora_rank=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'], learning_rate=cfg['learning_rate'],
                  num_train_epochs=cfg['epochs'], seed=cfg['seed'], cutoff_len=cfg['cutoff_len'],
                  per_device_train_batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
                  max_steps=math.ceil(102/cfg['batch_size']/cfg['gradient_accumulation_steps'])*cfg['epochs'])
    (out/'training.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    sources = [config_path, out/'training.yaml', *sft.glob('*.json'),
               *[Path('scripts/care')/n for n in ['build_aiops_lora_v2.py','run_aiops_lora_v2.py','evaluate_aiops_lora_v2.py','build_sft.py','evidence_reranker.py']],
               *[Path(cfg['data'])/n for n in ['incidents.json','metadata.json','features.npz']],
               Path(cfg['evidence'])/'full.npz', Path(cfg['model'])/'download_manifest.json']
    seal = {'created_at':time.time(), 'protocol':cfg, 'sha256':{str(p):sha(p) for p in sources}}
    (out/'protocol.json').write_text(json.dumps(seal, indent=2)+'\n')
    state = {'pid':os.getpid(),'state':'waiting_for_gpu','started_at':time.time(),'adapter':str(adapter)}
    def save(stage):
        state.update(state=stage,updated_at=time.time())
        p=out/'status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out/'status.json')
    save('waiting_for_gpu')
    try:
        gpu_lock=Path('artifacts/care/finetuning-gpu.lock').open('w');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        save('training')
        with (out/'train.log').open('w') as log:
            subprocess.run([str(Path(sys.executable).parent/'llamafactory-cli'),'train',str(out/'training.yaml')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        ts=json.loads((adapter/'trainer_state.json').read_text())
        if ts['global_step']!=config['max_steps'] or not math.isclose(ts['epoch'],cfg['epochs'],abs_tol=1e-6):
            raise ValueError('Did not complete the fixed training budget: '+str(ts['epoch']))
        save('evaluating')
        with (out/'evaluate.log').open('w') as log:
            subprocess.run([sys.executable,'scripts/care/evaluate_aiops_lora_v2.py'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for name,digest in seal['sha256'].items(): assert sha(Path(name))==digest, name
        state['finished_at']=time.time(); save('complete')
    except BaseException as exc:
        state['error']=str(exc);save('failed');raise


if __name__=='__main__':main()
