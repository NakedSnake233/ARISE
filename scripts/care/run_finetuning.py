"""Wait for real SFT data, run three-epoch LoRA, then evaluate held-out incidents.

Each dataset has its own adapter. A project GPU lock serializes these jobs.
No test answers are consumed by the trainer. Failures remain visible in status.
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['aiops2022', 'trainticket'], required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    folder = ROOT/'artifacts/care'/('finetuning-'+args.dataset)
    folder.mkdir(parents=True, exist_ok=True)
    # Prevent two supervisors from overwriting the same run or status file.
    run_lock = (folder/'run.lock').open('w')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = ROOT/'artifacts/care'/('correction-'+args.dataset)/'evaluation.json'
    status_path = folder/'status.json'
    old = json.loads(status_path.read_text()) if status_path.exists() else {}
    if old.get('state') == 'complete' and report.exists():
        print('Already complete:', report)
        return
    state = {'dataset': args.dataset, 'pid': os.getpid(), 'state': 'waiting_for_sft',
             'started_at': time.time()}

    def save():
        state['updated_at'] = time.time()
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(status_path)

    save()
    try:
        upstream = ROOT/'artifacts/care'/('pipeline-'+args.dataset)/'status.json'
        while True:
            current = json.loads(upstream.read_text()) if upstream.exists() else {}
            if current.get('state') == 'complete':
                break
            if current.get('state') == 'failed':
                raise RuntimeError('Data pipeline failed: '+str(current.get('error')))
            if not args.wait:
                raise RuntimeError('Real-data SFT pipeline is incomplete; use --wait')
            time.sleep(30)
            save()
        dataset = ROOT/'data/care'/('sft-'+args.dataset)
        checksums = {}
        counts = {}
        for name in ['train', 'validation', 'test_inputs']:
            path = dataset/(name+'.json')
            content = path.read_bytes()
            checksums[name] = hashlib.sha256(content).hexdigest()
            counts[name] = len(json.loads(content))
        if not all(counts.values()):
            raise ValueError('All data splits must be nonempty')
        state.update({'sample_counts': counts, 'input_sha256': checksums, 'state': 'waiting_for_gpu'})
        save()
        gpu_lock = (ROOT/'artifacts/care/finetuning-gpu.lock').open('w')
        while True:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(30)
                save()
        adapter = ROOT/'artifacts/care'/('qwen3-care-lora' if args.dataset == 'aiops2022'
                                          else 'qwen3-care-lora-trainticket')
        config = yaml.safe_load((ROOT/'configs/care/qwen3_lora.yaml').read_text())
        config.update({'dataset_dir': str(dataset), 'output_dir': str(adapter),
                       'save_total_limit': 1})
        # Transformers 4.51 computes its default step budget with floor division
        # but steps once more for the final partial accumulation group. Set the
        # exact single-device budget so three epochs do not stop at ~2.8 epochs.
        if int(os.environ.get('WORLD_SIZE','1'))!=1:
            raise ValueError('This reproduction supervisor requires one training device')
        batches=math.ceil(counts['train']/config['per_device_train_batch_size'])
        config['max_steps']=math.ceil(batches/config['gradient_accumulation_steps'])*int(config['num_train_epochs'])
        # Resume only this dataset's epoch checkpoint after an interrupted run.
        checkpoints = sorted(adapter.glob('checkpoint-*'),
                             key=lambda p: int(p.name.split('-')[-1]))
        if checkpoints:
            config['resume_from_checkpoint'] = str(checkpoints[-1])
            trainer_state=json.loads((checkpoints[-1]/'trainer_state.json').read_text())
            epoch=float(trainer_state.get('epoch',0.))
            # Epoch checkpoints already consumed the final partial accumulation
            # group. Trainer's floor-based modulo would wrongly skip batches in
            # the NEXT epoch; restart that epoch at its first batch instead.
            if math.isclose(epoch,round(epoch),abs_tol=1e-8):
                config['ignore_data_skip']=True
        configuration = folder/'training.yaml'
        configuration.write_text(yaml.safe_dump(config, sort_keys=False))
        environment = dict(os.environ, TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
        if not ((adapter/'train_results.json').exists() and (adapter/'adapter_config.json').exists()):
            state['state'] = 'training'; save()
            with (folder/'train.log').open('a') as log:
                subprocess.run([str(Path(sys.executable).parent/'llamafactory-cli'), 'train', str(configuration)],
                               cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['state'] = 'evaluating'; save()
        command = [sys.executable, str(ROOT/'scripts/care/evaluate_correction.py'),
                   '--data', str(ROOT/'data/care'/(args.dataset+'-prepared')),
                   '--base', str(ROOT/'artifacts/care'/('base-'+args.dataset)),
                   '--sft', str(dataset), '--adapter', str(adapter), '--output', str(report.parent)]
        with (folder/'evaluate.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        state.update({'state': 'complete', 'evaluation': str(report), 'adapter': str(adapter),
                      'finished_at': time.time()})
        save()
    except BaseException as exc:
        state.update({'state': 'failed', 'error': str(exc)}); save()
        raise


if __name__ == '__main__':
    main()
