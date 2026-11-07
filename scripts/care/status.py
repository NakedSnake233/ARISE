"""Report completed artifacts rather than treating schedules as collected data."""
import collections
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    status={}
    repair=ROOT/'artifacts/care/trainticket-log-repair.json'
    if repair.exists():
        record=json.loads(repair.read_text())
        status['trainticket_log_repair']={k:record[k] for k in ['state','reason','archived_at','raw_collection_preserved']}
    manifest=ROOT/'models/Qwen3-8B/download_manifest.json'
    status['qwen_download_verified']=manifest.exists()
    check=ROOT/'artifacts/care/qwen-load-check.json'
    status['qwen_gpu_check']=json.loads(check.read_text()) if check.exists() else None
    source=ROOT/'data/care/aiops2022-original'
    verified=list(source.rglob('*.csv.verified.json'))
    status['aiops_verified_raw_modal_files']=sum('trace_jaeger-span' in p.name or 'log_filebeat-testbed-log-service' in p.name for p in verified)
    status['aiops_expected_raw_modal_files']=14
    for dataset in ['trainticket-collected','trainticket-pilot']:
        folder=ROOT/'data/care'/dataset;records=[json.loads(p.read_text()) for p in folder.glob('*/incident.json')]
        complete=[r for r in records if r.get('status')=='complete']
        schedule=folder/'schedule.json'
        status[dataset]={'planned':len(json.loads(schedule.read_text())) if schedule.exists() else 0,
                         'states':dict(collections.Counter(r['status'] for r in records)),
                         'complete_by_fault':dict(collections.Counter(r['fault'] for r in complete)),
                         'traces_in_complete_incidents':sum(r.get('trace_count',0) for r in complete)}
        campaign=folder/'campaign-status.json'
        if campaign.exists():status[dataset]['campaign']=json.loads(campaign.read_text())
    for dataset in ['aiops2022','trainticket']:
        for stage in ['pipeline','finetuning']:
            p=ROOT/'artifacts/care'/(stage+'-'+dataset)/'status.json'
            state=json.loads(p.read_text()) if p.exists() else {'state':'not_started'}
            # Host jobs can be invisible in the sandbox's PID namespace. Do not
            # mistake a missing /proc entry here for an exited training process.
            if state.get('updated_at'):
                state['status_age_seconds']=round(time.time()-state['updated_at'])
            if stage=='finetuning':
                adapter='qwen3-care-lora' if dataset=='aiops2022' else 'qwen3-care-lora-trainticket'
                log=ROOT/'artifacts/care'/adapter/'trainer_log.jsonl'
                if log.exists():
                    lines=log.read_text().splitlines()
                    if lines:
                        try:state['training_progress']=json.loads(lines[-1])
                        except json.JSONDecodeError:pass
                    age=round(time.time()-log.stat().st_mtime)
                    state['trainer_log_age_seconds']=age
                    if state.get('state')=='training' and age>600:
                        state['warning']='No trainer log update for over 10 minutes; verify host process liveness. The saved stage alone does not prove it is running.'
            status[stage+'_'+dataset]=state
        p=ROOT/'data/care'/(dataset+'-prepared')/'metadata.json'
        if p.exists():status['prepared_'+dataset]=json.loads(p.read_text())['counts']
        p=ROOT/'artifacts/care'/('correction-'+dataset)/'evaluation.json'
        if p.exists():
            evaluation=json.loads(p.read_text())
            status['care_evaluation_'+dataset]={k:evaluation[k] for k in ['incidents','base','care','changed_predictions','invalid_llm_abstentions']}
    paper=ROOT/'artifacts/care/base-aiops2022-paper128'
    if (paper/'status.json').exists():
        status['paper128_base_aiops2022']=json.loads((paper/'status.json').read_text())
        if (paper/'summary.json').exists():
            summary=json.loads((paper/'summary.json').read_text())
            status['paper128_base_aiops2022'].update(counts=summary['counts'],primary=summary['primary'],
                                                   five_seed_mean=summary['mean'])
    if 'trainticket_log_repair' in status:
        pipeline=status.get('pipeline_trainticket',{}).get('state','not_started')
        finetuning=status.get('finetuning_trainticket',{}).get('state','not_started')
        status['trainticket_log_repair']['state']=finetuning if pipeline=='complete' else pipeline
    full=ROOT/'artifacts/care/full-v3'
    if (full/'status.json').exists():
        run=json.loads((full/'status.json').read_text())
        run['completed_count']=len(run.pop('completed',[]))
        run['verified_result_files']=len(list(full.glob('*/*/seed-*/correction/result.json')))
        run['status_age_seconds']=round(time.time()-run['updated_at'])
        run['report_available']=(full/'REPORT.md').exists()
        if (full/'verification.json').exists():run['deterministic_verification']=json.loads((full/'verification.json').read_text())['status']
        if (full/'EXPERIMENT_AUDIT.json').exists():
            audit=json.loads((full/'EXPERIMENT_AUDIT.json').read_text())
            run['integrity_audit']={'verdict':audit['verdict'],'acceptance_status':audit['acceptance_status'],'summary':audit['summary']}
        run['data_model_seal_available']=(full/'data_model_seal.json').exists()
        if (full/'summary.json').exists():
            summary=json.loads((full/'summary.json').read_text())
            run['full_model_five_seed_results']={d:{k:summary[d]['methods']['full'][k] for k in ['base','corrected']}
                                               for d in ['trainticket','aiops2022']}
        status['rpc_correction_full_experiment']=run
    lora=ROOT/'artifacts/care/lora-aiops-paper128-v2'
    if (lora/'status.json').exists():
        run=json.loads((lora/'status.json').read_text())
        log=lora/'adapter/trainer_log.jsonl'
        if log.exists():
            lines=log.read_text().splitlines()
            if lines:
                try:run['training_progress']=json.loads(lines[-1])
                except json.JSONDecodeError:pass
            run['trainer_log_age_seconds']=round(time.time()-log.stat().st_mtime)
        if (lora/'result.json').exists():
            result=json.loads((lora/'result.json').read_text())
            run['base']=result['base'];run['evidence_only']=result['evidence_only']
            run['primary_order']=result['orders'][str(result['protocol']['primary_order_seed'])]
        if (lora/'verification.json').exists():run['deterministic_verification']=json.loads((lora/'verification.json').read_text())['status']
        if (lora/'EXPERIMENT_AUDIT.json').exists():
            audit=json.loads((lora/'EXPERIMENT_AUDIT.json').read_text())
            run['integrity_audit']={k:audit.get(k) for k in ['verdict','acceptance_status','summary']}
        status['aiops_paper128_lora_v2']=run
    print(json.dumps(status,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
