"""Labels-only pair-holdout scoring with separate smoke and full coverage."""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,dataset_dir,json_dump,sha256
from .tt_pair_data import FOLDER
from .s3v2_infer import SCENES
from .train_evidence import fuse_all
from .scenario_v2 import load_cases,mask_path,OUTPUT as MASK_ROOT
from .metrics import ranking_metrics
from .s3v2_report import table
from .tt_pair_models import CONTROLS

EXPECTED_METHODS={'ARISE','context-only','ET-spec','ET-spec+gate+aug','RootFreqPrior','Random',
                  'ARISE[V-b]','ARISE[V-d]','Eadro','Eadro+MD','MULAN',*CONTROLS}


def collect(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    state=json.loads((FOLDER/'queue_state.json').read_text());blocks={};coverage=[];inputs={}
    for name,row in state['jobs'].items():
        cfg=row['spec']['expected_config'];coverage.append(dict(job=name,status=row['status'],**cfg))
        if cfg['component']=='train_only_preprocessing' or row['status']!='completed':continue
        folder=ROOT/row['output'];meta=json.loads((folder/'run_meta.json').read_text())
        if sha256(folder/'scores.npz')!=meta['scores_sha256']:raise ValueError('score changed')
        for path,digest in meta['mask_sha256'].items():
            if sha256(ROOT/path)!=digest:raise ValueError('mask changed')
        inputs[str((folder/'run_meta.json').relative_to(ROOT))]=sha256(folder/'run_meta.json')
        with np.load(folder/'scores.npz') as z:arrays={k:z[k].copy() for k in z.files}
        if arrays['incident_ids'].tolist()!=row['spec']['test_ids'] or arrays['scenario_keys'].tolist()!=SCENES:raise ValueError('identity or scene mismatch')
        blocks.setdefault((cfg['fold'],cfg['seed'],cfg['smoke']),{})[cfg.get('method') or cfg['component']]=(arrays,name)
    services,cases,_=load_cases('tt');a0s={key:a for _,key,a in cases}
    label_path=dataset_dir('tt')/'incidents.json';labels={r['incident_id']:services.index(r['root_service']) for r in json.loads(label_path.read_text())}
    inputs[str(label_path.relative_to(ROOT))]=sha256(label_path);rows=[];loaded_masks={}
    for (fold,seed,smoke),parts in blocks.items():
        methods={};ctx=parts.get('context')
        if ctx:methods['context-only']=ctx[0]['scores'],ctx[0]['incident_ids'].tolist(),ctx[1]
        for kind,(z,source) in parts.items():
            ids=z['incident_ids'].tolist()
            if kind=='context':continue
            if kind in ('experts','controls','variants'):
                for method,scores in zip(z['method_names'].tolist(),z['scores']):
                    if kind=='variants':
                        if ctx is None:continue
                        if ids!=ctx[0]['incident_ids'].tolist():raise ValueError('fusion incident mismatch')
                        scores=np.stack([fuse_all((.25,c),(.75,e)) for c,e in zip(ctx[0]['scores'],scores)])
                    methods[method]=scores,ids,source
                if kind=='experts' and ctx is not None:
                    if ids!=ctx[0]['incident_ids'].tolist():raise ValueError('fusion incident mismatch')
                    methods['ARISE']=(np.stack([fuse_all((.25,c),(.75,e)) for c,e in zip(ctx[0]['scores'],z['evidence'])]),ids,source)
            else:methods[kind]=z['scores'],ids,source
        for method,(scores,ids,source) in methods.items():
            for j,scene in enumerate(SCENES):
                if (seed,j) not in loaded_masks and j:
                    path=mask_path(MASK_ROOT,'tt',float(scene.rsplit('_',1)[1]),seed)
                    with np.load(path) as z:loaded_masks[seed,j]={k:z[k].copy() for k in z.files}
                ranked=np.argsort(-scores[j],axis=-1,kind='stable')
                for i,key in enumerate(ids):
                    a0=a0s[key];mask=np.ones_like(a0) if j==0 else loaded_masks[seed,j][key];a=a0*mask;target=labels[key]
                    rows.append(dict(dataset='tt',method=method,fold=fold,seed=seed,smoke=smoke,incident_id=key,
                        scenario='S0' if j==0 else 'S3v2_all',scenario_param='none' if j==0 else scene.rsplit('_',1)[1],
                        rank_of_root=int(np.flatnonzero(ranked[i]==target)[0]+1),
                        stratum='root_originally_unobserved' if not a0[target].any() else 'root_newly_missing' if not a[target].any() else 'root_still_observed',
                        deleted=int((a0*(1-mask)).sum()),original_available=int(a0.sum()),
                        deleted_services=int((a0.any(1)&~a.any(1)).sum()),eligible_services=int(a0.any(1).sum()),source_job=source))
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('no finished model artifacts')
    if frame.duplicated(['method','fold','seed','smoke','incident_id','scenario_param']).any():raise ValueError('duplicate prediction key')
    frame.to_parquet(output/'predictions.parquet',index=False);summaries=[]
    for (method,smoke,param),g in frame.groupby(['method','smoke','scenario_param']):
        full=not smoke and len(g)==750 and g.incident_id.nunique()==150 and set(g.seed)==set(range(42,47)) and set(g.fold)==set(range(5))
        for stratum in ('all','root_newly_missing','root_still_observed','root_originally_unobserved'):
            p=g if stratum=='all' else g[g.stratum==stratum]
            metric=ranking_metrics(p.rank_of_root.to_numpy()) if len(p) else {k:None for k in ('PR@1','PR@3','PR@5','MAP@3','MAP@5','MRR')}
            summaries.append(dict(method=method,smoke=smoke,scenario_param=param,stratum=stratum,
                coverage='full_5fold_5seed' if full else 'smoke' if smoke else 'partial',event_seed_rows=len(p),events=p.incident_id.nunique(),
                actual_cell_deletion_rate=p.deleted.sum()/max(1,p.original_available.sum()),
                actual_service_deletion_fraction=p.deleted_services.sum()/max(1,p.eligible_services.sum()),**metric))
    summary=pd.DataFrame(summaries);summary.to_csv(output/'metrics.csv',index=False);pd.DataFrame(coverage).to_csv(output/'job_coverage.csv',index=False)
    valid=summary[(summary.stratum=='all')&~summary.smoke].copy();valid['Top1 (%)']=valid['PR@1']*100
    display=valid.pivot(index=['method','coverage'],columns='scenario_param',values='Top1 (%)').reindex(columns=['none','0.10','0.25','0.50']).round(2).reset_index()
    lines=['# TT 组合分组验证','',f'任务状态：{state["counts"]}；只有full_5fold_5seed行达到完整范围。','',table(display),'',
        '以上均为重新拟合新训练折的模型；V-b/V-d与ARISE使用同一新context，融合为0.25 context+0.75 evidence。标签置乱方法是负对照，不是待优化的基线。', '',
        '同一服务×故障类型的三次重复同时留出，仍然是已查看150事件上的事后敏感性分析，不能当成新的独立数据集。实际删除率及根因观测分层见metrics.csv。尚无跨强度/负载的新采集结果。','',
        '烟测和完整测试分开，未完成组件不会进入完整覆盖结论；任务失败和未完成状态保留在job_coverage.csv。本轮没有因测试成绩选择模型、超参数或种子。','']
    (output/'REPORT.md').write_text('\n'.join(lines));json_dump(output/'receipt.json',dict(status='pair_holdout_evaluation',queue_counts=state['counts'],rows=len(frame),input_receipts=inputs,source_sha256=sha256(Path(__file__))))
    if state['stage']=='full' and state['status']=='completed':
        if set(valid.method)!=EXPECTED_METHODS or len(valid)!=len(EXPECTED_METHODS)*4 or set(valid.coverage)!={'full_5fold_5seed'}:
            raise ValueError('terminal queue has incomplete method/scene coverage; no final receipt published')
        json_dump(FOLDER/'final_evaluation.json',dict(report=str((output/'REPORT.md').relative_to(ROOT)),rows=len(frame)))
    return str(output/'REPORT.md')


if __name__=='__main__':
    from .disk_guard_fast import DiskGuard
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    with (FOLDER/'.collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (FOLDER/'final_evaluation.json').exists() and a.output is None:print((FOLDER/'final_evaluation.json').read_text())
        else:
            output=a.output or ROOT/'results_v2'/('TT_pair_evaluation_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
            g=DiskGuard(ROOT);g.reserve('tt_pair_collection',output,ROOT/'logs/tt_pair_collection.log')
            try:print(collect(output))
            finally:g.release('tt_pair_collection')
