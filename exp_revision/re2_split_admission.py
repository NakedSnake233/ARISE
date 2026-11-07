"""Admit affordable model resources independently with one immutable blueprint.

This is an execution partition of the existing 600-job protocol. A GPU subset
is not a complete model comparison; the full collection index is published
only after both resource subsets have passed their own compute admission.
"""
from __future__ import annotations
import json

from .common import ROOT,json_dump,sha256
from . import remaining_queue as q
from . import re2_data as data
from . import t8_admission as admission
from . import t8_manifest as original

BLUEPRINT=ROOT/'results_v2/T8_inputs/re2_model_execution_blueprint.json'
SUBSETS={'gpu':'62_full_re2_gpu_models.json','cpu':'63_full_re2_cpu_models.json'}
GPU_COMPONENTS={'context','Eadro','MULAN','Eadro+MD','MULAN+MD'}


def jobs(source_commit):
    """Keep every original full-model scientific and runtime job field."""
    result=[];catalog=json.loads(data.CATALOG.read_text())
    for ds in data.DATASETS:
        parts=json.loads((ROOT/f'splits/{ds}_kfold5.json').read_text())
        services=len(catalog[ds]['services'])
        for fold in range(5):
            for seed in range(42,47):
                for component in original.COMPONENTS:
                    ids=parts[f'fold_{fold}']['test'];shape=[12,len(ids),services]
                    if component=='evidence':shape=[5,*shape]
                    result.append(dict(id=f'full_models_{ds}_{component.replace("+","plus")}_f{fold}_s{seed}',
                        resource='gpu' if component in GPU_COMPONENTS else 'cpu',cores=4 if component=='evidence' else 1,
                        max_attempts=2,timeout_seconds=1800 if component=='evidence' else 900,source_commit=source_commit,
                        expected_config=dict(task='T8',dataset=ds,fold=fold,smoke=False,component=component,seed=seed),
                        command=['-m','exp_revision.re2_models','--dataset',ds,'--fold',str(fold),'--seed',str(seed),'--component',component],
                        test_ids=ids,score_shape=shape,expected_status=['completed_re2_scores']))
    return result


def blueprint():
    if not all(original.complete(name) for name in ('45_full_re2_fold.json','46_smoke_re2_models.json')):return None
    if not BLUEPRINT.exists():
        projection=original.forecast('models');source_commit=admission.commit()
        sources=original.closure('models',data.DATASETS)
        sources['exp_revision/re2_split_admission.py']=sha256(ROOT/'exp_revision/re2_split_admission.py')
        json_dump(BLUEPRINT,dict(status='planned_unadmitted_model_matrix',source_commit=source_commit,
            source_sha256=sources,jobs=jobs(source_commit),forecast=projection,
            complete_comparison_requires_both_resources=True))
    plan=json.loads(BLUEPRINT.read_text())
    if plan['status']!='planned_unadmitted_model_matrix' or plan['jobs']!=jobs(plan['source_commit']):
        raise ValueError('immutable model blueprint differs from the original protocol')
    for relative,digest in plan['source_sha256'].items():
        if sha256(ROOT/relative)!=digest:raise ValueError('model blueprint input changed: '+relative)
    return plan


def create_resource(resource):
    if resource not in SUBSETS:raise ValueError('unknown model resource')
    # Never duplicate an already admitted original all-resource matrix.
    if (q.MANIFESTS/'47_full_re2_models.json').exists():return None
    path=q.MANIFESTS/SUBSETS[resource]
    if path.exists():return None
    plan=blueprint()
    if plan is None:return None
    details=[r for r in plan['forecast']['details'] if r['resource']==resource]
    projection=dict(kind='models',forecast_hours={r:plan['forecast']['forecast_hours'][r] if r==resource else 0. for r in ('cpu','gpu')},
        details=details,basis=plan['forecast']['basis'],execution_partition=resource,
        complete_comparison=False)
    admission.admit_forecast(projection,'models_'+resource)
    selected=[j for j in plan['jobs'] if j['resource']==resource]
    sources={**plan['source_sha256'],str(BLUEPRINT.relative_to(ROOT)):sha256(BLUEPRINT)}
    return admission.write_manifest(path,'full_re2_'+resource+'_models',selected,sources,projection)


def create_index():
    path=q.MANIFESTS/'47_full_re2_models.json'
    if path.exists() or not all((q.MANIFESTS/n).exists() for n in SUBSETS.values()):return None
    plan=blueprint()
    if plan is None:return None
    combined=[];sources={};projection=plan['forecast']
    for resource,name in SUBSETS.items():
        source=q.MANIFESTS/name;manifest=json.loads(source.read_text())
        expected=[j for j in plan['jobs'] if j['resource']==resource]
        if manifest['jobs']!=expected:raise ValueError('model subset differs from immutable blueprint')
        combined+=manifest['jobs']
        for relative,digest in manifest['source_sha256'].items():
            if relative in sources and sources[relative]!=digest:raise ValueError('inconsistent subset source pins')
            sources[relative]=digest
        sources[str(source.relative_to(ROOT))]=sha256(source)
    if len({j['id'] for j in combined})!=len(plan['jobs']):raise ValueError('duplicate or missing split model jobs')
    # A pure index reuses the exact admitted specifications. It adds no jobs or
    # compute reservation; pending_forecasts deduplicates shared IDs.
    return admission.write_manifest(path,'full_models',plan['jobs'],sources,projection)
