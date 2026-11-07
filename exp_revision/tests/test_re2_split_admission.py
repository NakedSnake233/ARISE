import json
from pathlib import Path

import pytest

from exp_revision import re2_split_admission as split
from exp_revision import t8_manifest as original
from exp_revision import t8_admission as admission
from exp_revision import remaining_queue as q


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(split,'ROOT',tmp_path)
    monkeypatch.setattr(original,'ROOT',tmp_path)
    monkeypatch.setattr(q,'MANIFESTS',tmp_path/'manifests/remaining')
    q.MANIFESTS.mkdir(parents=True)
    monkeypatch.setattr(split,'BLUEPRINT',tmp_path/'results_v2/T8_inputs/blueprint.json')
    catalog=tmp_path/'catalog.json'
    monkeypatch.setattr(split.data,'CATALOG',catalog)
    catalog.write_text(json.dumps({ds:{'services':[f's{i}' for i in range(n)]} for ds,n in zip(split.data.DATASETS,(11,15,68))}))
    (tmp_path/'splits').mkdir()
    for ds in split.data.DATASETS:
        (tmp_path/f'splits/{ds}_kfold5.json').write_text(json.dumps({f'fold_{f}':{'test':[f'{ds}:event_{f:03d}']} for f in range(5)}))
    source=tmp_path/'exp_revision/re2_split_admission.py';source.parent.mkdir()
    source.write_bytes(Path(split.__file__).read_bytes())
    (tmp_path/'exp_revision/t8_admission.py').write_bytes(Path(admission.__file__).read_bytes())
    monkeypatch.setattr(original,'complete',lambda *args:True)
    monkeypatch.setattr(original,'closure',lambda *args:{})
    monkeypatch.setattr(admission,'commit',lambda:'first-frozen-commit')
    details=[dict(dataset=ds,component=c,resource='gpu' if c in split.GPU_COMPONENTS else 'cpu',
                  smoke_seconds=1.,forecast_hours=25/3600) for ds in split.data.DATASETS for c in original.COMPONENTS]
    projection=dict(kind='models',forecast_hours={'cpu':10.,'gpu':2.},details=details,basis='fixture observed smoke')
    monkeypatch.setattr(original,'forecast',lambda *args:json.loads(json.dumps(projection)))
    monkeypatch.setattr(admission,'state',lambda:{'jobs':{}})
    monkeypatch.setattr(admission,'pending_forecasts',lambda *args:{'cpu':0.,'gpu':0.})
    monkeypatch.setattr(admission,'charged',lambda *args:195. if args[1]=='cpu' else 40.)
    monkeypatch.setattr(q,'CPU_BUDGET',200.)
    monkeypatch.setattr(q,'GPU_BUDGET',48.)
    # Budget forecast receipts also stay inside the isolated root.
    monkeypatch.setattr(admission,'ROOT',tmp_path)
    return tmp_path


def test_exact_original_specs_and_disjoint_resource_partition(workspace,monkeypatch):
    captured={}
    monkeypatch.setattr(original,'dependencies',lambda *args:None)
    monkeypatch.setattr(original,'state',lambda:{'jobs':{}})
    monkeypatch.setattr(original,'outstanding_forecasts',lambda *args:{'cpu':0.,'gpu':0.})
    monkeypatch.setattr(q,'charged',lambda *args:0.)
    monkeypatch.setattr(original.subprocess,'check_output',lambda *args,**kwargs:'first-frozen-commit\n')
    monkeypatch.setattr(original,'json_dump',lambda path,value:captured.update(value))
    original.create('models',False)
    jobs=split.jobs('first-frozen-commit')
    assert jobs==captured['jobs']
    gpu=[j for j in jobs if j['resource']=='gpu'];cpu=[j for j in jobs if j['resource']=='cpu']
    assert len(gpu)==375 and len(cpu)==225
    assert not ({j['id'] for j in gpu}&{j['id'] for j in cpu})
    assert len({j['id'] for j in gpu+cpu})==600


def test_gpu_only_under_original_cap_does_not_publish_cpu_or_full(workspace):
    gpu=split.create_resource('gpu')
    assert gpu.exists()
    assert len(json.loads(gpu.read_text())['jobs'])==375
    with pytest.raises(ValueError,match='cpu budget'):
        split.create_resource('cpu')
    assert not (q.MANIFESTS/split.SUBSETS['cpu']).exists()
    assert split.create_index() is None
    assert not (q.MANIFESTS/'47_full_re2_models.json').exists()


def test_index_reuses_frozen_specs_across_commit_and_has_no_extra_jobs(workspace,monkeypatch):
    gpu=split.create_resource('gpu');old_bytes=gpu.read_bytes();old=json.loads(old_bytes)
    monkeypatch.setattr(admission,'commit',lambda:'later-doc-only-commit')
    monkeypatch.setattr(admission,'charged',lambda *args:100. if args[1]=='cpu' else 40.)
    cpu=split.create_resource('cpu');index=split.create_index()
    value=json.loads(index.read_text());cpu_value=json.loads(cpu.read_text())
    assert len(value['jobs'])==600
    assert {j['source_commit'] for j in value['jobs']}=={'first-frozen-commit'}
    assert {j['id']:j for j in value['jobs']}=={j['id']:j for j in old['jobs']+cpu_value['jobs']}
    assert split.create_resource('gpu') is None and split.create_resource('cpu') is None and split.create_index() is None
    assert gpu.read_bytes()==old_bytes


def test_blueprint_changed_input_and_subset_are_rejected(workspace,monkeypatch):
    split.create_resource('gpu')
    source=workspace/'exp_revision/re2_split_admission.py';source.write_text('changed fixture input')
    with pytest.raises(ValueError,match='blueprint input changed'):split.blueprint()
    source.write_bytes(Path(split.__file__).read_bytes())
    monkeypatch.setattr(admission,'charged',lambda *args:100. if args[1]=='cpu' else 40.)
    split.create_resource('cpu')
    gpu=q.MANIFESTS/split.SUBSETS['gpu'];value=json.loads(gpu.read_text());value['jobs'][0]['expected_config']['seed']=999
    gpu.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='subset differs'):split.create_index()


def test_missing_verified_prerequisite_and_existing_full_matrix_are_idempotent(workspace,monkeypatch):
    monkeypatch.setattr(original,'complete',lambda *args:False)
    assert split.create_resource('gpu') is None and not split.BLUEPRINT.exists()
    (q.MANIFESTS/'47_full_re2_models.json').write_text('{}')
    assert split.create_resource('gpu') is None
    assert split.create_resource('cpu') is None
