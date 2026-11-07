import ast
import inspect
import os
from pathlib import Path
import textwrap

import pytest

from exp_revision import disk_guard as base
from exp_revision import disk_guard_fast as fast
from exp_revision.disk_guard_local import DiskGuard as LocalGuard


@pytest.fixture
def tree(tmp_path):
    for name in ('results_v2/job','cache/job_tmp/job','logs'):
        (tmp_path/name).mkdir(parents=True)
    for i in range(30):
        folder=tmp_path/'cache'/f'd{i%5}'
        folder.mkdir(exist_ok=True)
        (folder/f'f{i}').write_bytes(bytes([i])*((i+1)*127))
    (tmp_path/'results_v2/job/data').write_bytes(b'abc')
    with (tmp_path/'cache/sparse').open('wb') as stream:
        stream.seek(8*1024**2);stream.write(b'x')
    os.link(tmp_path/'results_v2/job/data',tmp_path/'logs/hardlink')
    (tmp_path/'cache/file_link').symlink_to(tmp_path/'results_v2/job/data')
    (tmp_path/'cache/dir_link').symlink_to(tmp_path/'results_v2',target_is_directory=True)
    (tmp_path/'cache/broken_link').symlink_to(tmp_path/'absent')
    os.mkfifo(tmp_path/'cache/fifo')
    return tmp_path


def test_scanner_matches_original_for_sparse_hardlinks_symlinks_and_special_files(tree):
    for path in [tree,*tree.rglob('*'),tree/'absent']:
        assert fast.usage_bytes(path)==base.usage_bytes(path),path


def test_snapshot_decision_algorithm_is_unchanged():
    original=ast.dump(ast.parse(textwrap.dedent(inspect.getsource(base.DiskGuard._snapshot))))
    accelerated=ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fast.DiskGuard._snapshot))))
    assert original==accelerated


@pytest.mark.parametrize('additional',[0,1,40])
def test_full_snapshot_matches_original_including_reservation_refusal(tree,monkeypatch,additional):
    original=LocalGuard(tree);accelerated=fast.DiskGuard(tree)
    fs=os.statvfs(tree)
    monkeypatch.setattr(base.os,'statvfs',lambda path:fs)
    state=dict(leases={'job':dict(paths=[str(tree/'results_v2/job'),str(tree/'logs'),str(tree/'cache/job_tmp/job')])})
    a=original._snapshot(state,additional);b=accelerated._snapshot(state,additional)
    a.pop('time');b.pop('time');assert a==b
    if additional==40:assert b['safe_to_launch'] is False


def test_fast_guard_keeps_namespace_and_live_lease_protection(tree):
    guard=fast.DiskGuard(tree)
    env=guard.reserve('job',tree/'results_v2/job',tree/'logs/job.log')
    guard.attach('job',os.getpid())
    with pytest.raises(RuntimeError,match='running writer'):guard.release('job')
    assert Path(env['TMPDIR']).is_relative_to(tree)
    assert fast.guarded_command(tree,'job',['python','-V'])[2]=='exp_revision.disk_guard_fast'


def test_snapshot_retains_job_and_aggregate_abort_signals(tree,monkeypatch):
    guard=fast.DiskGuard(tree)
    output=tree/'results_v2/job/oversize'
    with output.open('wb') as stream:
        stream.seek(base.JOB_BYTES);stream.write(b'x')
    state=dict(leases={'job':dict(paths=[str(output)])})
    snapshot=guard._snapshot(state)
    assert snapshot['abort_job_keys']==['job']
    monkeypatch.setitem(base.POLICY,'max_experiment_bytes',1)
    snapshot=guard._snapshot(state)
    assert snapshot['abort_job_keys']==['job'] and not snapshot['safe_to_launch']


def test_v5_scheduler_preserves_adoption_and_resource_limits(tmp_path,monkeypatch):
    from exp_revision import extended_queue_v5 as queue
    from exp_revision.tests import test_parallel_runtime as existing
    source=tmp_path/'exp_revision/disk_guard_fast.py';source.parent.mkdir()
    source.write_bytes(Path(fast.__file__).read_bytes())
    monkeypatch.setattr(existing,'queue',queue)
    existing.test_scheduler_adopts_children_and_fills_twelve_cpu_slots_without_changing_specs(tmp_path,monkeypatch)
