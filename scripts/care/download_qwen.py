"""Download the official unquantized Qwen3-8B snapshot, resume and verify hashes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('models/Qwen3-8B'))
    p.add_argument('--revision', default='b968826d9c46dd6066d109eabc6255188de91218')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info('Qwen/Qwen3-8B', revision=args.revision, files_metadata=True)
    files = [f for f in info.siblings if not f.rfilename.startswith('.')]
    missing = sum(max(0, (f.size or 0) - ((args.output / f.rfilename).stat().st_size
                  if (args.output / f.rfilename).exists() else 0)) for f in files)
    if shutil.disk_usage(args.output).free < missing + 5 * 1024**3:
        raise RuntimeError('Insufficient space: require missing model bytes plus 5 GiB reserve')
    print(f'Downloading Qwen/Qwen3-8B revision={info.sha}, bytes={sum(f.size or 0 for f in files)}', flush=True)
    snapshot_download('Qwen/Qwen3-8B', revision=info.sha, local_dir=args.output,
                      allow_patterns=[f.rfilename for f in files], max_workers=2)
    manifest = {'repo_id': 'Qwen/Qwen3-8B', 'revision': info.sha, 'files': []}
    for f in files:
        path = args.output / f.rfilename
        if path.stat().st_size != f.size:
            raise RuntimeError(f'Size mismatch: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        sha = digest.hexdigest()
        expected = f.lfs.sha256 if f.lfs else None
        if expected and sha != expected:
            raise RuntimeError(f'SHA256 mismatch: {path}')
        manifest['files'].append({'path': f.rfilename, 'size': f.size, 'sha256': sha,
                                  'upstream_lfs_sha256': expected})
        print(f'Verified {f.rfilename}', flush=True)
    (args.output / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
