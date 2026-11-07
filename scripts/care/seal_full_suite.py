"""Post-run integrity seal and explicit validation before reusing run caches.

This is a post-run byte inventory, not retroactive proof of when raw data were
created. It complements the pre-run protocol and original download sidecars.
Never overwrite an existing seal or silently accept changed/missing files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    out=Path('artifacts/care/full-v3');seal=out/'data_model_seal.json'
    if args.verify:
        manifest=json.loads(seal.read_text());fail=[]
        def check(entry):
            name,record=entry;path=Path(name)
            return name if not path.exists() or path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256'] else None
        with ThreadPoolExecutor(4) as pool:
            fail=[name for name in pool.map(check,manifest['files'].items()) if name]
        if fail:raise RuntimeError('Sealed artifacts changed or missing: '+repr(fail))
        print('Verified',len(manifest['files']),'sealed raw/data/cache/model files.');return
    if seal.exists():raise ValueError('Seal exists; use --verify, never overwrite')
    cfg=json.loads((out/'protocol.json').read_text())['config'];paths={out/'protocol.json',Path('configs/care/full-v3.json')}
    expectations={}
    for dataset,dc in cfg['datasets'].items():
        data=Path(dc['data']);rows=json.loads((data/'incidents.json').read_text());meta=json.loads((data/'metadata.json').read_text())
        paths.update(data/f for f in ['incidents.json','metadata.json','features.npz'])
        for name,expected in meta['source_hashes'].items():paths.add(Path(name));expectations[name]=expected
        for row in rows:
            source=Path(row['source']);paths.add(source)
            proof=source.with_name(source.name+'.verified.json')
            if proof.exists():paths.add(proof);expectations[str(source)]=json.loads(proof.read_text())['sha256']
            if dataset=='trainticket':paths.update(source.parent.glob('*.gz'))
        paths.update((out/dataset/'evidence').glob('*'))
        paths.update(Path(dc['rpc']).glob('*.npz'));paths.update(Path(dc['rpc']).glob('*.json'))
        for result in (out/dataset).glob('*/seed-*/correction/result.json'):
            report=json.loads(result.read_text());base=Path(report['base_directory'])
            paths.update(base/f for f in ['model.pt','predictions.npz','evaluation.json','history.json'])
            paths.update(result.parent.glob('*.pkl'));paths.update(result.parent.glob('*.npz'));paths.update(result.parent.glob('*.json'))
    paths=sorted(p for p in paths if p.is_file())
    total=sum(p.stat().st_size for p in paths);print('Hashing',len(paths),'files,',total,'bytes',flush=True)
    def record(path):
        name=str(path);h=digest(path)
        if name in expectations and expectations[name]!=h:raise ValueError('Existing raw source hash mismatch: '+name)
        return name,{'sha256':h,'bytes':path.stat().st_size}
    start=time.time()
    with ThreadPoolExecutor(4) as pool:files=dict(pool.map(record,paths))
    manifest={'created_at':time.time(),'hash_seconds':time.time()-start,'scope':'Post-run raw data, annotations, prepared features, RPC/evidence caches with ordered names, reused and new models, and predictions. Not a pre-run timestamp attestation.',
              'existing_source_hashes_checked':len(expectations),'total_bytes':total,'files':files}
    temporary=seal.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(seal)
    print('Sealed',len(files),'files; checked',len(expectations),'historical source hashes;',round(manifest['hash_seconds'],2),'seconds',flush=True)


if __name__=='__main__':main()
