"""Read nested ZIP directory via HTTP Range; download only missing original files.

The public mirror stores training_data_with_faults.zip uncompressed in its outer
ZIP. This avoids downloading the unrelated normal/test archives or a second copy.
"""
import argparse
import hashlib
import io
import json
import shutil
import struct
import time
import zipfile
from pathlib import Path
import requests

URL = 'https://zenodo.org/api/records/19176851/files/aiops2022.zip/content'
SIZE = 9506493153


class RangeFile(io.RawIOBase):
    def __init__(self, url=URL, size=SIZE, offset=0):
        self.url, self.size, self.offset, self.pos = url, size, offset, 0
        self.session = requests.Session()
        self.cache_start, self.cache = -1, b''

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if self.pos < 0:
            raise ValueError('negative seek')
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size)
        if n <= 0:
            return b''
        if self.cache_start <= self.pos and self.pos + n <= self.cache_start + len(self.cache):
            begin = self.pos - self.cache_start
            self.pos += n
            return self.cache[begin:begin+n]
        begin = self.offset + self.pos
        length = min(self.size-self.pos, max(n, 16 * 1024**2))
        end = begin + length - 1
        for attempt in range(5):
            try:
                response = self.session.get(self.url, headers={'Range': f'bytes={begin}-{end}',
                                             'Accept-Encoding': 'identity'}, timeout=(20, 120))
                response.raise_for_status()
                expected = f'bytes {begin}-{end}/{SIZE}'
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise RuntimeError('Server did not honor exact byte range')
                if len(response.content) != length:
                    raise IOError('Truncated range')
                self.cache_start, self.cache = self.pos, response.content
                self.pos += n
                return self.cache[:n]
            except (requests.RequestException, IOError):
                if attempt == 4:
                    raise
                time.sleep(2**attempt)


def open_training_zip():
    remote = RangeFile()
    outer = zipfile.ZipFile(remote)
    info = outer.getinfo('training_data_with_faults.zip')
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError('Outer training member is compressed; direct nested ranges unavailable')
    remote.seek(info.header_offset)
    header = remote.read(30)
    if header[:4] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP header')
    name_length, extra_length = struct.unpack_from('<HH', header, 26)
    offset = info.header_offset + 30 + name_length + extra_length
    return zipfile.ZipFile(RangeFile(size=info.file_size, offset=offset))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('data/care/aiops2022-original'))
    p.add_argument('--index-only', action='store_true')
    p.add_argument('--contains', action='append', default=[], help='Include members matching ANY substring')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    z = open_training_zip()
    index = [{'path': i.filename, 'size': i.file_size, 'compressed': i.compress_size,
              'crc32': f'{i.CRC:08x}'} for i in z.infolist()
             if not i.is_dir() and not i.filename.startswith('__MACOSX/') and '/.DS_Store' not in i.filename]
    (args.output / 'upstream_index.json').write_text(json.dumps(index, indent=2) + '\n')
    selected = [x for x in index if not args.contains or any(s in x['path'] for s in args.contains)]
    print(f'{len(index)} files; selected {len(selected)}; unpacked={sum(x["size"] for x in selected)}', flush=True)
    if args.index_only:
        for row in index:
            print(row, flush=True)
        return
    remaining = sum(x['size'] for x in selected if not (args.output / x['path']).exists())
    if shutil.disk_usage(args.output).free < remaining + 8 * 1024**3:
        raise RuntimeError('Need selected uncompressed bytes plus 8 GiB reserve')
    for row in selected:
        path = (args.output / row['path']).resolve()
        if not path.is_relative_to(args.output.resolve()):
            raise ValueError('Unsafe archive path')
        proof = path.with_name(path.name + '.verified.json')
        if path.exists() and proof.exists() and path.stat().st_size == row['size']:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        tmp = path.with_name(path.name + '.partial')
        print('Extracting', row['path'], row['size'], flush=True)
        with z.open(row['path']) as source, tmp.open('wb') as out:
            for block in iter(lambda: source.read(4 * 1024**2), b''):
                if shutil.disk_usage(args.output).free < 8 * 1024**3:
                    raise RuntimeError('Stopping at 8 GiB free-space reserve')
                digest.update(block)
                out.write(block)
        # ZipExtFile verifies the upstream CRC32 at EOF.
        if tmp.stat().st_size != row['size']:
            raise IOError('Wrong extracted size')
        tmp.replace(path)
        proof.write_text(json.dumps({**row, 'sha256': digest.hexdigest(), 'source': URL}, indent=2)+'\n')


if __name__ == '__main__':
    main()
