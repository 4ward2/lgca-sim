"""Anonymous TLS downloads and fail-closed tar extraction."""
import datetime
import platform
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath
import numpy as np
import torch
from .protocol import DATA_SHA, DT_SHA, sha, write_json,read_json

BASE = 'https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/'

def safe_extract(archive, destination):
    root = Path(destination).resolve(); root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for m in members:
            p = PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name or not (m.isdir() or m.isfile()):
                raise ValueError('unsafe archive member: '+m.name)
            if not (root/m.name).resolve().is_relative_to(root):
                raise ValueError('archive escapes root')
        tar.extractall(root, members=members, filter='data')
    return [m.name for m in members]

def fetch_archive(url, out, expected_sha=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out/url.rsplit('/',1)[-1]
    if (out/'manifest.json').exists():
        previous=read_json(out/'manifest.json')
        if previous.get('url')!=url: raise ValueError('source URL changed in existing fetch artifact')
        if expected_sha and previous['sha256']!=expected_sha: raise ValueError('conflicting source pin')
        expected_sha=previous['sha256']
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as f:
                final_url = response.url
                while chunk := response.read(1024*1024):
                    f.write(chunk)
            break
        except OSError:
            if attempt == 2: raise
            time.sleep(1)
    actual = sha(path)
    if expected_sha and actual != expected_sha:
        raise ValueError('archive hash mismatch')
    members = safe_extract(path, out/'raw')
    manifest = dict(url=url, final_url=final_url, timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    bytes=path.stat().st_size, sha256=actual, members=members,
                    files={str(p.relative_to(out)):sha(p) for p in sorted((out/'raw').rglob('*')) if p.is_file()},
                    generator_commit=DATA_SHA, baseline_commit=DT_SHA,
                    versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    write_json(out/'manifest.json',manifest)
    write_json(out/'source_pin.json',dict(url=url,sha256=actual))
    return manifest
