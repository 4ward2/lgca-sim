"""Record installed distribution identities and available wheel byte hashes."""
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from .protocol import sha

def environment_lock(wheel_directory=None):
    packages={}
    for name in ('torch','numpy','scipy'):
        distribution=importlib.metadata.distribution(name)
        records=[p for p in distribution.files or [] if str(p).endswith('.dist-info/RECORD')]
        packages[name]=dict(version=distribution.version,record_sha256=sha(distribution.locate_file(records[0])) if records else None)
    root=Path(wheel_directory) if wheel_directory else None
    wheels={p.name:sha(p) for p in root.glob('*.whl')} if root and root.exists() else {}
    return dict(python=platform.python_version(),platform=platform.platform(),packages=packages,wheel_hashes=wheels,
                runner_image=os.environ.get('ImageVersion'),wheel_status='recorded' if wheels else 'missing; freeze must stop')

def fetch_cpu_wheels(destination):
    """Remote smoke records exact installed-version wheels without installing."""
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    packages=[f'{name}=={importlib.metadata.version(name)}' for name in ('torch','numpy','scipy')]
    for package in packages:
        command=[sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:','--dest',str(destination),package]
        if package.startswith('torch=='): command+=['--index-url','https://download.pytorch.org/whl/cpu']
        subprocess.run(command,check=True,capture_output=True,text=True,timeout=600)
