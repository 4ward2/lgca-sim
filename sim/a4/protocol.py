"""Frozen candidate protocol; an experiment is not locked by importing it."""
import hashlib
import json
from pathlib import Path

DATA_SHA = '5c9e54c44eb1ca0986854a92506e3ff1376b6de0'
DT_SHA = '152fb1182fb5dc11275859407ceec46102c20e42'
SEEDS = list(range(173101, 173111))
PILOTS = [173001, 173002, 173003]
LENGTHS = [40, 44, 48, 64, 128, 256, 512]
ARMS = ['b', 'r', 'f', 'dt-small', 'shared64', 'unshared64']
OBSERVE = [1,2,4,8,16,30,32,64,128,200,256,512,1024,1025]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def check_seed(seed, purpose='model'):
    allowed = SEEDS + PILOTS if purpose == 'model' else [173901,731032]+[732000+l for l in LENGTHS]+[740000+100*i+j for i in range(10) for j in range(10)]
    if seed in (12345,24680) or seed not in allowed:
        raise ValueError('unregistered seed')
    return seed

def validate_freeze(freeze, root):
    if freeze.get('status') != 'LOCKED':
        raise ValueError('experiment is not LOCKED')
    if freeze.get('seeds') != SEEDS or freeze.get('lengths') != LENGTHS:
        raise ValueError('matrix mismatch')
    if not freeze.get('files') or freeze.get('gate',{}).get('status')!='READY_FOR_REVIEW':
        raise ValueError('missing qualified freeze evidence')
    for name, expected in freeze['files'].items():
        path=(Path(root)/name).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError('freeze path escapes repository')
        if sha(path) != expected:
            raise ValueError('hash mismatch: '+name)
    return True
