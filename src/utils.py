import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

def config(path=None):
    with open(path or ROOT/'config.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resolve(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p

def save_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def seed_everything(seed, threads=4):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=True)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def get_device(cfg):
    request=cfg['training']['device']
    if request=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no compatible GPU is available.')
    return torch.device('cuda' if request!='cpu' and torch.cuda.is_available() else 'cpu')
