import csv
import hashlib
import json
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from src.utils import resolve, save_json, ROOT

def stack_path(row,plane,cfg):
    # Default manifests are relative to the project. Changing dataset.root
    # relocates the known MRNet split/plane layout without editing every row.
    value=row[plane+'_path']
    from pathlib import Path
    p=Path(value)
    return p if p.is_absolute() else resolve(cfg['dataset']['root'])/row['split']/plane/(row['study_id']+'.npy')

def read_manifest(cfg):
    with resolve(cfg['dataset']['manifest']).open(newline='') as f: rows=list(csv.DictReader(f))
    keys=[t['key'] for t in cfg['targets']]
    ids=[r['study_id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate study IDs in the manifest.')
    for r in rows:
        for key in keys:
            if r.get(key) not in ('0','1'): raise ValueError(f'Missing/unreliable target {key} in study {r["study_id"]}.')
        for plane in cfg['preprocessing']['planes']:
            if not stack_path(r,plane,cfg).is_file(): raise FileNotFoundError(f'Missing {plane} stack for study {r["study_id"]}.')
    return rows

def split_and_check(rows,cfg):
    path=resolve(cfg['dataset']['splits'])
    by_id={r['study_id']:r for r in rows}
    patient={}
    if cfg['dataset'].get('patient_mapping'):
        with resolve(cfg['dataset']['patient_mapping']).open(newline='') as f:
            patient={r['study_id']:r['patient_id'] for r in csv.DictReader(f)}
        if any(not patient.get(k) for k in by_id): raise ValueError('Patient mapping must cover every study.')
    if path.exists(): split=json.loads(path.read_text())
    else:
        train=[r['study_id'] for r in rows if r['split']=='train']; valid=[r['study_id'] for r in rows if r['split']=='valid']
        if patient:
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=cfg['training']['validation_fraction'],random_state=cfg['seed']).split(train,groups=[patient[k] for k in train]))
            fit=[train[i] for i in a]; tune=[train[i] for i in b]
        else:
            pattern=[''.join(by_id[k][t['key']] for t in cfg['targets']) for k in train]
            if min(Counter(pattern).values())<2: pattern=[by_id[k]['abnormal'] for k in train]
            fit,tune=train_test_split(train,test_size=cfg['training']['validation_fraction'],random_state=cfg['seed'],stratify=pattern)
        split={'fit':fit,'tune':tune,'official_valid':valid}; save_json(path,split)
    roles={}
    for role,ids in split.items():
        for k in ids:
            if k in roles or k not in by_id: raise ValueError('Study split leakage or an unknown study ID was found.')
            if by_id[k]['split']!=('valid' if role=='official_valid' else 'train'): raise ValueError('Official validation split was changed.')
            roles[k]=role
    if set(roles)!=set(by_id): raise ValueError('The split does not cover all studies.')
    if patient:
        groups={}
        for k,role in roles.items():
            if patient[k] in groups and groups[patient[k]]!=role: raise ValueError('Patient leakage across partitions.')
            groups[patient[k]]=role
    # Content checks prevent identical series in different partitions.
    hashes={}; file_hashes={}; duplicate_within=0
    for row in rows:
        for p in cfg['preprocessing']['planes']:
            path=stack_path(row,p,cfg)
            with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            file_hashes[row['study_id']+'/'+p]=digest
            role=roles[row['study_id']]
            if digest in hashes:
                if hashes[digest]!=role: raise ValueError('Identical MRI series found in different data partitions. Resolve before training.')
                duplicate_within+=1
            hashes[digest]=role
    report={'dataset':'MRNet-v1.0','split_counts':{k:len(v) for k,v in split.items()},
            'study_overlap':0,'exact_series_overlap_across_splits':0,
            'within_partition_duplicate_series':duplicate_within,
            'patient_grouping_verified':bool(patient),
            'limitation':'Patient linkage is absent from this MRNet copy; study separation cannot prove patient independence.' if not patient else None,
            'prior_evaluation_disclosure':'The official 120-exam validation set was previously evaluated in the earlier baseline; it is not a new external test set.'}
    save_json(resolve(cfg['evaluation']['directory'])/'leakage_audit.json',report)
    save_json(ROOT/'data/content_hashes.json',file_hashes)
    return split,report

def volumes(row,cfg):
    return {p:np.load(stack_path(row,p,cfg),allow_pickle=False) for p in cfg['preprocessing']['planes']}
