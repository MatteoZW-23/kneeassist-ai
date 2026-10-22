import csv
import json
import random
import numpy as np
import pytest
import torch
from src.utils import config
from src.data.dataset import split_and_check
from src.training.train import rng_state,restore_rng

def test_patient_overlap_blocks_training(tmp_path):
    cfg=config();splits=tmp_path/'splits.json';splits.write_text(json.dumps({'fit':['a'],'tune':[],'official_valid':['b']}))
    mapping=tmp_path/'patients.csv';mapping.write_text('study_id,patient_id\na,P1\nb,P1\n')
    cfg['dataset']['splits']=str(splits);cfg['dataset']['patient_mapping']=str(mapping)
    with pytest.raises(ValueError,match='Patient leakage'):
        split_and_check([{'study_id':'a','split':'train'},{'study_id':'b','split':'valid'}],cfg)

def test_identical_series_across_splits_blocks_training(tmp_path):
    cfg=config();splits=tmp_path/'splits.json';splits.write_text(json.dumps({'fit':['a'],'tune':[],'official_valid':['b']}))
    cfg['dataset']['splits']=str(splits)
    volume=tmp_path/'volume.npy';np.save(volume,np.arange(2048,dtype=np.float32).reshape(2,32,32))
    rows=[{'study_id':study,'split':split,**{p+'_path':str(volume) for p in cfg['preprocessing']['planes']}}
          for study,split in [('a','train'),('b','valid')]]
    with pytest.raises(ValueError,match='Identical MRI series'):split_and_check(rows,cfg)

def test_resume_random_state_roundtrip():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    saved=rng_state()
    expected=(random.random(),np.random.rand(),torch.rand(3))
    restore_rng(saved)
    actual=(random.random(),np.random.rand(),torch.rand(3))
    assert expected[0]==actual[0] and expected[1]==actual[1] and torch.equal(expected[2],actual[2])
