import numpy as np
import torch
from src.utils import config
from src.models.study_model import StudyModel
from src.training.finetune import enable_final_blocks
from src.evaluation.calibration import fit_calibration,apply_calibration,balanced_thresholds

def test_encoder_gradient_and_frozen_batchnorm():
    torch.set_num_threads(2)
    for architecture in ['resnet18','efficientnet_b0']:
        model=StudyModel(architecture,config(),pretrained=False)
        params=enable_final_blocks(model,architecture); model.train()
        assert not model.encoder.training
        assert not next(model.encoder.parameters()).requires_grad
        before=params[0].detach().clone()
        loss=model({'sagittal':torch.randn(2,3,64,64)}).square().sum()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert sum(p.grad.abs().sum().item() for p in params)>0
        assert torch.equal(params[0].detach(),before)

def test_calibration_ranking_and_thresholds():
    rng=np.random.default_rng(42);z=rng.normal(size=(200,1))
    y=(rng.random((200,1))<1/(1+np.exp(-z))).astype(int)
    targets=[{'key':'acl'}]
    calibration=fit_calibration(y,z,targets)
    p=apply_calibration(z,calibration)
    assert np.array_equal(np.argsort(z[:,0]),np.argsort(p[:,0]))
    thresholds=balanced_thresholds(y,p,targets)
    assert 0<=thresholds['acl']<=1 and np.isfinite(p).all()
    assert calibration['cases']==200
