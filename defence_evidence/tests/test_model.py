import numpy as np
import torch
from src.utils import config
from src.models.study_model import StudyModel
from src.evaluation.metrics import calculate

def test_architectures_and_masking():
    cfg=config()
    for arch in cfg['model']['architectures']:
        model=StudyModel(arch,cfg,pretrained=False).eval()
        f=torch.zeros(2,model.plane_dim*3);mask=torch.tensor([[1.,1.,1.],[1.,0.,0.]])
        a=model.feature_logits(f,mask)
        f[1,model.plane_dim:]=1e5
        b=model.feature_logits(f,mask)
        assert a.shape==(2,3) and torch.allclose(a,b)

def test_extended_architecture_factories():
    cfg=config()
    expected={'resnet50':2048,'densenet121':1024,'swin_t':768}
    for arch,channels in expected.items():
        model=StudyModel(arch,cfg,pretrained=False).eval()
        assert model.channels==channels and model.head[-1].out_features==3

def test_metrics_known_confusion():
    targets=[{'key':'x'}];y=np.array([[0],[0],[1],[1]]);p=np.array([[.1],[.7],[.4],[.9]])
    m=calculate(y,p,targets,{'x':.5})
    assert m['x']['confusion_matrix']==[[1,1],[1,1]]
    assert m['x']['accuracy']==.5 and m['x']['f1']==.5 and m['x']['auroc']==.75
