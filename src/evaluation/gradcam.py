import numpy as np
import torch
import torch.nn.functional as F
from src.models.aggregation import aggregate
from src.data.preprocessing import InputError

def explain(predictor,volumes,target,plane):
    keys=[t['key'] for t in predictor.model_cfg['targets']]
    if target not in keys or plane not in volumes:raise InputError('Choose an available sequence and a supported finding.')
    tensors,prepared=predictor.tensors(volumes);model=predictor.model;model.eval()
    features=[];mask=[];activation=None
    # Compute final spatial maps without keeping the full encoder graph. Grad-CAM
    # needs gradients from these exact maps through pooling and the study head.
    for p in model.planes:
        if p not in tensors:
            features.append(torch.zeros(model.plane_dim,device=predictor.device));mask.append(0.);continue
        with torch.no_grad(),torch.autocast(device_type=predictor.device.type,enabled=False):maps=model.encoder(tensors[p].float())
        if p==plane:
            activation=maps.detach().float().requires_grad_(True);f=aggregate(activation)
        else:f=aggregate(maps.float()).detach()
        features.append(f);mask.append(1.)
    vector=torch.cat(features).unsqueeze(0);m=torch.tensor([mask],device=predictor.device)
    # Float32 head gradients avoid underflow in weak attention signals.
    logit=model.feature_logits(vector,m)[0,keys.index(target)]
    grad=torch.autograd.grad(logit,activation)[0]
    cam=torch.relu((grad.mean((-2,-1),keepdim=True)*activation).sum(1))
    size=predictor.model_cfg['preprocessing']['image_size']
    cam=F.interpolate(cam.unsqueeze(1),size=(size,size),mode='bilinear',align_corners=False)[:,0]
    maximum=cam.max();has_signal=bool(maximum.detach()>1e-10)
    if has_signal:cam=cam/maximum
    heatmaps=cam.detach().cpu().numpy();original=prepared[plane][1]
    # Warm attention overlay; opacity increases with positive contribution.
    rgb=np.repeat(original[:,:,:,None],3,axis=3)
    colors=np.stack([np.ones_like(heatmaps),.4*heatmaps,np.zeros_like(heatmaps)],axis=-1)
    alpha=(.65*heatmaps)[...,None];overlay=np.clip(rgb*(1-alpha)+colors*alpha,0,1)
    return {'original':original,'heatmaps':heatmaps,'overlay':overlay,'indices':prepared[plane][2],
            'suggested_slice':int(heatmaps.sum((1,2)).argmax()),'has_positive_attention':has_signal,
            'note':'Positive model attention for this finding. It is not a confirmed lesion or segmentation.'}
