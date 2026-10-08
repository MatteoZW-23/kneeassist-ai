import numpy as np
import torch
import torch.nn.functional as F
from src.models.aggregation import aggregate
from src.data.preprocessing import InputError


def describe_attention_region(heatmap):
    """Describe the strongest Grad-CAM area in displayed-image coordinates.

    Grad-CAM is not a lesion detector. The returned location intentionally uses
    only image positions (for example, "upper-centre of the displayed image")
    because this prototype has no validated anatomical localisation labels.
    """
    values = np.asarray(heatmap, dtype=float)
    finite = np.isfinite(values)
    if values.ndim != 2 or not finite.any():
        return {"available": False, "reason": "No usable attention values for this slice."}
    peak = float(np.nanmax(values))
    if peak <= 1e-10:
        return {"available": False, "reason": "No positive Grad-CAM signal for this slice."}

    # Keep only the strongest part of this slice's own heatmap. This describes
    # contribution to the selected model score, not a clinical contour.
    active = values >= (0.70 * peak)
    rows, cols = np.where(active)
    if len(rows) == 0:
        return {"available": False, "reason": "No concentrated Grad-CAM region for this slice."}

    height, width = values.shape
    centre_row, centre_col = float(rows.mean()), float(cols.mean())
    vertical = ("upper", "central", "lower")[min(2, int(centre_row * 3 / height))]
    horizontal = ("left", "centre", "right")[min(2, int(centre_col * 3 / width))]
    peak_row, peak_col = np.unravel_index(np.nanargmax(values), values.shape)
    return {
        "available": True,
        "display_region": f"{vertical}-{horizontal} area of the displayed image",
        "peak_pixel": {"row": int(peak_row + 1), "column": int(peak_col + 1)},
        "highlighted_area_percent": round(float(active.mean() * 100), 1),
        "peak_relative_attention": round(peak, 3),
    }


def is_valid_attention_payload(payload):
    """Validate the parallel arrays stored in the Streamlit Grad-CAM cache.

    The cache is browser-session state, so an older payload must be rejected
    rather than indexed after application code changes.
    """
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        return False
    indices = payload.get("indices")
    regions = payload.get("regions")
    if not isinstance(indices, (list, tuple)) or not isinstance(regions, list) or not indices:
        return False
    try:
        original = np.asarray(payload["original"])
        heatmaps = np.asarray(payload["heatmaps"])
        overlay = np.asarray(payload["overlay"])
    except (KeyError, TypeError, ValueError):
        return False
    slice_count = len(indices)
    if (
        original.ndim != 3
        or heatmaps.ndim != 3
        or overlay.ndim != 4
        or original.shape != heatmaps.shape
        or overlay.shape != original.shape + (3,)
        or original.shape[0] != slice_count
        or len(regions) != slice_count
        or not np.isfinite(original).all()
        or not np.isfinite(heatmaps).all()
        or not np.isfinite(overlay).all()
    ):
        return False
    suggested = payload.get("suggested_slice")
    return isinstance(suggested, (int, np.integer)) and not isinstance(suggested, bool) and 0 <= int(suggested) < slice_count


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
        attention_module = model.attention_modules[p] if "attention" in model.aggregation_method else None
        if p==plane:
            activation=maps.detach().float().requires_grad_(True)
            f=aggregate(activation, method=model.aggregation_method, attention_module=attention_module)
        else:
            f=aggregate(maps.float(), method=model.aggregation_method, attention_module=attention_module).detach()
        features.append(f);mask.append(1.)
    vector=torch.cat(features).unsqueeze(0);m=torch.tensor([mask],device=predictor.device)
    # Float32 head gradients avoid underflow in weak attention signals.
    logit=model.feature_logits(vector,m)[0,keys.index(target)]
    grad=torch.autograd.grad(logit,activation)[0]
    cam=torch.relu((grad.mean((-2,-1),keepdim=True)*activation).sum(1))
    feature_map_size = tuple(int(value) for value in activation.shape[-2:])
    size=predictor.model_cfg['preprocessing']['image_size']
    cam=F.interpolate(cam.unsqueeze(1),size=(size,size),mode='bilinear',align_corners=False)[:,0]
    maximum=cam.max();has_signal=bool(maximum.detach()>1e-10)
    if has_signal:cam=cam/maximum
    heatmaps=cam.detach().cpu().numpy();original=prepared[plane][1]
    # Warm attention overlay; opacity increases with positive contribution.
    rgb=np.repeat(original[:,:,:,None],3,axis=3)
    colors=np.stack([np.ones_like(heatmaps),.4*heatmaps,np.zeros_like(heatmaps)],axis=-1)
    alpha=(.65*heatmaps)[...,None];overlay=np.clip(rgb*(1-alpha)+colors*alpha,0,1)
    regions=[describe_attention_region(heatmap) for heatmap in heatmaps]
    return {'schema_version':2,'original':original,'heatmaps':heatmaps,'overlay':overlay,'indices':prepared[plane][2],
            'suggested_slice':int(heatmaps.sum((1,2)).argmax()),'has_positive_attention':has_signal,
            'regions':regions,'feature_map_size':feature_map_size,
            'note':'Grad-CAM shows image areas that contributed positively to this selected model score. The map is coarse and is enlarged for display. It is not a confirmed lesion, anatomical label, detection box or segmentation mask.'}
