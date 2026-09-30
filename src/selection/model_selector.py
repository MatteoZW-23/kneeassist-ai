"""Transparent selection from a registry of existing, validated research checkpoints."""
import json
from pathlib import Path

from src.utils import ROOT


def load_registry(path=None):
    registry_path=Path(path or ROOT/'model_registry'/'active_models.json')
    with registry_path.open(encoding='utf-8') as stream:
        registry=json.load(stream)
    if not isinstance(registry.get('models'),list):
        raise ValueError('Model registry does not contain a models list.')
    return registry


def _score(metrics, weights):
    return sum(float(metrics.get(key,0.0))*float(weight) for key,weight in weights.items())


def select_models(available_planes, targets, path=None):
    """Choose the strongest eligible model per target using stored validation evidence.

    This function does not invent a candidate, combine models, or override the
    registry. An incomplete study remains eligible only when an entry explicitly
    permits its available planes.
    """
    registry=load_registry(path);planes=set(available_planes);weights=registry['selection_policy']['weights']
    compatible=[]
    for model in registry['models']:
        checkpoint=ROOT/model['checkpoint']
        supported=set(model.get('supported_planes',[]))
        if not checkpoint.is_file() or not planes or not planes.issubset(supported) or len(planes)<int(model.get('minimum_planes',1)):
            continue
        compatible.append(model)
    selected=[]
    for target in targets:
        candidates=[model for model in compatible if target in model.get('supported_targets',[]) and target in model.get('validation_metrics',{})]
        if not candidates:
            selected.append({'target':target,'status':'No compatible validated model','reason':'No registry entry supports the available sequences and target.'})
            continue
        winner=max(candidates,key=lambda model:_score(model['validation_metrics'][target],weights))
        metrics=winner['validation_metrics'][target]
        selected.append({
            'target':target,'status':'Selected','model_id':winner['id'],'architecture':winner['architecture'],
            'version':winner['version'],'selection_score':_score(metrics,weights),'metrics':metrics,
            'reason':f"Highest registry score among {len(candidates)} compatible validated research model(s), weighted toward AUROC and sensitivity.",
            'checkpoint':winner['checkpoint'],'limitations':winner['limitations']
        })
    return {'policy':registry['selection_policy'],'available_planes':sorted(planes),'compatible_models':[model['id'] for model in compatible],'selections':selected}
