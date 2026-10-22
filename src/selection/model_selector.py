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


METRIC_ALIASES = {
    # Older evidence used the generic PR-AUC label even though calculate() stores
    # sklearn average precision. Keep old registries readable but use the stored
    # metric field for every new selection.
    'pr_auc': 'average_precision',
}


def metric_value(metrics, name):
    """Return a validated metric, including the legacy PR-AUC alias."""
    if name in metrics:
        return float(metrics[name])
    alias = METRIC_ALIASES.get(name)
    if alias and alias in metrics:
        return float(metrics[alias])
    raise ValueError(f"Selection policy metric '{name}' is missing from registered validation evidence.")


def _score(metrics, weights):
    return sum(metric_value(metrics, key) * float(weight) for key, weight in weights.items())


def select_models(available_planes, targets, path=None):
    """Choose the strongest eligible model per target after input compatibility checks.

    The candidate comparison happens after the MRI input is inspected. It uses
    stored validation evidence only: an uploaded patient's prediction is never
    used to choose a model. An incomplete study remains eligible only when an
    entry explicitly permits its available planes.
    """
    registry=load_registry(path);planes=set(available_planes);weights=registry['selection_policy']['weights']
    # ``benchmark_candidates`` is the fixed experimental population.  It is
    # intentionally separate from ``models``: the latter contains checkpoints
    # approved for deployment.  Selection first ranks the entire recorded
    # validation population, then verifies that the preselected winner has a
    # locally available registered checkpoint.  It never silently promotes a
    # lower-ranked candidate because a benchmark artefact was archived.
    candidates_source=registry.get('benchmark_candidates',registry['models'])
    deployed={model['id']:model for model in registry['models']}
    compatible=[]
    for model in candidates_source:
        supported=set(model.get('supported_planes',[]))
        if not planes or not planes.issubset(supported) or len(planes)<int(model.get('minimum_planes',1)):
            continue
        compatible.append(model)
    selected=[]
    candidates_by_target={}
    for target in targets:
        candidates=[model for model in compatible if target in model.get('supported_targets',[]) and target in model.get('validation_metrics',{})]
        ranked=sorted(candidates,key=lambda model:_score(model['validation_metrics'][target],weights),reverse=True)
        candidates_by_target[target]=[
            {
                'model_id':model['id'],
                'architecture':model['architecture'],
                'selection_score':_score(model['validation_metrics'][target],weights),
                'metrics':model['validation_metrics'][target],
                'locally_available': bool(
                    model['id'] in deployed and (ROOT/deployed[model['id']]['checkpoint']).is_file()
                ),
            }
            for model in ranked
        ]
        if not candidates:
            selected.append({'target':target,'status':'No compatible validated model','reason':'No registry entry supports the available sequences and target.'})
            continue
        winner=ranked[0]
        deployed_winner=deployed.get(winner['id'])
        if deployed_winner is None or not (ROOT/deployed_winner['checkpoint']).is_file():
            selected.append({
                'target':target,
                'status':'Selected validation candidate unavailable',
                'model_id':winner['id'],
                'architecture':winner['architecture'],
                'selection_score':_score(winner['validation_metrics'][target],weights),
                'reason':'The highest-ranked compatible validation candidate is not installed as a registered local checkpoint. No lower-ranked candidate was substituted.',
            })
            continue
        metrics=winner['validation_metrics'][target]
        selected.append({
            'target':target,'status':'Selected','model_id':winner['id'],'architecture':winner['architecture'],
            'version':deployed_winner['version'],'selection_score':_score(metrics,weights),'metrics':metrics,
            'reason':f"Selected after upload from {len(candidates)} compatible benchmark candidates: highest fixed validation-evidence score, weighted toward AUROC and sensitivity. The locally registered winning checkpoint was verified before inference.",
            'checkpoint':deployed_winner['checkpoint'],'limitations':deployed_winner.get('limitations',winner.get('limitations',[]))
        })
    return {
        'policy':registry['selection_policy'],
        'available_planes':sorted(planes),
        'compatible_models':[model['id'] for model in compatible],
        'locally_available_models':[model['id'] for model in compatible if model['id'] in deployed and (ROOT/deployed[model['id']]['checkpoint']).is_file()],
        'candidates_by_target':candidates_by_target,
        'selections':selected,
    }
