"""Validated ensemble planning and aggregation for research inference.

An ensemble is deliberately unavailable until the registry records members,
locked validation evidence and non-guessed weights.  This module never turns a
set of experimental checkpoints into an ensemble merely because they exist.
"""
import json
from pathlib import Path

import numpy as np

from src.utils import ROOT


class EnsembleError(ValueError):
    """Raised when an ensemble specification is incomplete or unsafe to use."""


def _registry(path=None):
    registry_path = Path(path or ROOT / 'model_registry' / 'active_models.json')
    with registry_path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _compatible(model, available_planes):
    checkpoint = ROOT / model.get('checkpoint', '')
    supported = set(model.get('supported_planes', []))
    planes = set(available_planes)
    return checkpoint.is_file() and bool(planes) and planes.issubset(supported) and len(planes) >= int(model.get('minimum_planes', 1))


def ensemble_plans(available_planes, targets, path=None):
    """Return an auditable plan for each target without loading any model.

    Eligible ensembles require an explicit ``validated_improvement`` registry
    status.  This prevents validation-only candidates and hand-picked weights
    from silently affecting a study result.
    """
    registry = _registry(path)
    models = {entry.get('id'): entry for entry in registry.get('models', [])}
    plans = []
    for target in targets:
        candidates = []
        for spec in registry.get('ensembles', []):
            if target not in spec.get('targets', []):
                continue
            members = [models.get(model_id) for model_id in spec.get('members', [])]
            if (spec.get('validation', {}).get('status') != 'validated_improvement' or not members or
                    any(member is None or target not in member.get('supported_targets', []) or not _compatible(member, available_planes) for member in members)):
                continue
            weights = spec.get('weights', {}).get(target, {})
            if set(weights) != set(spec.get('members', [])):
                continue
            try:
                _normalised_weights(weights, spec['members'])
            except EnsembleError:
                continue
            candidates.append(spec)
        if candidates:
            winner = max(candidates, key=lambda spec: float(spec['validation']['metrics'][target]['auroc']))
            plans.append({'target': target, 'status': 'Eligible validated ensemble', 'ensemble_id': winner['id'],
                          'members': winner['members'], 'weights': winner['weights'][target],
                          'validation': winner['validation']})
        else:
            plans.append({'target': target, 'status': 'Single-model fallback',
                          'reason': 'No compatible ensemble with locked evidence of validation improvement is registered.'})
    return {'policy': registry.get('ensemble_policy', {}), 'available_planes': sorted(set(available_planes)), 'plans': plans}


def _normalised_weights(weights, member_ids):
    if set(weights) != set(member_ids):
        raise EnsembleError('Ensemble weights must cover each member exactly once.')
    values = np.asarray([float(weights[member_id]) for member_id in member_ids], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise EnsembleError('Ensemble weights must be finite, non-negative and have a positive sum.')
    return values / values.sum()


def aggregate_probabilities(member_probabilities, weights, member_ids, disagreement_thresholds=None):
    """Compute a weighted probability and an explicit member-disagreement flag.

    Callers must supply calibrated probabilities from an approved registry
    ensemble.  The return value is a numeric aggregation only; it does not make
    a diagnostic claim.
    """
    if set(member_probabilities) != set(member_ids):
        raise EnsembleError('Probabilities must be supplied for each ensemble member.')
    values = np.asarray([float(member_probabilities[member_id]) for member_id in member_ids], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise EnsembleError('Member probabilities must be finite values between zero and one.')
    normalised = _normalised_weights(weights, member_ids)
    deviation = float(np.std(values))
    limits = disagreement_thresholds or {'moderate': 0.10, 'high': 0.20}
    if deviation >= float(limits['high']):
        level = 'High disagreement'
    elif deviation >= float(limits['moderate']):
        level = 'Moderate disagreement'
    else:
        level = 'Low disagreement'
    return {'probability': float(np.dot(normalised, values)), 'member_probabilities': dict(member_probabilities),
            'weights': {member_id: float(weight) for member_id, weight in zip(member_ids, normalised)},
            'disagreement_standard_deviation': deviation, 'disagreement_level': level,
            'note': 'Model disagreement is a spread of approved model scores, not a probability that the output is correct.'}
