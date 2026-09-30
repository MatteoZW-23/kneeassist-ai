import pytest

from src.localization.registry import localization_status
from src.selection.ensemble import EnsembleError, aggregate_probabilities, ensemble_plans


def test_empty_registry_uses_single_model_fallback_and_no_localizer():
    plans = ensemble_plans(['axial', 'coronal', 'sagittal'], ['abnormal', 'acl', 'meniscus'])
    assert all(plan['status'] == 'Single-model fallback' for plan in plans['plans'])
    status = localization_status()
    assert not status['detection']['available']
    assert not status['segmentation']['available']


def test_approved_ensemble_math_and_disagreement_are_explicit():
    output = aggregate_probabilities(
        {'efficientnet': 0.94, 'swin': 0.90, 'densenet': 0.87},
        {'efficientnet': 0.50, 'swin': 0.30, 'densenet': 0.20},
        ['efficientnet', 'swin', 'densenet'],
    )
    assert output['probability'] == pytest.approx(0.914)
    assert output['disagreement_level'] == 'Low disagreement'


def test_ensemble_rejects_missing_members_or_invalid_probabilities():
    with pytest.raises(EnsembleError):
        aggregate_probabilities({'one': 0.6}, {'one': 1.0, 'two': 1.0}, ['one', 'two'])
    with pytest.raises(EnsembleError):
        aggregate_probabilities({'one': 1.1}, {'one': 1.0}, ['one'])
