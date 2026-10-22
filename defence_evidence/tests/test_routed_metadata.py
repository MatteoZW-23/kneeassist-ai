import copy

from src.inference.routed_predictor import RoutedPredictor
from src.utils import config


class _FakePredictor:
    def __init__(self, probabilities, variations):
        self._probabilities = probabilities
        self._variations = variations

    def predict(self, volumes, case_reference, uncertainty_samples=1):
        findings = []
        for key, probability in self._probabilities.items():
            findings.append({
                'key': key,
                'finding': key,
                'probability': probability,
                'threshold': 0.5,
                'flagged': probability >= 0.5,
                'uncertainty': self._variations[key],
                'uncertainty_level': 'small',
                'status': 'Flagged for research review' if probability >= 0.5 else 'Below research threshold',
            })
        return {
            'findings': findings,
            'available_planes': list(volumes),
            'missing_planes': [],
            'score_separation': 'far from threshold',
            'probability_note': 'arbitrary base note',
            'uncertainty_note': 'arbitrary base note',
            'architecture': 'fake',
            'checkpoint_file': 'fake.pth',
            'checkpoint_sha256': 'fake',
        }

    def explain(self, volumes, target, plane):
        return {'model_id': self._probabilities['model_id'], 'target': target, 'plane': plane}


def test_routed_metadata_is_recomputed_from_the_merged_target_findings():
    cfg = copy.deepcopy(config())
    routed = object.__new__(RoutedPredictor)
    routed.cfg = cfg
    routed.models = {
        'dense': {'checkpoint_sha256': 'dense-hash', 'supported_targets': ['abnormal']},
        'resnet': {'checkpoint_sha256': 'resnet-hash', 'supported_targets': ['acl']},
        'swin': {'checkpoint_sha256': 'swin-hash', 'supported_targets': ['meniscus']},
    }
    predictors = {
        'dense': _FakePredictor({'abnormal': 0.8, 'acl': 0.9, 'meniscus': 0.9}, {'abnormal': 0.01, 'acl': 0.01, 'meniscus': 0.01}),
        'resnet': _FakePredictor({'abnormal': 0.9, 'acl': 0.55, 'meniscus': 0.9}, {'abnormal': 0.01, 'acl': 0.20, 'meniscus': 0.01}),
        'swin': _FakePredictor({'abnormal': 0.9, 'acl': 0.9, 'meniscus': 0.7}, {'abnormal': 0.01, 'acl': 0.01, 'meniscus': 0.30}),
    }
    routed._predictor = lambda model_id: predictors[model_id]
    selections = [
        {'target': 'abnormal', 'status': 'Selected', 'model_id': 'dense', 'architecture': 'densenet121', 'reason': 'fixed evidence'},
        {'target': 'acl', 'status': 'Selected', 'model_id': 'resnet', 'architecture': 'resnet18', 'reason': 'fixed evidence'},
        {'target': 'meniscus', 'status': 'Selected', 'model_id': 'swin', 'architecture': 'swin_t', 'reason': 'fixed evidence'},
    ]

    result = routed.predict({'axial': object()}, 'test', selections, uncertainty_samples=10)

    assert [item['probability'] for item in result['findings']] == [0.8, 0.55, 0.7]
    assert result['score_separation'] == 'near threshold'
    assert '0.170' in result['uncertainty_note']
    assert result['selected_models']['acl']['checkpoint_sha256'] == 'resnet-hash'
    assert result['architecture'] == 'validation-based target routing'
