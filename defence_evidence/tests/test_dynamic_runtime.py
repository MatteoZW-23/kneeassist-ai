"""Regression checks using actual trained DMS artifacts, never synthetic performance."""
import copy
import csv
import json

import numpy as np
import pytest

from src.data.preprocessing import InputError, load_uploads
from src.dynamic_selection.selector import load_selector_artifact, save_selector_artifact
from src.evaluation.gradcam import is_valid_attention_payload
from src.inference.dynamic_predictor import DynamicSelectionPredictor, DynamicSelectionUnavailable, dynamic_selection_status
from src.inference.predictor import text_summary
from src.utils import ROOT, config


ARTIFACT = ROOT / 'runs/dynamic_selection_v1/selector/dynamic_selector.joblib'
PREDICTIONS = ROOT / 'runs/dynamic_selection_v1/final_predictions.csv'
pytestmark = pytest.mark.skipif(not ARTIFACT.is_file() or not PREDICTIONS.is_file(), reason='Completed local DMS training is required')


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    cfg = copy.deepcopy(config())
    artifact = load_selector_artifact(ARTIFACT, require_active=False)
    # A temporary active copy permits verification before live promotion. The
    # real checkpoint files and learned competence estimators are unchanged.
    artifact['activation_status'] = 'active'
    path = tmp_path_factory.mktemp('dms-runtime') / 'selector.joblib'
    save_selector_artifact(path, artifact)
    cfg['dynamic_selection'] = {'enabled': True, 'artifact': str(path)}
    study = load_uploads([('study.zip',(ROOT/'sample_cases/mrnet_1130.zip').read_bytes())],cfg['preprocessing'])
    predictor = DynamicSelectionPredictor(cfg)
    result = predictor.predict(study,'DMS-runtime-verification')
    return cfg, study, predictor, result


def test_real_all_five_outputs_match_saved_fp32_evaluation_and_final_selection(runtime):
    _, study, predictor, result = runtime
    with PREDICTIONS.open(newline='',encoding='utf-8') as stream:
        recorded = next(row for row in csv.DictReader(stream) if row['study_id']=='1130')
    dynamic = result['dynamic_selection']
    assert dynamic['models_evaluated']==5
    assert dynamic['selected_model_id']==recorded['selected_model_id']
    for model_id, actual in dynamic['base_probabilities'].items():
        expected = [float(recorded[f'{model_id}::{target}']) for target in ('abnormal','acl','meniscus')]
        assert np.allclose(actual,expected,atol=1e-4)
    assert np.allclose([f['probability'] for f in result['findings']],dynamic['base_probabilities'][dynamic['selected_model_id']],atol=1e-8)
    repeat = predictor.predict(study,'DMS-runtime-repeat')
    assert repeat['dynamic_selection']['selected_model_id']==dynamic['selected_model_id']
    assert repeat['dynamic_selection']['base_probabilities']==dynamic['base_probabilities']
    assert dynamic['principal_factors']
    assert len(dynamic['candidate_metadata'])==5
    assert all(entry['seconds_including_checkpoint_load']>0 for entry in dynamic['candidate_metadata'].values())


def test_selected_model_gradcam_and_exports_use_the_actual_selected_architecture(runtime):
    _, study, predictor, result = runtime
    selected = result['dynamic_selection']['selected_model_id']
    attention = predictor.explain(study,'acl','sagittal',selected)
    assert is_valid_attention_payload(attention)
    assert np.isfinite(attention['overlay']).all()
    summary = text_summary(result)
    assert 'Models evaluated: 5 of 5' in summary
    assert selected in summary
    assert 'Estimated mean Brier loss:' in summary
    assert 'MC-dropout variation ±0.000' not in summary
    serialised = json.dumps(result,allow_nan=False)
    assert selected in serialised


def test_every_registered_candidate_can_generate_gradcam_for_the_same_real_study(runtime):
    """Each selectable architecture must support attention before it can be registered."""
    _, study, predictor, _ = runtime
    for model_id in predictor.artifact['model_ids']:
        attention = predictor.explain(study, 'acl', 'sagittal', model_id)
        assert is_valid_attention_payload(attention), model_id
        assert np.isfinite(attention['overlay']).all(), model_id


def test_invalid_study_and_unavailable_selector_are_rejected_without_results(runtime,tmp_path):
    cfg, _, predictor, _ = runtime
    with pytest.raises(InputError,match='intensity variation'):
        predictor.predict(
            {plane: np.ones((4, 32, 32), dtype=np.float32) for plane in predictor.artifact['planes']},
            'invalid',
        )
    unavailable = copy.deepcopy(cfg)
    unavailable['dynamic_selection']['artifact']=str(tmp_path/'missing.joblib')
    assert not dynamic_selection_status(unavailable)['available']
    with pytest.raises(DynamicSelectionUnavailable,match='unavailable'):
        DynamicSelectionPredictor(unavailable)


def test_incomplete_study_is_rejected_before_any_dynamic_model_runs(runtime):
    _, study, predictor, _ = runtime
    with pytest.raises(InputError, match='complete three-plane'):
        predictor.predict({'axial': study['axial']}, 'incomplete-dms-study')


def test_missing_checkpoint_is_rejected_before_partial_inference(runtime,tmp_path):
    cfg, _, _, _ = runtime
    artifact = load_selector_artifact(ARTIFACT,require_active=False)
    artifact['activation_status']='active'
    artifact['model_entries'][0]['checkpoint']=str(tmp_path/'missing.pth')
    path=tmp_path/'selector.joblib'
    save_selector_artifact(path,artifact)
    cfg=copy.deepcopy(cfg)
    cfg['dynamic_selection']['artifact']=str(path)
    assert not dynamic_selection_status(cfg)['available']
    with pytest.raises(DynamicSelectionUnavailable,match='checkpoint is unavailable'):
        DynamicSelectionPredictor(cfg)
