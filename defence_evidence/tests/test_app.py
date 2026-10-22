from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
from src.utils import ROOT,config

def button(at,label):
    return next(b for b in at.button if b.label==label)

def test_empty_and_invalid_upload_are_readable():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not at.exception
    button(at,'Analyse study').click().run()
    assert at.error and 'Upload' in at.error[0].value and not at.exception
    at.file_uploader[0].set_value([('axial.npy',b'not an MRI','application/octet-stream')]).run()
    button(at,'Analyse study').click().run()
    assert at.error and not at.exception


def test_model_evaluation_and_research_workflow_pages_render_from_saved_evidence():
    at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
    assert not at.exception
    navigation = next(item for item in at.radio if item.label == 'Workspace')
    navigation.set_value('Model evaluation and selection').run()
    assert not at.exception
    assert any('verified selector is active' in item.value for item in at.success)
    assert len(at.dataframe) >= 5
    navigation.set_value('Research workflow').run()
    assert not at.exception
    assert any('Workflow stages' in item.value for item in at.subheader)

def _routed_fixture_available():
    registry = json.loads((ROOT / 'model_registry' / 'active_models.json').read_text(encoding='utf-8'))
    checkpoints = [ROOT / item['checkpoint'] for item in registry['models'] if item.get('deployment_status') == 'selected']
    return (ROOT / 'sample_cases/mrnet_1130.zip').is_file() and len(checkpoints) == 3 and all(path.is_file() for path in checkpoints)


@pytest.mark.skipif(not _routed_fixture_available(), reason='Registered routed checkpoints and local MRI fixture required')
def test_real_upload_predictions_gradcam_and_clear():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
    raw=(ROOT/'sample_cases/mrnet_1130.zip').read_bytes()
    at.file_uploader[0].set_value([('mrnet_1130.zip',raw,'application/zip')]).run()
    at.text_input[0].set_value('UI-VALIDATION-1130').run()
    button(at,'Analyse study').click().run()
    assert not at.exception and not at.error
    result=at.session_state['result']
    assert len(result['findings'])==3 and result['case_reference']=='UI-VALIDATION-1130'
    assert result['dynamic_selection']['models_evaluated']==5
    assert len(result['dynamic_selection']['base_probabilities'])==5
    assert len(at.metric)==3 and len(at.download_button)==2
    # Attention generation is explicit so a failed visual explanation cannot
    # interrupt ordinary prediction review.
    button(at,'Generate attention map').click().run()
    assert not at.exception and not at.error
    assert at.session_state['attention_cache']
    attention=next(iter(at.session_state['attention_cache'].values()))
    assert attention['heatmaps'].ndim==3 and attention['original'].shape==attention['heatmaps'].shape
    assert attention['overlay'].shape[-1]==3
    assert len(attention['regions'])==len(attention['heatmaps'])
    assert any('This overlay explains the model' in item.value for item in at.info)
    assert any('classifier explanation only' in item.value for item in at.warning)
    # Changing a case reference must discard the previous predictions.
    at.text_input[0].set_value('DIFFERENT-CASE').run()
    assert 'result' not in at.session_state
    button(at,'Clear current case').click().run()
    assert not at.exception and len(at.metric)==0

    assert at.text_input[0].value == ''
    assert not at.file_uploader[0].value

    # A genuinely different uploaded study must trigger a new all-five-model
    # analysis rather than reuse the prior case's result.
    other=(ROOT/'sample_cases/mrnet_1229_abnormal_research_example.zip').read_bytes()
    at.file_uploader[0].set_value([('mrnet_1229.zip',other,'application/zip')]).run()
    at.text_input[0].set_value('UI-VALIDATION-1229').run()
    button(at,'Analyse study').click().run()
    updated=at.session_state['result']
    assert updated['case_reference']=='UI-VALIDATION-1229'
    assert updated['dynamic_selection']['models_evaluated']==5
    assert updated['dynamic_selection']['base_probabilities'] != result['dynamic_selection']['base_probabilities']
