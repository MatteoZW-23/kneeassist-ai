from pathlib import Path
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

@pytest.mark.skipif(
    not ((ROOT / config()['evaluation']['checkpoint']).exists() and (ROOT / 'sample_cases/mrnet_1130.zip').exists()),
    reason='Trained checkpoint and local validation MRI fixture required',
)
def test_real_upload_predictions_gradcam_and_clear():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
    raw=(ROOT/'sample_cases/mrnet_1130.zip').read_bytes()
    at.file_uploader[0].set_value([('mrnet_1130.zip',raw,'application/zip')]).run()
    at.text_input[0].set_value('UI-VALIDATION-1130').run()
    button(at,'Analyse study').click().run()
    assert not at.exception and not at.error
    result=at.session_state['result']
    assert len(result['findings'])==3 and result['case_reference']=='UI-VALIDATION-1130'
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
