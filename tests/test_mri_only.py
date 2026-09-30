import pytest
from streamlit.testing.v1 import AppTest
from src.utils import ROOT,config
from src.data.preprocessing import load_uploads,InputError

def test_picture_uploads_are_rejected():
    for name in ['scan.jpg','scan.jpeg','scan.png']:
        with pytest.raises(InputError,match='JPEG/PNG'):
            load_uploads([(name,b'picture')],config()['preprocessing'])

def test_dashboard_offers_mri_only():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception
    assert not app.radio
    assert len(app.file_uploader)==1
    assert app.file_uploader[0].label=='MRI study files'
    assert not any(button.label=='Clear picture' for button in app.button)
