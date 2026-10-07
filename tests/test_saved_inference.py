import numpy as np
import csv
import pytest
from src.utils import ROOT,config
from src.inference.predictor import Predictor
from src.data.preprocessing import load_uploads

RELEASE_FIXTURES = (
    ROOT / config()['evaluation']['checkpoint'],
    ROOT / 'sample_cases/mrnet_1130.zip',
    ROOT / config()['evaluation']['directory'] / 'final/predictions.csv',
)


@pytest.mark.skipif(not all(path.exists() for path in RELEASE_FIXTURES), reason='Local validation MRI fixture and recorded predictions required')
def test_saved_model_inference_attention_and_cpu_fallback():
    cfg=config();raw=(ROOT/'sample_cases/mrnet_1130.zip').read_bytes()
    volumes=load_uploads([('study.zip',raw)],cfg['preprocessing'])
    predictor=Predictor(cfg=cfg);first=predictor.predict(volumes,'test')
    second=predictor.predict(volumes,'test')
    a=np.array([f['probability'] for f in first['findings']]);b=np.array([f['probability'] for f in second['findings']])
    assert np.allclose(a,b,atol=1e-6) and ((a>=0)&(a<=1)).all()
    with (ROOT/config()['evaluation']['directory']/'final/predictions.csv').open() as f:
        recorded=next(row for row in csv.DictReader(f) if row['study_id']=='1130')
    expected=np.array([float(recorded[t['key']+'_probability']) for t in cfg['targets']])
    assert np.allclose(a,expected,atol=1e-5)
    attention=predictor.explain(volumes,'acl','sagittal')
    assert np.isfinite(attention['heatmaps']).all() and np.isfinite(attention['overlay']).all()
    assert len(attention['indices'])==len(attention['heatmaps'])
    assert len(attention['feature_map_size'])==2 and all(value > 0 for value in attention['feature_map_size'])
    assert 'not confidence' in first['confidence_note'].lower()
    assert 'do not localise' in first['attention_note'].lower()
    assert 'calibrated measure of diagnostic uncertainty' in first['uncertainty_note']
    missing=predictor.predict({'sagittal':volumes['sagittal']})
    assert len(missing['missing_planes'])==2 and missing['incomplete_study_warning']
    cfg['training']['device']='cpu';cpu=Predictor(cfg=cfg).predict(volumes,'test')
    c=np.array([f['probability'] for f in cpu['findings']])
    assert np.allclose(a,c,atol=.01)
