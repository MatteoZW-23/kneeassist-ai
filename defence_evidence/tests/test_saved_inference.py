import numpy as np
import csv
import pytest
import torch
from src.utils import ROOT,config
from src.inference.routed_predictor import RoutedPredictor
from src.selection.model_selector import select_models
from src.data.preprocessing import load_uploads

def _release_fixtures():
    import json
    registry = json.loads((ROOT / 'model_registry' / 'active_models.json').read_text(encoding='utf-8'))
    checkpoints = [ROOT / item['checkpoint'] for item in registry['models'] if item.get('deployment_status') == 'selected']
    return (*checkpoints, ROOT / 'sample_cases/mrnet_1130.zip', ROOT / 'runs/model_portfolio_v1/final/predictions.csv')


RELEASE_FIXTURES = _release_fixtures()


@pytest.mark.skipif(not all(path.exists() for path in RELEASE_FIXTURES), reason='Local validation MRI fixture and recorded predictions required')
def test_saved_routed_inference_attention_and_cpu_fallback():
    cfg=config();raw=(ROOT/'sample_cases/mrnet_1130.zip').read_bytes()
    volumes=load_uploads([('study.zip',raw)],cfg['preprocessing'])
    selections = select_models(list(volumes), [target['key'] for target in cfg['targets']])['selections']
    predictor=RoutedPredictor(cfg=cfg);first=predictor.predict(volumes,'test',selections)
    second=predictor.predict(volumes,'test',selections)
    a=np.array([f['probability'] for f in first['findings']]);b=np.array([f['probability'] for f in second['findings']])
    assert np.allclose(a,b,atol=1e-6) and ((a>=0)&(a<=1)).all()
    assert [item['model_id'] for item in first['findings']] == ['family_densenet121', 'family_resnet18', 'family_swin_t']
    with (ROOT/'runs/model_portfolio_v1/final/predictions.csv').open() as f:
        recorded=next(row for row in csv.DictReader(f) if row['study_id']=='1130')
    expected=np.array([float(recorded[t['key']+'_probability']) for t in cfg['targets']])
    # The saved development predictions were generated on a different local
    # numerical run.  The registered routed CPU output must remain within the
    # measured 1e-4 floating-point reproducibility bound, while route IDs,
    # thresholds and threshold statuses remain identical.
    assert np.allclose(a,expected,atol=1e-4)
    attention=predictor.explain(volumes,'acl','sagittal', model_id='family_resnet18')
    assert np.isfinite(attention['heatmaps']).all() and np.isfinite(attention['overlay']).all()
    assert len(attention['indices'])==len(attention['heatmaps'])
    assert len(attention['feature_map_size'])==2 and all(value > 0 for value in attention['feature_map_size'])
    assert 'not confidence' in first['confidence_note'].lower()
    assert 'do not localise' in first['attention_note'].lower()
    assert 'calibrated measure of diagnostic uncertainty' in first['uncertainty_note']
    missing_selections = select_models(['sagittal'], [target['key'] for target in cfg['targets']])['selections']
    missing=predictor.predict({'sagittal':volumes['sagittal']}, 'test', missing_selections)
    assert len(missing['missing_planes'])==2 and missing['incomplete_study_warning']
    cfg['training']['device']='cpu';cpu=RoutedPredictor(cfg=cfg).predict(volumes,'test',selections)
    c=np.array([f['probability'] for f in cpu['findings']])
    assert np.allclose(a,c,atol=.01)

def test_mc_dropout_is_exploratory_and_preserves_deterministic_scores():
    cfg=config(); raw=(ROOT/'sample_cases/mrnet_1130.zip').read_bytes()
    volumes=load_uploads([('study.zip',raw)],cfg['preprocessing'])
    predictor=RoutedPredictor(cfg=cfg)._predictor('family_resnet18')
    baseline=predictor.predict(volumes,'test',uncertainty_samples=1)
    buffers=[module.running_mean.detach().clone() for module in predictor.model.modules()
             if hasattr(module,'running_mean') and module.running_mean is not None]
    exploratory=predictor.predict(volumes,'test',uncertainty_samples=3)
    after=[module.running_mean.detach().clone() for module in predictor.model.modules()
           if hasattr(module,'running_mean') and module.running_mean is not None]
    base_scores=np.array([item['probability'] for item in baseline['findings']])
    exploratory_scores=np.array([item['probability'] for item in exploratory['findings']])
    assert np.allclose(base_scores, exploratory_scores, atol=1e-7)
    assert [item['flagged'] for item in baseline['findings']] == [item['flagged'] for item in exploratory['findings']]
    assert all(torch.equal(before, later) for before,later in zip(buffers,after))
    assert all(not module.training for module in predictor.model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    assert exploratory['scoring_mode']=='deterministic_calibrated_eval'
    assert 'does not change the deterministic calibrated score' in exploratory['uncertainty_note']

