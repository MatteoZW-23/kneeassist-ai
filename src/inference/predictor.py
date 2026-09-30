import argparse
import hashlib
import json
import threading
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from src.utils import config,resolve,get_device,seed_everything
from src.data.preprocessing import prepare,load_uploads,InputError
from src.models.study_model import StudyModel

class Predictor:
    def __init__(self,path=None,cfg=None):
        self.cfg=cfg or config();self.device=get_device(self.cfg)
        seed_everything(self.cfg['seed'],self.cfg['training']['cpu_threads'])
        path=resolve(path or self.cfg['evaluation']['checkpoint'])
        if not path.exists():raise InputError('The trained model is not available yet. Finish training before analysing cases.')
        # Only application-owned local checkpoints are loaded, never uploaded models.
        saved=torch.load(path,map_location=self.device,weights_only=False)
        self.model_cfg=saved['config'];self.architecture=saved['architecture'];self.thresholds=saved['thresholds']
        self.calibration=saved.get('calibration')
        self.training_stage=saved.get('training_stage','frozen_encoder')
        self.checkpoint_name=path.name
        with path.open('rb') as stream:self.checkpoint_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
        self.model=StudyModel(self.architecture,self.model_cfg,pretrained=False).to(self.device)
        # Handle potential dimension mismatches with new aggregation methods
        try:
            self.model.load_state_dict(saved['model'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                # Fallback to original aggregation method for compatibility
                print(f"Warning: Checkpoint dimension mismatch. Using original aggregation method.")
                self.model.aggregation_method = 'mean_max'
                # Rebuild model with original dimensions
                dim = self.model.plane_dim * len(self.model.planes)
                self.model.register_buffer('feature_mean', torch.zeros(dim))
                self.model.register_buffer('feature_std', torch.ones(dim))
                self.model.head = nn.Sequential(
                    nn.Linear(dim + len(self.model.planes), self.model_cfg['model']['hidden_dim']),
                    nn.ReLU(),
                    nn.Dropout(self.model_cfg['model']['dropout']),
                    nn.Linear(self.model_cfg['model']['hidden_dim'], len(self.model_cfg['targets']))
                )
                self.model.load_state_dict(saved['model'])
            else:
                raise
        self.model.eval();self.lock=threading.RLock()
        self.amp=self.model_cfg['training']['mixed_precision'] and self.device.type=='cuda'
        # Fine-tuned calibration/evaluation logits use FP32 consistently.
        if self.model_cfg['model'].get('training_mode')=='staged_final_blocks':self.amp=False

    def tensors(self,volumes):
        if not volumes:raise InputError('Upload at least one MRI sequence.')
        expected=self.model_cfg['preprocessing']['planes']
        if any(p not in expected for p in volumes):raise InputError('An unknown imaging plane was supplied.')
        prepared={p:prepare(v,self.model_cfg['preprocessing']) for p,v in volumes.items()}
        return {p:values[0].to(self.device) for p,values in prepared.items()},prepared

    def predict(self,volumes,case_reference='Research case',uncertainty_samples=1):
        with self.lock:
            tensors,prepared=self.tensors(volumes)
            
            # Inference does not need an autograd graph. MC dropout still works
            # under no_grad because only the dropout masks, not gradients, vary.
            with torch.no_grad():
                if uncertainty_samples > 1 and self.model_cfg['model']['dropout'] > 0:
                    self.model.train()
                    try:
                        logits_samples = []
                        with torch.autocast(device_type=self.device.type,enabled=self.amp):
                            for _ in range(uncertainty_samples):
                                logits_samples.append(self.model(tensors).float()[0].cpu().numpy())
                        logits_mean = np.mean(logits_samples, axis=0)
                        logits_std = np.std(logits_samples, axis=0)
                    finally:
                        self.model.eval()
                else:
                    logits_mean = self.model(tensors).float()[0].cpu().numpy()
                    logits_std = np.zeros_like(logits_mean)
            
            from src.evaluation.calibration import apply_calibration
            scores=apply_calibration(logits_mean,self.calibration)
            
        if not np.isfinite(scores).all():raise InputError('The model returned an invalid result. Check the input study.')
        findings=[];distances=[]
        for t,p,std in zip(self.model_cfg['targets'],scores,logits_std):
            threshold=self.thresholds[t['key']];positive=bool(p>=threshold);distances.append(abs(float(p)-threshold))
            
            # Calculate uncertainty metrics
            uncertainty_level = 'Low' if std < 0.1 else 'Moderate' if std < 0.2 else 'High'
            
            findings.append({'key':t['key'],'finding':t['display'],'probability':float(p),
                             'threshold':threshold,'flagged':positive,
                             'uncertainty':float(std),'uncertainty_level':uncertainty_level,
                             'status':'Flagged for review' if positive else 'Below model threshold'})
        missing=[p for p in self.model.planes if p not in volumes]
        margin=min(distances)
        margin_label='High' if margin>=self.cfg['ui']['margin_high'] else 'Moderate' if margin>=self.cfg['ui']['margin_moderate'] else 'Low'
        
        # Add uncertainty information to result
        avg_uncertainty = np.mean([f['uncertainty'] for f in findings])
        uncertainty_summary = f"Average prediction uncertainty: {avg_uncertainty:.3f} ({'Low' if avg_uncertainty < 0.1 else 'Moderate' if avg_uncertainty < 0.2 else 'High'})"
        
        return {'system':'KneeAssist AI V1','case_reference':str(case_reference)[:120],
                'created_at':datetime.now(timezone.utc).isoformat(),'architecture':self.architecture,
                'training_stage':self.training_stage,'checkpoint_file':self.checkpoint_name,
                'checkpoint_sha256':self.checkpoint_sha256,
                'training_dataset':'MRNet-v1.0','findings':findings,'available_planes':list(volumes),
                'missing_planes':missing,'sampled_slices':{p:v[2] for p,v in prepared.items()},
                'model_confidence':'Not clinically established','score_separation':margin_label,
                'confidence_note':'Score separation is distance from decision thresholds, not a probability that the model is correct.',
                'probability_note':('Probabilities were calibrated on a small reserved MRNet development subset; clinical and external calibration are not established.' if self.calibration else 'Model probabilities are uncalibrated and are not validated estimates of clinical risk.'),
                'uncertainty_note':uncertainty_summary,
                'warning':'Decision-support output. Clinical review required. Below-threshold results do not rule out injury.',
                'attention_note':'Heatmaps show model attention, not confirmed lesions.',
                'incomplete_study_warning':'Missing sequence performance has not been separately validated.' if missing else None}

    def explain(self,volumes,target,plane):
        from src.evaluation.gradcam import explain
        with self.lock:return explain(self,volumes,target,plane)

def text_summary(result):
    lines=['KneeAssist AI V1',f"Case: {result['case_reference']}",f"Model: {result['architecture']}",'']
    if result.get('checkpoint_sha256'):
        lines+=['Checkpoint SHA256: '+result['checkpoint_sha256'],'Training stage: '+result.get('training_stage','unspecified'),'']
    for f in result['findings']:
        uncertainty_info = f" (±{f['uncertainty']:.3f}, {f['uncertainty_level']} uncertainty)" if 'uncertainty' in f else ""
        lines.append(f"{f['finding']}: {f['probability']:.1%} — {f['status']} (threshold {f['threshold']:.2f}){uncertainty_info}")
    lines+=['',f"Score separation: {result['score_separation']}",f"Model confidence: {result['model_confidence']}",
            result['confidence_note'],result['probability_note'],'']
    if 'uncertainty_note' in result:
        lines+=[result['uncertainty_note'],'']
    lines+=[result['warning'],result['attention_note']]
    if result['missing_planes']:lines+=['Missing sequences: '+', '.join(result['missing_planes']),result['incomplete_study_warning']]
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--case',default='Research case');p.add_argument('--json',action='store_true')
    args=p.parse_args();cfg=config()
    files=[(resolve(f).name,resolve(f).read_bytes()) for f in args.files]
    result=Predictor(cfg=cfg).predict(load_uploads(files,cfg['preprocessing']),args.case)
    print(json.dumps(result,indent=2) if args.json else text_summary(result))

if __name__=='__main__':main()
