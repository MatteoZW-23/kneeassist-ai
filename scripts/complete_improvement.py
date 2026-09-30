"""One locked, resumable improvement pipeline. Keeps the deployed model until checks pass."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import msvcrt
import numpy as np
import torch
import yaml
from scipy.special import expit
from src.utils import ROOT,config,resolve,save_json,get_device,seed_everything,fingerprint
from src.data.dataset import read_manifest,split_and_check,volumes
from src.training.finetune import fit,infer_rows
from src.training.train import checkpoint
from src.models.study_model import StudyModel
from src.evaluation.calibration import fit_calibration,apply_calibration,balanced_thresholds,reliability
from src.evaluation.metrics import calculate

RUN=ROOT/'runs/mri_finetune_02'

def status(stage,**details):
    save_json(RUN/'completion_status.json',{'stage':stage,'pid':os.getpid(),
              'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**details})
    print(stage,details,flush=True)

def command(args,log):
    with (RUN/log).open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    cfg=config(ROOT/'configs/mri_finetune_02.yaml')
    if (RUN/'verification.json').exists():
        verification=json.loads((RUN/'verification.json').read_text())
        status('awaiting_browser_verification' if verification.get('browser_verification_pending') else 'ready_for_research_testing',message='Preserving completed artifacts.');return
    protocol=json.loads((RUN/'protocol.json').read_text())
    if fingerprint(cfg)!=protocol['config_sha256']:raise ValueError('Locked experiment configuration changed.')
    seed_everything(cfg['seed'],cfg['training']['cpu_threads']);device=get_device(cfg)
    status('checking_splits',device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None)
    rows=read_manifest(cfg);split,_=split_and_check(rows,cfg)
    if fingerprint(split)!=protocol['split_sha256']:raise ValueError('Locked experiment split changed.')
    reports=[]
    for architecture in cfg['model']['architectures']:
        status('training',architecture=architecture)
        reports.append(fit(architecture,cfg,rows,split,device))
    incumbent=json.loads((ROOT/'runs/mrnet_fresh_01/results/efficientnet_b0/internal_metrics.json').read_text())
    incumbent['candidate']='incumbent_frozen_baseline'
    winner=max(reports,key=lambda r:r['metrics']['macro']['auroc'])
    improved=winner['metrics']['macro']['auroc']>incumbent['metrics']['macro']['auroc']+1e-6
    status('selecting_and_calibrating',candidate_improved=improved)
    if improved:
        saved=torch.load(resolve(cfg['paths']['models'])/f"{winner['architecture']}_best.pth",map_location=device,weights_only=False)
        model=StudyModel(saved['architecture'],cfg,pretrained=False).to(device);model.load_state_dict(saved['model'])
        lookup={r['study_id']:r for r in rows};cal_rows=[lookup[k] for k in split['calibration']]
        labels=np.array([[int(r[t['key']]) for t in cfg['targets']] for r in cal_rows])
        logits=infer_rows(model,cal_rows,cfg,device)
        calibration=fit_calibration(labels,logits,cfg['targets']); probabilities=apply_calibration(logits,calibration)
        saved['calibration']=calibration;saved['thresholds']=balanced_thresholds(labels,probabilities,cfg['targets'])
        save_json(RUN/'results/calibration.json',{'calibration':calibration,'thresholds':saved['thresholds'],
                 'before':reliability(labels,expit(logits),cfg['targets']),
                 'after':reliability(labels,probabilities,cfg['targets']),
                 'warning':'In-sample calibration diagnostics; not held-out reliability estimates.'})
        del model;torch.cuda.empty_cache()
    else:
        winner=incumbent
        saved=torch.load(ROOT/'runs/mrnet_fresh_01/models/best_model.pth',map_location='cpu',weights_only=False)
    checkpoint(resolve(cfg['evaluation']['checkpoint']),saved)
    save_json(RUN/'results/model_comparison.json',{'criterion':'Internal tuning macro AUROC only',
              'winner':winner['architecture'],'candidate_improved':improved,'models':reports+[incumbent],
              'comparability_limits':protocol['limitations']})
    from src.evaluation.evaluate import evaluate
    status('evaluating_official_validation');evaluate(cfg,rows,split)
    status('evaluating_external_references')
    for name in ['kneemri','fastmri']:
        destination=RUN/'results'/f'external_{name}'
        if not (destination/'metrics.json').exists():
            if improved:
                command(['-m',f'src.evaluation.external_{name}','--config','configs/mri_finetune_02.yaml',
                         '--output',str(destination)],f'external_{name}.log')
            else:
                shutil.copytree(ROOT/f'runs/mrnet_fresh_01/results/external_{name}',destination,dirs_exist_ok=True)
    status('verifying_candidate_inference')
    from src.inference.predictor import Predictor,text_summary
    predictor=Predictor(cfg=cfg);example=next(r for r in rows if r['study_id']==split['official_valid'][0])
    case=volumes(example,cfg);prediction=predictor.predict(case,'Automated verification')
    attention=predictor.explain(case,'acl','sagittal')
    if not np.isfinite(attention['heatmaps']).all():raise RuntimeError('Invalid attention output.')
    (RUN/'example_summary.txt').write_text(text_summary(prediction),encoding='utf-8')
    save_json(RUN/'example_prediction.json',prediction)
    del predictor;torch.cuda.empty_cache()
    # Retain the original deployment config until inference/evaluation succeed.
    backup=RUN/'previous_deployment.yaml'
    if not backup.exists():shutil.copy2(ROOT/'config.yaml',backup)
    deployed=json.loads(json.dumps(cfg));deployed['lifecycle'].update(training_enabled=False,inference_enabled=True,state='research_testing_verification')
    deployed['external_kneemri']['results']=str(RUN/'results/external_kneemri')
    try:
        (ROOT/'config.yaml').write_text(yaml.safe_dump(deployed,sort_keys=False))
        status('testing_application');command(['-m','pytest','tests','-q'],'test_results.txt')
        status('executing_notebooks');command(['execute_notebooks.py'],'notebook_execution.log')
    except Exception:
        shutil.copy2(backup,ROOT/'config.yaml');raise
    deployed['lifecycle']['state']='ready_for_research_testing'
    (ROOT/'config.yaml').write_text(yaml.safe_dump(deployed,sort_keys=False))
    summary={'candidate_improved_internal_auroc':improved,'selected_architecture':saved['architecture'],
             'selected_stage':saved.get('training_stage','frozen_encoder'),
             'mrnet':json.loads((RUN/'results/final/metrics.json').read_text()),
             'kneemri':json.loads((RUN/'results/external_kneemri/metrics.json').read_text()),
             'fastmri':json.loads((RUN/'results/external_fastmri/metrics.json').read_text())}
    save_json(RUN/'validation_summary.json',summary)
    report=['# Improvement experiment readiness','',f"Internal-validation improvement: {improved}.",
            f"Selected: {saved['architecture']} / {saved.get('training_stage','frozen encoder baseline')}.",
            '', 'See validation_summary.json for all measured results and test_results.txt for software checks.',
            'External results did not determine model selection or hyperparameters.',
            'MRNet-only training. KneeMRI previously evaluated; fastMRI labels are a limited annotation reference.',
            'One seed; no proven patient linkage. Research testing only; no clinical readiness claim.',
            'Browser upload, prediction, attention, export and clear checks must be verified after the deployment reload.']
    (RUN/'READINESS_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    save_json(RUN/'verification.json',{'software_pipeline_passed':True,'browser_verification_pending':True,
               'clinical_ready':False,'selected_checkpoint':cfg['evaluation']['checkpoint']})
    status('awaiting_browser_verification',candidate_improved=improved)

if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'pipeline.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Another improvement pipeline is already running.')
        try:main()
        except Exception as error:
            status('failed',error=str(error));traceback.print_exc();raise
