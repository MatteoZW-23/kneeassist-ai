"""Create a reproducible, non-destructive experiment; never regenerate its split."""
import json
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from src.utils import ROOT,config,save_json,fingerprint
from src.data.dataset import read_manifest

def main():
    run=ROOT/'runs/mri_finetune_02'; run.mkdir(parents=True,exist_ok=True)
    cfg_path=ROOT/'configs/mri_finetune_02.yaml'
    if cfg_path.exists():return
    cfg=config(ROOT/'configs/mrnet_fresh_01.yaml')
    rows=read_manifest(cfg);lookup={r['study_id']:r for r in rows}
    split=json.loads((ROOT/'data/splits.json').read_text())
    # Binary ACL stratification guarantees injury references in the reserved group.
    fit,cal=train_test_split(split['fit'],test_size=100,random_state=20260926,
                              stratify=[lookup[k]['acl'] for k in split['fit']])
    split['fit']=fit;split['calibration']=cal
    for role,ids in split.items():
        for target in cfg['targets']:
            if len({lookup[k][target['key']] for k in ids})!=2:
                raise ValueError(f'{role} lacks both classes for {target["key"]}')
    save_json(run/'splits.json',split)
    cfg['dataset']['splits']='runs/mri_finetune_02/splits.json'
    cfg['run_directory']='runs/mri_finetune_02'
    cfg['model'].update(training_mode='staged_final_blocks',encoder_chunk_size=4)
    cfg['training'].update(epochs=6,warmup_epochs=40,early_stopping_patience=3,
                           batch_size=1,gradient_accumulation=4,learning_rate=.00003,
                           encoder_learning_rate=.00001)
    cfg['evaluation'].update(checkpoint='runs/mri_finetune_02/models/best_model.pth',
                              directory='runs/mri_finetune_02/results',
                              threshold_selection='reserved_calibration_youden')
    cfg['paths']['models']='runs/mri_finetune_02/models'
    # Frozen ImageNet features are immutable and can be reused; no trainable features cached.
    cfg['paths']['features']='runs/mrnet_fresh_01/features'
    cfg['lifecycle'].update(state='improvement_experiment',training_enabled=True,inference_enabled=False)
    cfg['external_fastmri']['label_mapping_verified']=False
    cfg['external_fastmri']['intended_use']='Exploratory meniscus evaluation only; non-exhaustive reference annotations'
    cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    save_json(run/'protocol.json',{'version':2,'config_sha256':fingerprint(cfg),'split_sha256':fingerprint(split),
        'counts':{k:len(v) for k,v in split.items()},'training_data':'MRNet only',
        'selection':'Internal tuning macro AUROC; incumbent included; no external metric-based selection',
        'calibration':'100 cases removed from old fitting partition; candidate-only calibration after selection',
        'operating_point':'Maximum Youden J on reserved calibration cohort; not a clinical operating point',
        'budget':'Two architectures, one seed, up to 6 fine-tuning epochs each, patience 3',
        'limitations':['Study independence only; patient linkage unavailable',
         'Internal tuning and official validation previously evaluated',
         'Incumbent previously trained on calibration subset: never calibrate incumbent on it',
         'New candidate uses 804 fitting cases vs incumbent 904; comparisons are development experiments',
         'KneeMRI previously evaluated; fastMRI reference labels non-exhaustive',
         'No multi-seed claim or clinical readiness claim']})

if __name__=='__main__':main()
