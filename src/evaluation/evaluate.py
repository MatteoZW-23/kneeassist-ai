import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import ROOT,save_json,resolve
from src.inference.predictor import Predictor,text_summary
from src.data.dataset import volumes
from src.evaluation.metrics import calculate
from src.evaluation.plots import curves

def evaluate(cfg,rows,split):
    predictor=Predictor(cfg=cfg);directory=resolve(cfg['evaluation']['directory'])/'final';directory.mkdir(parents=True,exist_ok=True)
    lookup={r['study_id']:r for r in rows};selected=[lookup[k] for k in split['official_valid']]
    y=[];p=[];outputs=[]
    for i,r in enumerate(selected):
        result=predictor.predict(volumes(r,cfg),'MRNet-'+r['study_id'])
        outputs.append(result);y.append([int(r[t['key']]) for t in cfg['targets']]);p.append([f['probability'] for f in result['findings']])
        if (i+1)%20==0:print(f'Evaluation: {i+1}/{len(selected)}',flush=True)
    y=np.array(y);p=np.array(p)
    metrics=calculate(y,p,cfg['targets'],predictor.thresholds)
    from src.evaluation.uncertainty import bootstrap_auroc
    from src.evaluation.calibration import reliability
    save_json(directory/'uncertainty.json',bootstrap_auroc(y,p,cfg['targets']))
    save_json(directory/'probability_reliability.json',reliability(y,p,cfg['targets']))
    report={'dataset':'MRNet-v1.0','architecture':predictor.architecture,'partition':'official_valid',
            'exams':len(selected),'metrics':metrics,'threshold_source':('Reserved MRNet calibration partition (100 studies)' if predictor.calibration else 'Internal validation only'),
            'selection_source':'Internal validation macro AUROC',
            'limitation':'Previously used official validation set, not an independent external test. Study-level separation only; patient linkage unavailable.'}
    save_json(directory/'metrics.json',report);save_json(directory/'prediction_examples.json',outputs[:cfg['evaluation']['prediction_examples']])
    with (directory/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['study_id']+[t['key']+'_label' for t in cfg['targets']]+[t['key']+'_probability' for t in cfg['targets']])
        for r,a,b in zip(selected,y,p):w.writerow([r['study_id'],*a,*b])
    curves(y,p,metrics,cfg['targets'],directory)
    # Prespecified first cases, not cherry-picked for correctness.
    for r,result in zip(selected[:cfg['evaluation']['prediction_examples']],outputs):
        v=volumes(r,cfg);target=cfg['targets'][0]['key'];attention=predictor.explain(v,target,'sagittal');idx=attention['suggested_slice']
        fig,ax=plt.subplots(1,2,figsize=(8,4))
        ax[0].imshow(attention['original'][idx],cmap='gray');ax[0].set_title('Original MRI slice')
        ax[1].imshow(attention['overlay'][idx]);ax[1].set_title('Model attention — not a lesion label')
        for a in ax:a.axis('off')
        fig.suptitle(f"Study {r['study_id']} · {result['findings'][0]['finding']}: {result['findings'][0]['probability']:.1%}")
        fig.tight_layout();fig.savefig(directory/f"example_{r['study_id']}.png",dpi=150);plt.close(fig)
        (directory/f"example_{r['study_id']}.txt").write_text(text_summary(result),encoding='utf-8')
    comparison=json.loads((resolve(cfg['evaluation']['directory'])/'model_comparison.json').read_text())
    text=['# KneeAssist AI — evaluation','',f"Selected architecture: **{predictor.architecture}**. Selected checkpoint stage: {predictor.training_stage}. Experiment mode: {predictor.model_cfg['model']['training_mode']}.",'',
          'Selection used internal validation macro AUROC. Threshold procedure: '+predictor.model_cfg['evaluation']['threshold_selection']+'. Official validation was not used for either decision.','',
          '| Model | Internal validation macro AUROC | Selected epoch |','|---|---:|---:|']
    for entry in comparison['models']:text.append(f"| {entry['architecture']} | {entry['metrics']['macro']['auroc']:.3f} | {entry['selected_epoch']} |")
    text+=['','## Official validation (120 exams)','','| Finding | AUROC | F1 | Sensitivity | Specificity | Precision | Accuracy |','|---|---:|---:|---:|---:|---:|---:|']
    for t in [*cfg['targets'],{'key':'macro','display':'Macro average'}]:
        m=metrics[t['key']];text.append('| '+t['display']+' | '+' | '.join(f"{m[k]:.3f}" for k in ['auroc','f1','sensitivity','specificity','precision','accuracy'])+' |')
    text+=['','AUROC is a ranking metric, not percentage accuracy. The official validation cohort was used in the earlier prototype, so these are development-validation results, not a new external test. Patient independence is not verified without a patient mapping. Probabilities and score-separation indicators are not clinically calibrated. Attention maps are not confirmed lesions.','',
           'The current checkpoints support abnormality, ACL tear and meniscal tear only. RSNA and X-ray data were not used for these checkpoints. Research and clinical decision-support prototype; clinician review required.']
    (resolve(cfg['evaluation']['directory'])/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
