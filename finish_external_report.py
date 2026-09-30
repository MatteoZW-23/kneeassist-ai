from pathlib import Path
import json, numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
out=Path('results/external_kneemri');r=json.loads((out/'metrics.json').read_text());df=pd.read_csv(out/'exam_predictions.csv');rng=np.random.default_rng(42);scores=[]
y=(df.severity.to_numpy()>0).astype(int);p=df.probability.to_numpy()
for _ in range(2000):
 i=rng.integers(0,len(y),len(y))
 if len(np.unique(y[i]))==2:scores.append(roc_auc_score(y[i],p[i]))
ci=np.quantile(scores,[.025,.975]).tolist();r['exam_bootstrap_auroc_95_ci']=ci;r['ci_note']='2000 exam-level bootstrap resamples; not patient-clustered because patient linkage is unavailable';(out/'metrics.json').write_text(json.dumps(r,indent=2))
m=r['primary']['acl'];tn,fp=m['confusion_matrix'][0];fn,tp=m['confusion_matrix'][1]
text=f'''# KneeAssist AI: external KneeMRI evaluation

917 MRI volumes from Clinical Hospital Centre Rijeka, grouped into {r['exams']} exam IDs. Model and threshold were frozen before inference. No retraining or threshold tuning used this cohort.

| Metric | ACL injury (partial or complete) |
|---|---:|
| AUROC | {m['auroc']:.3f} |
| Approximate exam-bootstrap 95% CI | {ci[0]:.3f}–{ci[1]:.3f} |
| F1 | {m['f1']:.3f} |
| Sensitivity | {m['sensitivity']:.3f} |
| Specificity | {m['specificity']:.3f} |
| Precision | {m['precision']:.3f} |
| Accuracy | {m['accuracy']:.3f} |

Correctly negative: {tn}; false positives: {fp}; missed injuries: {fn}; correctly positive: {tp}. The operating threshold is {m['threshold']:.2f}, retained from MRNet internal validation.

## Interpretation and limits
The model receives full sagittal volumes with missing axial and coronal planes. These results combine institution, acquisition and missing-plane differences. Repeated series are averaged per exam and maximum injury severity used as the reference. Exam IDs do not establish patient independence. Confidence intervals therefore resample exams, not verified patient clusters. No claim of clinical readiness follows from these results.

Only ACL has reference labels. Healthy ACL is not equivalent to a normal knee. General abnormality and meniscus outputs must not be evaluated against these labels. Secondary complete-rupture versus healthy results (partial injuries excluded) are in metrics.json. No training or model selection should reuse this entire cohort while still describing it as untouched external evaluation.

Source: https://zenodo.org/records/14789903 . CC BY-NC-ND 4.0. Original local files were not altered. Notebook 05 shows the evaluation outputs; sample_cases/kneemri_external_example.zip is a local research example only.
'''
(out/'REPORT.md').write_text(text,encoding='utf-8');print(text)
import nbformat
from nbclient import NotebookClient
path=Path('notebooks/05_External_KneeMRI_Evaluation.ipynb');n=nbformat.read(path,4);NotebookClient(n,timeout=180,kernel_name='kneeassist-ai',resources={'metadata':{'path':str(Path.cwd())}}).execute();nbformat.write(n,path)
print('External evaluation notebook executed')
