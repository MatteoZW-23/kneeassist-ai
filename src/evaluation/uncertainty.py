"""Study bootstrap intervals, conditional on this fitted model and cohort."""
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_auroc(labels, probabilities, targets, repetitions=1000, seed=20260926):
    rng=np.random.default_rng(seed);values={t['key']:[] for t in targets};macros=[]
    for _ in range(repetitions):
        indices=rng.integers(0,len(labels),len(labels));auc=[]
        for j,target in enumerate(targets):
            y=labels[indices,j]
            if len(np.unique(y))!=2:continue
            score=float(roc_auc_score(y,probabilities[indices,j]))
            values[target['key']].append(score);auc.append(score)
        if len(auc)==len(targets):macros.append(float(np.mean(auc)))
    values['macro']=macros
    return {'method':'study bootstrap percentile intervals','repetitions':repetitions,'seed':seed,
            'auroc_95_intervals':{key:{'low':float(np.percentile(v,2.5)),
                                      'high':float(np.percentile(v,97.5)),'valid_resamples':len(v)}
                                  for key,v in values.items() if v},
            'limitation':'Conditional sampling uncertainty only; patient clustering, selection bias and training-seed variability are not represented.'}
