import numpy as np
from sklearn.metrics import (roc_auc_score,average_precision_score,f1_score,
    precision_score,recall_score,accuracy_score,confusion_matrix)

def calculate(y,p,targets,thresholds):
    result={}
    for j,target in enumerate(targets):
        t=target['key']; pred=p[:,j]>=thresholds[t]
        tn,fp,fn,tp=confusion_matrix(y[:,j],pred,labels=[0,1]).ravel()
        both=len(np.unique(y[:,j]))==2
        result[t]={'auroc':float(roc_auc_score(y[:,j],p[:,j])) if both else None,
          'average_precision':float(average_precision_score(y[:,j],p[:,j])) if y[:,j].sum() else None,
          'f1':float(f1_score(y[:,j],pred,zero_division=0)),
          'sensitivity':float(recall_score(y[:,j],pred,zero_division=0)),
          'specificity':float(tn/(tn+fp)) if tn+fp else None,
          'precision':float(precision_score(y[:,j],pred,zero_division=0)),
          'accuracy':float(accuracy_score(y[:,j],pred)),
          'threshold':float(thresholds[t]),'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]],
          'positive_cases':int(y[:,j].sum()),'negative_cases':int(len(y)-y[:,j].sum())}
    fields=['auroc','average_precision','f1','sensitivity','specificity','precision','accuracy']
    result['macro']={k:float(np.mean([result[t['key']][k] for t in targets if result[t['key']][k] is not None]))
                     if any(result[t['key']][k] is not None for t in targets) else None for k in fields}
    return result

def choose_thresholds(y,p,targets):
    result={}
    for j,t in enumerate(targets):
        candidates=np.linspace(.05,.95,91)
        scores=[f1_score(y[:,j],p[:,j]>=v,zero_division=0) for v in candidates]
        best=max(scores)
        tied=[v for v,s in zip(candidates,scores) if s==best]
        result[t['key']]=float(min(tied,key=lambda v:abs(v-.5)))
    return result
