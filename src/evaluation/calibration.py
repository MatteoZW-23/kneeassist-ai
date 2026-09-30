"""Positive-slope Platt scaling, fitted exclusively on reserved calibration cases."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_curve, brier_score_loss, log_loss


def fit_calibration(labels, logits, targets):
    params=[]
    for j,target in enumerate(targets):
        y=np.asarray(labels[:,j],dtype=float); x=np.asarray(logits[:,j],dtype=float)
        if len(np.unique(y))!=2:
            raise ValueError(f'Calibration requires both classes for {target["key"]}.')
        def objective(theta):
            z=np.exp(theta[0])*x+theta[1]
            return np.mean(np.logaddexp(0,z)-y*z)+.01*np.sum(theta**2)
        result=minimize(objective,[0.,0.],method='L-BFGS-B',bounds=[(-3,3),(-8,8)])
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('Probability calibration failed to converge.')
        params.append({'scale':float(np.exp(result.x[0])),'bias':float(result.x[1])})
    return {'method':'positive_slope_platt','source':'reserved_MRNet_calibration',
            'cases':int(len(labels)),'parameters':params,
            'limitation':'Small development calibration cohort; external clinical calibration is not established.'}


def apply_calibration(logits, calibration=None):
    logits=np.asarray(logits,dtype=float)
    if calibration:
        logits=logits*np.array([p['scale'] for p in calibration['parameters']])+np.array([p['bias'] for p in calibration['parameters']])
    return expit(logits)


def balanced_thresholds(labels, probabilities, targets):
    """Prespecified maximum Youden J; ties prefer the threshold nearest 0.5."""
    thresholds={}
    for j,target in enumerate(targets):
        fpr,tpr,cutoffs=roc_curve(labels[:,j],probabilities[:,j],drop_intermediate=False)
        good=np.isfinite(cutoffs)&(cutoffs>=0)&(cutoffs<=1)
        cutoffs=cutoffs[good]; scores=(tpr-fpr)[good]
        tied=cutoffs[np.isclose(scores,scores.max(),rtol=0,atol=1e-12)]
        thresholds[target['key']]=float(tied[np.argmin(abs(tied-.5))])
    return thresholds


def reliability(labels, probabilities, targets):
    return {t['key']:{'brier':float(brier_score_loss(labels[:,j],probabilities[:,j])),
                      'log_loss':float(log_loss(labels[:,j],probabilities[:,j],labels=[0,1]))}
            for j,t in enumerate(targets)}
