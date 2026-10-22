"""Build measured DMS comparisons and figures from saved final predictions.

This performs no fitting, threshold selection or model promotion. Final cohort
outcomes remain evaluation data only.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import ROOT,save_json


LABELS={'abnormal':'General abnormality','acl':'ACL tear','meniscus':'Meniscal tear'}
NAMES={'dms_resnet18':'ResNet-18','dms_resnet50':'ResNet-50','dms_densenet121':'DenseNet-121',
       'dms_efficientnet_b0':'EfficientNet-B0','dms_swin_t':'Swin-T','dynamic_selection':'Dynamic selection',
       'fixed_validation_selected':'Fixed validation-selected model',
       'mean_probability_ensemble':'Mean-probability ensemble',
       'oracle_observed_brier':'Oracle (analysis only)'}


def calibration_bins(truth,scores):
    groups=[]
    for lower,upper in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(scores>=lower)&((scores<upper) if upper<1 else (scores<=upper))
        if mask.any():
            groups.append({'count':int(mask.sum()),'mean_score':float(scores[mask].mean()),
                           'positive_fraction':float(truth[mask].mean())})
    ece=sum(item['count']*abs(item['mean_score']-item['positive_fraction']) for item in groups)/len(truth)
    return groups,float(ece)


def report_metrics(truth, scores, thresholds):
    """Return study-level metrics for a fixed, already defined method.

    Thresholds are supplied rather than fitted here.  This prevents the final
    cohort labels from being used to tune a comparator's classification cut-off.
    """
    report={}
    for index,target in enumerate(LABELS):
        y=truth[:,index]
        p=scores[:,index]
        predicted=(p>=thresholds[:,index]).astype(int)
        tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
        _,ece=calibration_bins(y,p)
        report[target]={
            'auroc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'accuracy':float(accuracy_score(y,predicted)),
            'sensitivity':float(recall_score(y,predicted,zero_division=0)),
            'specificity':float(tn/(tn+fp)) if tn+fp else None,
            'precision':float(precision_score(y,predicted,zero_division=0)),
            'f1':float(f1_score(y,predicted,zero_division=0)),
            'brier':float(np.mean((p-y)**2)),
            'ece_10_equal_width_bins':float(ece),
            'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]],
        }
    report['macro']={
        metric:float(np.mean([report[target][metric] for target in LABELS if report[target][metric] is not None]))
        for metric in ('auroc','average_precision','accuracy','sensitivity','specificity','precision','f1','brier','ece_10_equal_width_bins')
    }
    return report


def paired_bootstrap_summary(truth, reference, methods, repeats=2000, seed=42):
    """Paired, whole-study bootstrap summaries for the exploratory cohort."""
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(truth),size=(repeats,len(truth)))
    output={}
    reference_brier=np.mean((reference-truth)**2,axis=1)
    for method_id,scores in methods.items():
        macro_auroc=[]
        macro_ap=[]
        macro_brier=[]
        brier_difference=[]
        auroc_difference=[]
        for sample in indices:
            y=truth[sample]
            p=scores[sample]
            # All three labels contain both classes in almost every resample.
            # Skip the exceptional sample rather than manufacture an AUROC.
            if any(np.unique(y[:,index]).size<2 for index in range(y.shape[1])):
                continue
            sample_reference_auroc=float(np.mean([roc_auc_score(y[:,index],reference[sample,index]) for index in range(y.shape[1])]))
            sample_auroc=float(np.mean([roc_auc_score(y[:,index],p[:,index]) for index in range(y.shape[1])]))
            macro_auroc.append(sample_auroc)
            macro_ap.append(float(np.mean([average_precision_score(y[:,index],p[:,index]) for index in range(y.shape[1])])) )
            macro_brier.append(float(np.mean((p-y)**2)))
            brier_difference.append(float(np.mean(np.mean((p-y)**2,axis=1)-reference_brier[sample])))
            auroc_difference.append(sample_auroc-sample_reference_auroc)
        output[method_id]={
            'macro_auroc_95_percent_interval':np.percentile(macro_auroc,[2.5,97.5]).tolist(),
            'macro_average_precision_95_percent_interval':np.percentile(macro_ap,[2.5,97.5]).tolist(),
            'macro_brier_95_percent_interval':np.percentile(macro_brier,[2.5,97.5]).tolist(),
            'mean_brier_minus_fixed_95_percent_interval':np.percentile(brier_difference,[2.5,97.5]).tolist(),
            'macro_auroc_minus_fixed_95_percent_interval':np.percentile(auroc_difference,[2.5,97.5]).tolist(),
            'paired_reference':'fixed_validation_selected',
            'interpretation':'Whole-study paired bootstrap on the recorded development cohort. It describes sampling variability here; it is not independent or external validation.',
        }
    return output


def main():
    folder=ROOT/'runs/dynamic_selection_v1'
    evaluation=json.loads((folder/'final_evaluation.json').read_text(encoding='utf-8'))
    with (folder/'final_predictions.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    targets=list(LABELS)
    truth=np.array([[int(row[target]) for target in targets] for row in rows])
    model_ids=list(evaluation['base_models'])
    predictions={model_id:np.array([[float(row[f'{model_id}::{target}']) for target in targets] for row in rows]) for model_id in model_ids}
    predictions['dynamic_selection']=np.array([[float(row[f"{row['selected_model_id']}::{target}"]) for target in targets] for row in rows])
    reports={**evaluation['base_models'],'dynamic_selection':evaluation['dynamic_selection']}
    figure_folder=folder/'figures';figure_folder.mkdir(exist_ok=True)
    calibration={}
    for model_id,scores in predictions.items():
        calibration[model_id]={}
        for index,target in enumerate(targets):
            groups,ece=calibration_bins(truth[:,index],scores[:,index])
            brier=float(np.mean((truth[:,index]-scores[:,index])**2))
            if not np.isclose(brier,reports[model_id][target]['brier'],atol=1e-12):
                raise RuntimeError(f'Saved metric and prediction mismatch for {model_id}/{target}')
            reports[model_id][target]['ece_10_equal_width_bins']=ece
            reports[model_id][target]['log_loss']=float(log_loss(truth[:,index],scores[:,index],labels=[0,1]))
            calibration[model_id][target]=groups
        for field in ('ece_10_equal_width_bins','log_loss'):
            reports[model_id]['macro'][field]=float(np.mean([reports[model_id][target][field] for target in targets]))

    # Comparator analysis is deliberately offline and read-only.  It does not
    # change the registry, selector or dashboard.  The fixed model is chosen
    # from the stored tuning-validation report rather than final cohort labels.
    tuning_reports=json.loads((folder/'base_model_reports.json').read_text(encoding='utf-8'))
    fixed_id=max(model_ids,key=lambda model_id:tuning_reports[model_id]['metrics']['macro']['auroc'])
    registry=json.loads((ROOT/'model_registry'/'dynamic_selection.json').read_text(encoding='utf-8'))
    threshold_by_model={entry['id']:np.asarray([entry['thresholds'][target] for target in targets],dtype=float) for entry in registry['models']}
    selected_ids=np.asarray([row['selected_model_id'] for row in rows])
    selected_indices=np.asarray([model_ids.index(model_id) for model_id in selected_ids])
    mean_ensemble=np.mean(np.stack([predictions[model_id] for model_id in model_ids]),axis=0)
    per_study_losses=np.stack([np.mean((predictions[model_id]-truth)**2,axis=1) for model_id in model_ids],axis=1)
    oracle_indices=per_study_losses.argmin(axis=1)
    oracle_ids=np.asarray([model_ids[index] for index in oracle_indices])
    oracle_scores=np.stack([predictions[model_id][index] for index,model_id in enumerate(oracle_ids)])
    dynamic_thresholds=np.stack([threshold_by_model[model_id] for model_id in selected_ids])
    oracle_thresholds=np.stack([threshold_by_model[model_id] for model_id in oracle_ids])
    fixed_thresholds=np.tile(threshold_by_model[fixed_id],(len(rows),1))
    ensemble_thresholds=np.full((len(rows),len(targets)),0.5,dtype=float)
    comparator_scores={
        'fixed_validation_selected':predictions[fixed_id],
        'mean_probability_ensemble':mean_ensemble,
        'dynamic_selection':predictions['dynamic_selection'],
        'oracle_observed_brier':oracle_scores,
    }
    comparator_thresholds={
        'fixed_validation_selected':fixed_thresholds,
        'mean_probability_ensemble':ensemble_thresholds,
        'dynamic_selection':dynamic_thresholds,
        'oracle_observed_brier':oracle_thresholds,
    }
    comparator_reports={method_id:report_metrics(truth,scores,comparator_thresholds[method_id]) for method_id,scores in comparator_scores.items()}
    # Brier and AUROC have no threshold dependence. These checks catch a stale
    # prediction CSV before a comparison is written.
    for target in targets:
        if not np.isclose(comparator_reports['dynamic_selection'][target]['brier'],evaluation['dynamic_selection'][target]['brier'],atol=1e-12):
            raise RuntimeError(f'Dynamic comparison mismatch for {target}.')
    bootstrap=paired_bootstrap_summary(
        truth,
        comparator_scores['fixed_validation_selected'],
        comparator_scores,
    )
    evaluation['comparative_analysis']={
        'purpose':'Exploratory comparison of the deployed dynamic selector with a tuning-selected fixed reference, a simple probability average and a label-informed oracle upper bound.',
        'cohort':'Same recorded 120-study development cohort with prior project exposure; not an independent test.',
        'fixed_model_selection':{
            'model_id':fixed_id,
            'architecture':NAMES[fixed_id],
            'criterion':'Highest macro AUROC in the recorded base-model tuning-validation report.',
            'selected_without_final_cohort_labels':True,
            'limitation':'The tuning role was also used for base-model early stopping, so this is a development reference rather than a final confirmatory baseline.',
        },
        'mean_probability_ensemble':{
            'definition':'Unweighted arithmetic mean of the five calibrated candidate probabilities.',
            'classification_threshold':'0.50 fixed before final-cohort comparison; no final labels were used to optimise it.',
        },
        'oracle_observed_brier':{
            'definition':'For each study, choose the candidate with the smallest observed mean Brier loss across the three labels.',
            'deployment_status':'Analysis only; it uses the unknown reference labels and cannot be used for an uploaded study.',
        },
        'reports':comparator_reports,
        'bootstrap':bootstrap,
        'selector_behaviour':{
            'selected_matches_observed_oracle_count':int(np.sum(selected_ids==oracle_ids)),
            'studies':int(len(rows)),
            'selected_matches_observed_oracle_fraction':float(np.mean(selected_ids==oracle_ids)),
            'dynamic_selection_frequency':{model_id:int(np.sum(selected_ids==model_id)) for model_id in model_ids},
            'observed_oracle_frequency':{model_id:int(np.sum(oracle_ids==model_id)) for model_id in model_ids},
            'interpretation':'The match rate measures agreement with an outcome-informed oracle on this cohort. It is not a clinical accuracy score.',
        },
        'conclusion':'Any apparent difference is exploratory because the cohort had prior project exposure. The analysis tests whether the selector adds value relative to transparent baselines; it does not prove clinical superiority.',
    }

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ('roc','precision_recall','calibration'):
        fig,axes=plt.subplots(1,3,figsize=(16,4.8))
        for index,(target,label) in enumerate(LABELS.items()):
            ax=axes[index]
            for model_id,scores in predictions.items():
                width=2.8 if model_id=='dynamic_selection' else 1.3
                if kind=='roc':
                    x,y,_=roc_curve(truth[:,index],scores[:,index]);suffix=f"AUROC {reports[model_id][target]['auroc']:.3f}"
                elif kind=='precision_recall':
                    y,x,_=precision_recall_curve(truth[:,index],scores[:,index]);suffix=f"AP {reports[model_id][target]['average_precision']:.3f}"
                else:
                    groups=calibration[model_id][target];x=[g['mean_score'] for g in groups];y=[g['positive_fraction'] for g in groups]
                    suffix=f"ECE {reports[model_id][target]['ece_10_equal_width_bins']:.3f}"
                ax.plot(x,y,label=NAMES[model_id]+' '+suffix,linewidth=width)
            if kind in ('roc','calibration'):ax.plot([0,1],[0,1],'--',color='grey',linewidth=.8)
            ax.set_title(label);ax.set_xlim(0,1);ax.set_ylim(0,1.03);ax.grid(alpha=.18)
            ax.set_xlabel({'roc':'False positive rate','precision_recall':'Recall','calibration':'Mean predicted probability'}[kind])
            ax.set_ylabel({'roc':'Sensitivity','precision_recall':'Precision','calibration':'Observed positive fraction'}[kind])
            ax.legend(fontsize=7,loc='best')
        fig.suptitle('MRNet final DMS evaluation · 120 studies with prior project exposure',fontsize=12)
        fig.tight_layout();filename={'roc':'roc_curves.png','precision_recall':'precision_recall_curves.png','calibration':'calibration_curves.png'}[kind]
        fig.savefig(figure_folder/filename,dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,(target,label) in zip(axes,LABELS.items()):
        cm=np.asarray(evaluation['dynamic_selection'][target]['confusion_matrix'])
        ax.imshow(cm,cmap='Blues')
        for (row,col),count in np.ndenumerate(cm):ax.text(col,row,str(count),ha='center',va='center',color='white' if count>cm.max()/2 else '#13364a',fontsize=14)
        ax.set_title(label);ax.set_xticks([0,1],['Below threshold','Flagged']);ax.set_yticks([0,1],['Negative label','Positive label']);ax.set_xlabel('DMS prediction')
    fig.suptitle('Dynamic selection · locked per-model calibration thresholds');fig.tight_layout();fig.savefig(figure_folder/'confusion_matrices.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.8))
    ids=list(predictions);x=np.arange(len(ids));width=.25
    for index,target in enumerate(targets):ax.bar(x+(index-1)*width,[reports[model_id][target]['auroc'] for model_id in ids],width,label=LABELS[target])
    ax.set_xticks(x,[NAMES[model_id] for model_id in ids],rotation=15);ax.set_ylim(0,1);ax.set_ylabel('AUROC');ax.set_title('Five base models and dynamic system · same 120 studies');ax.legend();fig.tight_layout();fig.savefig(figure_folder/'model_comparison.png',dpi=160);plt.close(fig)

    # Transparent comparator view: this keeps the ensemble and label-informed
    # oracle separate from the deployable dynamic system.
    comparator_ids=list(comparator_scores)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    display=[NAMES[item] for item in comparator_ids]
    axes[0].bar(display,[comparator_reports[item]['macro']['auroc'] for item in comparator_ids],color=['#6c7a89','#9b59b6','#1f77b4','#95a5a6'])
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Macro AUROC');axes[0].tick_params(axis='x',rotation=18);axes[0].set_title('Discrimination')
    axes[1].bar(display,[comparator_reports[item]['macro']['brier'] for item in comparator_ids],color=['#6c7a89','#9b59b6','#1f77b4','#95a5a6'])
    axes[1].set_ylim(0,max(comparator_reports[item]['macro']['brier'] for item in comparator_ids)*1.25);axes[1].set_ylabel('Macro Brier loss (lower is better)');axes[1].tick_params(axis='x',rotation=18);axes[1].set_title('Probability error')
    fig.suptitle('Exploratory method comparison · recorded 120-study development cohort')
    fig.tight_layout();fig.savefig(figure_folder/'comparator_analysis.png',dpi=160);plt.close(fig)

    # Paired descriptive uncertainty estimate. This is evaluation, not a tuning
    # signal. Every fixed baseline is included rather than choosing one from test.
    rng=np.random.default_rng(42);indices=rng.integers(0,len(rows),size=(2000,len(rows)))
    dms_losses=np.mean((predictions['dynamic_selection']-truth)**2,axis=1)
    comparisons={}
    for model_id in model_ids:
        difference=dms_losses-np.mean((predictions[model_id]-truth)**2,axis=1)
        boot=difference[indices].mean(axis=1)
        comparisons[model_id]={'dms_minus_fixed_mean_brier':float(difference.mean()),
                               'paired_bootstrap_95_percent_interval':np.percentile(boot,[2.5,97.5]).tolist(),
                               'interpretation':'Negative favours DMS. Descriptive interval from one previously exposed MRNet cohort; no multiple-comparison adjustment or independent generalisation claim.'}
    evaluation['calibration_method']='10 equal-width probability bins for ECE; Brier and log loss from saved final predictions'
    evaluation['paired_brier_comparisons']=comparisons
    fixed_auroc=float(comparator_reports['fixed_validation_selected']['macro']['auroc'])
    dms_auroc=float(comparator_reports['dynamic_selection']['macro']['auroc'])
    evaluation['comparison_conclusion']=(
        'DMS changes architecture by study. Against the preselected fixed development reference '
        f'({NAMES[fixed_id]}), its recorded macro AUROC was {dms_auroc:.3f} versus {fixed_auroc:.3f}. '
        'This exploratory difference, on a cohort with prior project exposure, does not establish independent or clinical superiority.'
    )
    save_json(folder/'final_evaluation.json',evaluation)
    save_json(figure_folder/'metrics.json',{'metrics':evaluation['dynamic_selection'],'partition':evaluation['partition'],'limitation':evaluation['limitation']})
    save_json(folder/'calibration_bins.json',calibration)
    save_json(folder/'comparative_analysis.json',evaluation['comparative_analysis'])
    save_json(folder/'postprocessing_status.json',{'status':'completed','fitting_performed':False,'studies':len(rows),'paired_bootstrap_repeats':2000,'figures':sorted(p.name for p in figure_folder.glob('*.png'))})
    print(json.dumps({'status':'completed','studies':len(rows),'macro_auroc':evaluation['dynamic_selection']['macro']['auroc'],'paired_brier_comparisons':comparisons},indent=2))


if __name__=='__main__':main()
