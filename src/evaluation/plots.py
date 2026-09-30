import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,ConfusionMatrixDisplay

def curves(y,p,metrics,targets,directory):
    directory.mkdir(parents=True,exist_ok=True)
    for kind in ['roc','precision_recall']:
        fig,ax=plt.subplots(figsize=(7,5))
        for j,t in enumerate(targets):
            if kind=='roc':
                a,b,_=roc_curve(y[:,j],p[:,j]);label=f"{t['display']} (AUC {metrics[t['key']]['auroc']:.3f})"
            else:
                b,a,_=precision_recall_curve(y[:,j],p[:,j]);label=f"{t['display']} (AP {metrics[t['key']]['average_precision']:.3f})"
            ax.plot(a,b,label=label)
        if kind=='roc':ax.plot([0,1],[0,1],'--',color='gray')
        ax.set(xlabel='False positive rate' if kind=='roc' else 'Recall',ylabel='Sensitivity' if kind=='roc' else 'Precision',
               title='KneeAssist AI · '+kind.replace('_',' ').title(),xlim=(0,1),ylim=(0,1.02))
        ax.legend(loc='lower right' if kind=='roc' else 'lower left');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(directory/(kind+'_curves.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(targets),figsize=(5*len(targets),4),squeeze=False)
    for ax,t in zip(axes[0],targets):
        ConfusionMatrixDisplay(__import__('numpy').array(metrics[t['key']]['confusion_matrix']),display_labels=['Negative','Positive']).plot(ax=ax,colorbar=False,cmap='Blues')
        ax.set_title(t['display'])
    fig.tight_layout();fig.savefig(directory/'confusion_matrices.png',dpi=160);plt.close(fig)

def training(history,directory):
    epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k,label in [('training_loss','Training'),('validation_loss','Internal validation')]:axes[0].plot(epochs,[r[k] for r in history],label=label)
    for k,label in [('training_auroc','Training'),('validation_auroc','Internal validation')]:axes[1].plot(epochs,[r[k] for r in history],label=label)
    axes[0].set(xlabel='Epoch',ylabel='Weighted BCE loss',title='Loss curves')
    axes[1].set(xlabel='Epoch',ylabel='Macro AUROC',title='Study-level AUROC')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(directory/'training_validation_curves.png',dpi=160);plt.close(fig)
