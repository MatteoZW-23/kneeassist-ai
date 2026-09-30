import argparse
import csv
import json
import os
import random
import shutil
import time
import traceback
import numpy as np
import torch
from src.utils import ROOT,config,resolve,save_json,seed_everything,get_device,fingerprint
from src.data.dataset import read_manifest,split_and_check,volumes
from src.data.preprocessing import prepare
from src.models.study_model import StudyModel
from src.training.losses import weighted_bce
from src.training.scheduler import build as scheduler_for
from src.evaluation.metrics import calculate,choose_thresholds
from src.evaluation.plots import curves,training as training_plot

def status(stage,**kw):
    report={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),**kw}
    save_json(ROOT/'logs/status.json',report);print(json.dumps(report),flush=True)

def checkpoint(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp);tmp.replace(path)

def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def features_for(model,rows,cfg,arch,device):
    hashes=json.loads((ROOT/'data/content_hashes.json').read_text())
    signature=fingerprint({'arch':arch,'preprocessing':cfg['preprocessing'],'hashes':hashes,'encoder_precision':'float32','version':2})[:16]
    folder=resolve(cfg.get('paths',{}).get('features','data/features'))/f'{arch}_{signature}';folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();vectors=[]
    model.eval()
    for i,r in enumerate(rows):
        dest=folder/(r['study_id']+'.npy')
        if dest.exists(): f=np.load(dest,allow_pickle=False)
        else:
            v=volumes(r,cfg);t={p:prepare(a,cfg['preprocessing'])[0].to(device) for p,a in v.items()}
            with torch.no_grad(),torch.autocast(device_type=device.type,enabled=cfg['training']['mixed_precision'] and device.type=='cuda'):
                f,_=model.study_features(t)
            f=f.float().cpu().numpy()
            if f.shape!=(model.plane_dim*len(model.planes),) or not np.isfinite(f).all():
                raise ValueError('Encoder produced invalid features; no cache was saved.')
            tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as stream:np.save(stream,f,allow_pickle=False)
            tmp.replace(dest)
        if f.shape!=(model.plane_dim*len(model.planes),) or not np.isfinite(f).all():raise ValueError('Invalid feature cache.')
        vectors.append(f)
        if i==0 or (i+1)%25==0 or i+1==len(rows):
            elapsed=time.monotonic()-start
            status('feature_extraction',architecture=arch,exams=i+1,total=len(rows),seconds=round(elapsed),eta_seconds=round(elapsed/(i+1)*(len(rows)-i-1)))
    return np.stack(vectors)

def train_arch(arch,rows,split,cfg,device,resume=False,stop_after=None):
    output=resolve(cfg['evaluation']['directory'])/arch;output.mkdir(parents=True,exist_ok=True)
    final=output/'internal_metrics.json'
    if final.exists() and resume:return json.loads(final.read_text())
    if final.exists():raise ValueError(f'{arch} already completed. Use --resume or a new results directory.')
    seed_everything(cfg['seed'],cfg['training']['cpu_threads'])
    model=StudyModel(arch,cfg,pretrained=True).to(device)
    x_np=features_for(model,rows,cfg,arch,device)
    lookup={r['study_id']:i for i,r in enumerate(rows)}
    fit=np.array([lookup[k] for k in split['fit']]);tune=np.array([lookup[k] for k in split['tune']])
    labels=np.array([[int(r[t['key']]) for t in cfg['targets']] for r in rows],dtype=np.float32)
    y=torch.tensor(labels,device=device);x=torch.tensor(x_np,device=device)
    with torch.no_grad():
        model.feature_mean.copy_(x[fit].mean(0));model.feature_std.copy_(x[fit].std(0,correction=0).clamp_min(.01))
    loss_fn=weighted_bce(y[fit],device)
    opt=torch.optim.AdamW(model.head.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    scheduler=scheduler_for(opt,cfg['training']);amp=cfg['training']['mixed_precision'] and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=[];best=-1.;stale=0;start_epoch=1
    model_dir=resolve(cfg.get('paths',{}).get('models','models'))
    last=model_dir/f'{arch}_last.pth';best_path=model_dir/f'{arch}_best.pth'
    signature=fingerprint({'config':cfg,'split':split})
    if resume and last.exists():
        old=torch.load(last,map_location=device,weights_only=False)
        if old['signature']!=signature:raise ValueError('Resume configuration or split differs from the saved run.')
        model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler']);scaler.load_state_dict(old['scaler'])
        history=old['history'];best=old['best'];stale=old['stale'];start_epoch=old['epoch']+1;restore_rng(old['rng'])
        status('resumed',architecture=arch,next_epoch=start_epoch)
    default={t['key']:t['threshold'] for t in cfg['targets']}
    def predict(indices):
        model.eval()
        with torch.no_grad():
            # Use the same AMP behavior as inference.
            with torch.autocast(device_type=device.type,enabled=amp):logits=model.feature_logits(x[indices],torch.ones((len(indices),len(model.planes)),device=device))
            loss=float(loss_fn(logits.float(),y[indices]))
        return logits.float().sigmoid().cpu().numpy(),loss
    for epoch in range(start_epoch,cfg['training']['epochs']+1):
        if stale>=cfg['training']['early_stopping_patience']:break
        model.train();loss_sum=0;count=0
        order=np.random.permutation(fit)
        for start in range(0,len(order),cfg['training']['batch_size']):
            idx=order[start:start+cfg['training']['batch_size']]
            mask=torch.ones((len(idx),len(model.planes)),device=device)
            drop=torch.rand(len(idx),device=device)<cfg['training']['modality_dropout']
            for i in drop.nonzero().flatten():mask[i,torch.randint(len(model.planes),(1,),device=device)]=0
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model.feature_logits(x[idx],mask);loss=loss_fn(logits,y[idx])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stopping before corrupting checkpoints.')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.head.parameters(),cfg['training']['gradient_clip'])
            scaler.step(opt);scaler.update();loss_sum+=float(loss.detach())*len(idx);count+=len(idx)
        pfit,_=predict(fit);ptune,vloss=predict(tune)
        train_m=calculate(labels[fit],pfit,cfg['targets'],default);val_m=calculate(labels[tune],ptune,cfg['targets'],default)
        auc=val_m['macro']['auroc'];scheduler.step(auc)
        row={'epoch':epoch,'training_loss':loss_sum/count,'validation_loss':vloss,
             'training_auroc':train_m['macro']['auroc'],'validation_auroc':auc,'learning_rate':opt.param_groups[0]['lr']}
        history.append(row);save_json(output/'history.json',history)
        if auc>best+1e-6:
            best=auc;stale=0
            checkpoint(best_path,{'model':model.state_dict(),'architecture':arch,'config':cfg,'epoch':epoch,
                                  'internal_auroc':best,'dataset':'MRNet-v1.0','split_counts':{k:len(v) for k,v in split.items()},
                                  'thresholds':default,'pretrained_encoder_frozen':True,'signature':signature})
        else:stale+=1
        checkpoint(last,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                         'scaler':scaler.state_dict(),'rng':rng_state(),'history':history,'best':best,'stale':stale,
                         'epoch':epoch,'signature':signature,'architecture':arch,'dataset':'MRNet-v1.0','config':cfg})
        status('training',architecture=arch,**row,best_validation_auroc=best)
        if stop_after and epoch>=stop_after:
            status('paused_for_resume_test',architecture=arch,epoch=epoch);return None
    best_ckpt=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(best_ckpt['model'])
    p,_=predict(tune)
    thresholds=choose_thresholds(labels[tune],p,cfg['targets']) if cfg['evaluation']['threshold_selection']=='internal_validation_f1' else default
    best_ckpt['thresholds']=thresholds;checkpoint(best_path,best_ckpt)
    report={'architecture':arch,'dataset':'MRNet-v1.0','selected_epoch':best_ckpt['epoch'],'epochs_run':len(history),
            'partition':'internal_validation','metrics':calculate(labels[tune],p,cfg['targets'],thresholds)}
    save_json(final,report);training_plot(history,output);curves(labels[tune],p,report['metrics'],cfg['targets'],output)
    del model,x,y;torch.cuda.empty_cache()
    return report

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--config');args=parser.parse_args();cfg=config(args.config)
    if not cfg.get('lifecycle',{}).get('training_enabled',True):
        print('Training is on hold until pending datasets and the validation plan are ready. See docs/RETRAINING_PLAN.md.');return
    if args.resume and (resolve(cfg['evaluation']['directory'])/'final/metrics.json').exists() and resolve(cfg['evaluation']['checkpoint']).exists():
        saved=torch.load(resolve(cfg['evaluation']['checkpoint']),map_location='cpu',weights_only=False)
        if fingerprint(saved['config'])!=fingerprint(cfg):
            raise ValueError('A completed run uses a different configuration. Archive it before starting a new experiment.')
        status('completed',architecture=saved['architecture'],message='This run is already complete; saved results were preserved.')
        return
    seed_everything(cfg['seed'],cfg['training']['cpu_threads']);device=get_device(cfg)
    status('starting',device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None)
    rows=read_manifest(cfg);split,audit=split_and_check(rows,cfg)
    reports=[]
    for arch in cfg['model']['architectures']:
        r=train_arch(arch,rows,split,cfg,device,args.resume,args.stop_after)
        if r is None:return
        reports.append(r)
    winner=max(reports,key=lambda r:r['metrics']['macro']['auroc'])
    source=resolve(cfg.get('paths',{}).get('models','models'))/f"{winner['architecture']}_best.pth"
    selected=resolve(cfg['evaluation']['checkpoint']);selected.parent.mkdir(parents=True,exist_ok=True)
    temporary=selected.with_suffix('.tmp');shutil.copy2(source,temporary);temporary.replace(selected)
    selection={'criterion':'Internal validation macro AUROC (not training accuracy, not official validation)',
               'winner':winner['architecture'],'models':reports,'dataset':'MRNet-v1.0'}
    save_json(resolve(cfg['evaluation']['directory'])/'model_comparison.json',selection)
    status('evaluating_selected_model',architecture=winner['architecture'])
    from src.evaluation.evaluate import evaluate
    evaluate(cfg,rows,split)
    status('completed',architecture=winner['architecture'])

if __name__=='__main__':
    try:run()
    except Exception as e:status('failed',error=str(e));traceback.print_exc();raise
