"""Memory-bounded MRI encoder fine-tuning. Run only against a locked experiment config."""
import argparse
import json
import time
import numpy as np
import torch
from torchvision.transforms import functional as TF
from src.utils import config, resolve, save_json, fingerprint, seed_everything, get_device
from src.models.study_model import StudyModel
from src.data.dataset import read_manifest, split_and_check, volumes
from src.data.preprocessing import prepare
from src.training.train import train_arch, checkpoint, rng_state, restore_rng
from src.training.losses import weighted_bce
from src.evaluation.metrics import calculate
from src.evaluation.plots import training as training_plot


def enable_final_blocks(model, architecture):
    model.encoder.requires_grad_(False)
    # Preserve pretrained early features and frozen BatchNorm statistics.
    count=1 if architecture=='resnet18' else 2
    for block in list(model.encoder.children())[-count:]:
        block.requires_grad_(True)
    model.head.requires_grad_(True)
    return [p for p in model.encoder.parameters() if p.requires_grad]


def tensors_for(row,cfg,device,augment=False):
    result={}
    for plane,volume in volumes(row,cfg).items():
        tensor=prepare(volume,cfg['preprocessing'])[0]
        if augment:
            # Same modest transform for every slice within each sequence.
            # No flips: preserve anatomical orientation and laterality.
            angle=float(np.random.uniform(-5,5)); scale=float(np.random.uniform(.95,1.05))
            tensor=TF.affine(tensor,angle,[0,0],scale,[0.,0.])
        result[plane]=tensor.to(device)
    if augment and np.random.random()<cfg['training']['modality_dropout']:
        del result[np.random.choice(list(result))]
    return result


def infer_rows(model,rows,cfg,device):
    model.eval(); logits=[]
    with torch.no_grad():
        for row in rows:
            logits.append(model(tensors_for(row,cfg,device)).float().cpu().numpy()[0])
    z=np.stack(logits)
    if not np.isfinite(z).all():raise RuntimeError('Nonfinite validation logits.')
    return z


def fit(architecture,cfg,rows,split,device):
    folder=resolve(cfg['evaluation']['directory'])/architecture
    report_path=folder/'finetune_metrics.json'
    if report_path.exists():return json.loads(report_path.read_text())
    # A fresh head on fitting-only frozen features, never a previously selected head.
    warm=json.loads(json.dumps(cfg))
    warm['evaluation']['directory']=str(resolve(cfg['evaluation']['directory'])/'warmup')
    warm['paths']['models']=str(resolve(cfg['paths']['models'])/'warmup')
    warm['training']['epochs']=cfg['training']['warmup_epochs']
    warm['training']['batch_size']=32
    warm['training']['learning_rate']=.0003
    warm['model']['training_mode']='frozen_encoder'
    train_arch(architecture,rows,split,warm,device,resume=True)
    seed_everything(cfg['seed'],cfg['training']['cpu_threads'])
    initial=torch.load(resolve(warm['paths']['models'])/f'{architecture}_best.pth',map_location=device,weights_only=False)
    model=StudyModel(architecture,cfg,pretrained=False).to(device)
    model.load_state_dict(initial['model'])
    encoder_params=enable_final_blocks(model,architecture)
    opt=torch.optim.AdamW([{'params':encoder_params,'lr':cfg['training']['encoder_learning_rate']},
                           {'params':model.head.parameters(),'lr':cfg['training']['learning_rate']}],
                          weight_decay=cfg['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=1)
    amp=cfg['training']['mixed_precision'] and device.type=='cuda'
    # Conservative initial scaling avoids T500 FP16 head-gradient overflow.
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=128.)
    lookup={r['study_id']:r for r in rows}
    fit_rows=[lookup[k] for k in split['fit']]; tune_rows=[lookup[k] for k in split['tune']]
    def labels(items):return np.array([[int(r[t['key']]) for t in cfg['targets']] for r in items],dtype=np.float32)
    yfit=labels(fit_rows); ytune=labels(tune_rows)
    loss_fn=weighted_bce(torch.tensor(yfit,device=device),device)
    defaults={t['key']:.5 for t in cfg['targets']}
    best_path=resolve(cfg['paths']['models'])/f'{architecture}_best.pth'
    last_path=resolve(cfg['paths']['models'])/f'{architecture}_last.pth'
    signature=fingerprint({'config':cfg,'split':split})
    history=[]; stale=0; start=1; best=initial['internal_auroc']
    if last_path.exists():
        saved=torch.load(last_path,map_location=device,weights_only=False)
        if saved['signature']!=signature:raise ValueError('Resume config/split mismatch.')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        history=saved['history']; best=saved['best']; stale=saved['stale']; start=saved['epoch']+1
        restore_rng(saved['rng'])
    else:
        initial.update(config=cfg,signature=signature,epoch=0,thresholds=defaults,
                       training_stage='frozen_warmup',pretrained_encoder_frozen=True)
        checkpoint(best_path,initial)
    for epoch in range(start,cfg['training']['epochs']+1):
        if stale>=cfg['training']['early_stopping_patience']:break
        model.train(); total=0.; started=time.monotonic(); opt.zero_grad(set_to_none=True)
        order=np.random.permutation(len(fit_rows)); accumulation=cfg['training']['gradient_accumulation']; overflows=0
        for position,index in enumerate(order):
            batch=tensors_for(fit_rows[index],cfg,device,augment=True)
            remaining=min(accumulation,len(order)-(position//accumulation)*accumulation)
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model(batch)
                loss=loss_fn(logits.float(),torch.tensor(yfit[index:index+1],device=device))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss.')
            scaler.scale(loss/remaining).backward(); total+=float(loss.detach())
            if (position+1)%accumulation==0 or position+1==len(order):
                scaler.unscale_(opt)
                params=[p for p in model.parameters() if p.requires_grad]
                finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
                if finite:
                    torch.nn.utils.clip_grad_norm_(params,cfg['training']['gradient_clip'],error_if_nonfinite=True)
                    overflows=0
                else:
                    overflows+=1
                    if not amp or overflows>8:raise RuntimeError('Persistent nonfinite gradients; optimizer was not updated.')
                    print('AMP overflow: skipping optimizer step and reducing loss scale.',flush=True)
                # GradScaler skips the optimizer update automatically after overflow.
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            if (position+1)%25==0 or position==0:
                progress={'stage':'fine_tuning','architecture':architecture,'epoch':epoch,
                          'studies':position+1,'total':len(order),'elapsed_seconds':round(time.monotonic()-started)}
                save_json(resolve(cfg['run_directory'])/'progress.json',progress);print(json.dumps(progress),flush=True)
        z=infer_rows(model,tune_rows,cfg,device)
        from scipy.special import expit
        metrics=calculate(ytune,expit(z),cfg['targets'],defaults); auc=metrics['macro']['auroc']
        validation_loss=float(loss_fn(torch.tensor(z,device=device),torch.tensor(ytune,device=device)))
        scheduler.step(auc)
        history.append({'epoch':epoch,'training_loss':total/len(order),'validation_loss':validation_loss,
                        'training_auroc':None,'validation_auroc':auc,'learning_rate':opt.param_groups[0]['lr']})
        save_json(folder/'history.json',history)
        if auc>best+1e-6:
            best=auc;stale=0
            checkpoint(best_path,{'model':model.state_dict(),'architecture':architecture,'config':cfg,
                       'epoch':epoch,'internal_auroc':auc,'thresholds':defaults,'dataset':'MRNet-v1.0',
                       'training_stage':'final_encoder_blocks_finetuned','pretrained_encoder_frozen':False,
                       'signature':signature,'split_counts':{k:len(v) for k,v in split.items()}})
        else:stale+=1
        checkpoint(last_path,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                             'scaler':scaler.state_dict(),'rng':rng_state(),'history':history,'best':best,
                             'stale':stale,'epoch':epoch,'signature':signature})
        print(json.dumps({'architecture':architecture,'epoch':epoch,'validation_auroc':auc,'best':best}),flush=True)
    saved=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(saved['model'])
    z=infer_rows(model,tune_rows,cfg,device)
    from scipy.special import expit
    report={'architecture':architecture,'selected_epoch':saved['epoch'],'stage':saved['training_stage'],
            'metrics':calculate(ytune,expit(z),cfg['targets'],defaults),'epochs_run':len(history)}
    save_json(report_path,report)
    if history:training_plot(history,folder)
    del model;torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=config(args.config)
    rows=read_manifest(cfg);split,_=split_and_check(rows,cfg)
    for architecture in cfg['model']['architectures']:fit(architecture,cfg,rows,split,get_device(cfg))

if __name__=='__main__':main()
