import torch

def build(optimizer,cfg):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',
            factor=cfg['scheduler_factor'],patience=cfg['scheduler_patience'])
