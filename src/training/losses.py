import torch

def weighted_bce(labels,device):
    positives=labels.sum(0)
    if (positives==0).any() or (positives==len(labels)).any():
        raise ValueError('Every target needs positive and negative fitting examples.')
    weights=(len(labels)-positives)/positives
    return torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
