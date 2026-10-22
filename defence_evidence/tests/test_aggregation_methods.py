import copy

import torch
from torch import nn

from src.models.study_model import StudyModel
from src.models import resnet_model
from src.utils import config


class _IdentityEncoder(nn.Module):
    def forward(self, values):
        return values


def test_supported_aggregation_methods_keep_head_and_mask_dimensions_aligned(monkeypatch):
    monkeypatch.setattr(resnet_model, 'build', lambda pretrained: (_IdentityEncoder(), 3))
    for method in ('mean_max', 'attention', 'mean_max_attention'):
        cfg = copy.deepcopy(config())
        cfg['model']['aggregation_method'] = method
        model = StudyModel('resnet18', cfg, pretrained=False).eval()
        output = model({'axial': torch.randn(2, 3, 8, 8)})
        assert output.shape == (1, len(cfg['targets']))
