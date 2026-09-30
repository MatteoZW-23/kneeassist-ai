from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights


def build(pretrained=True):
    net=densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    return net.features,1024
