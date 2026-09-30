from torch import nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

def build(pretrained=True):
    net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(net.children())[:-2]),512

def build50(pretrained=True):
    net=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    return nn.Sequential(*list(net.children())[:-2]),2048
