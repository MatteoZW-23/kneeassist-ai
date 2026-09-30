from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build(pretrained=True):
    net=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    return net.features,1280
