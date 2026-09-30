from torch import nn
from torchvision.models import swin_t, Swin_T_Weights


class SwinFeatureMaps(nn.Module):
    """Adapt torchvision Swin NHWC feature tensors to the study aggregator's NCHW contract."""
    def __init__(self,features):
        super().__init__();self.features=features

    def forward(self,x):
        x=self.features(x)
        return x.permute(0,3,1,2).contiguous()


def build(pretrained=True):
    net=swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
    return SwinFeatureMaps(net.features),768
