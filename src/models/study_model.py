import os
import torch
from torch import nn
from src.utils import ROOT
from src.models import resnet_model,efficientnet_model,densenet_model,transformer_model
from src.models.aggregation import aggregate, AttentionAggregation

class StudyModel(nn.Module):
    def __init__(self,architecture,cfg,pretrained=True):
        super().__init__()
        self.cfg=cfg  # Store config for compatibility
        os.environ['TORCH_HOME']=str(ROOT/'models/pretrained')
        factory={
            'resnet18':resnet_model.build,
            'resnet50':resnet_model.build50,
            'densenet121':densenet_model.build,
            'efficientnet_b0':efficientnet_model.build,
            'swin_t':transformer_model.build,
        }
        if architecture not in factory: raise ValueError('Unsupported model architecture.')
        self.encoder,self.channels=factory[architecture](pretrained)
        self.encoder.requires_grad_(False);self.encoder.eval()
        self.encoder_chunk_size=cfg['model'].get('encoder_chunk_size',4)
        self.planes=cfg['preprocessing']['planes']
        self.aggregation_method=cfg['model'].get('aggregation_method','mean_max')
        
        # Initialize attention modules if needed
        self.attention_modules=nn.ModuleDict()
        if 'attention' in self.aggregation_method:
            for plane in self.planes:
                self.attention_modules[plane]=AttentionAggregation(self.channels)
        
        # ``aggregate`` returns two channel vectors for mean/max and attention
        # pooling, and three for mean/max/attention pooling.  Keep this value
        # consistent with both the classifier head and missing-plane padding.
        plane_dims = {
            'mean_max': 2 * self.channels,
            'attention': 2 * self.channels,
            'mean_max_attention': 3 * self.channels,
        }
        if self.aggregation_method not in plane_dims:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        self.plane_dim=plane_dims[self.aggregation_method]
        dim=self.plane_dim*len(self.planes)
            
        self.register_buffer('feature_mean',torch.zeros(dim))
        self.register_buffer('feature_std',torch.ones(dim))
        self.head=nn.Sequential(nn.Linear(dim+len(self.planes),cfg['model']['hidden_dim']),nn.ReLU(),
                               nn.Dropout(cfg['model']['dropout']),nn.Linear(cfg['model']['hidden_dim'],len(cfg['targets'])))

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def feature_logits(self,features,mask):
        x=(features-self.feature_mean)/self.feature_std
        x=x*mask.repeat_interleave(self.plane_dim,dim=-1)
        return self.head(torch.cat([x,mask],dim=-1))

    def study_features(self,tensors):
        exemplar=next(iter(tensors.values()));features=[];mask=[]
        for plane in self.planes:
            if plane in tensors:
                # T500/cuDNN can produce NaNs for FP16 convolutions. The frozen
                # encoder stays FP32; the trainable head uses CUDA AMP.
                with torch.autocast(device_type=exemplar.device.type,enabled=False):
                    if self.training and any(p.requires_grad for p in self.encoder.parameters()):
                        from torch.utils.checkpoint import checkpoint
                        maps=[checkpoint(self.encoder,chunk.float(),use_reentrant=False)
                              for chunk in tensors[plane].split(self.encoder_chunk_size)]
                        maps=torch.cat(maps)
                    else:
                        maps=self.encoder(tensors[plane].float())
                    
                    # Use appropriate aggregation method
                    attention_module=self.attention_modules[plane] if 'attention' in self.aggregation_method else None
                    aggregated=aggregate(maps,method=self.aggregation_method,attention_module=attention_module)
                    features.append(aggregated)
                mask.append(1.)
            else:
                # Missing planes always reserve the exact per-plane feature size.
                features.append(torch.zeros(self.plane_dim,device=exemplar.device))
                mask.append(0.)
        return torch.cat(features),torch.tensor(mask,device=exemplar.device)

    def forward(self,tensors):
        f,m=self.study_features(tensors)
        return self.feature_logits(f.unsqueeze(0),m.unsqueeze(0))
