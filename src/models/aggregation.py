import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregation(nn.Module):
    """Learnable attention-based slice aggregation for better feature representation."""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, maps):
        # maps: (batch, channels, height, width) -> pooled to (slices, feature_dim)
        slices = maps.mean((-2, -1))  # (slices, feature_dim)
        attention_weights = torch.softmax(self.attention(slices), dim=0)  # (slices, 1)
        weighted_mean = (slices * attention_weights).sum(0)  # (feature_dim,)
        return weighted_mean

def aggregate(maps, method='mean_max', attention_module=None):
    """
    Aggregate slice-level features to study-level representation.
    
    Args:
        maps: Tensor of shape (slices, channels, height, width)
        method: Aggregation method ('mean_max', 'attention', 'mean_max_attention')
        attention_module: Optional AttentionAggregation module for attention-based methods
    
    Returns:
        Aggregated feature vector
    """
    slices = maps.mean((-2, -1))  # (slices, feature_dim)
    
    if method == 'mean_max':
        # Original method: simple mean and max pooling
        return torch.cat([slices.mean(0), slices.max(0).values])
    
    elif method == 'attention':
        # Attention-based aggregation only
        if attention_module is None:
            raise ValueError("Attention module required for attention aggregation")
        weighted_mean = attention_module(maps)
        return torch.cat([weighted_mean, slices.max(0).values])
    
    elif method == 'mean_max_attention':
        # Combined approach: mean, max, and attention-weighted
        if attention_module is None:
            raise ValueError("Attention module required for mean_max_attention aggregation")
        weighted_mean = attention_module(maps)
        return torch.cat([slices.mean(0), slices.max(0).values, weighted_mean])
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
