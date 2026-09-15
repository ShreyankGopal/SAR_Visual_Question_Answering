import torch
import torch.nn as nn


class SARProjector(nn.Module):
    """
    MLP projector for SAR encoder features to LLM embedding space.
    
    Maps SAR encoder output [B, N_patches, d_sar] to [B, N_patches, llm_hidden_size]
    where d_sar=1024 (SwinV2-Base) and llm_hidden_size=4096 (Vicuna-7B).
    
    This is a 2-layer MLP with GELU activation, trained while SAR encoder and LLM remain frozen.
    """
    def __init__(self, d_sar=1024, llm_hidden_size=4096, hidden_dim=4096):
        super().__init__()
        self.d_sar = d_sar
        self.llm_hidden_size = llm_hidden_size
        
        self.proj = nn.Sequential(
            nn.Linear(d_sar, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_hidden_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N_patches, d_sar] - SAR encoder features
        
        Returns:
            [B, N_patches, llm_hidden_size] - Projected features for LLM
        """
        return self.proj(x)
    
    @property
    def config(self):
        return {"mm_projector_type": 'sar_mlp'}
