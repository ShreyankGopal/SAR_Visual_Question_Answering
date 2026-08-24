"""
model/sar_projector.py
----------------------
MLP projector that maps SAR visual features into the LLM embedding space.

Architecture:
    Linear(d_sar, hidden_dim)  →  GELU  →  Linear(hidden_dim, llm_hidden_size)

This is the ONLY trainable component that bridges the SAR encoder and Vicuna.
The projector is kept simple (two-layer MLP) and fully trainable; LoRA is NOT
applied here.
"""
from typing import Optional

import torch
from torch import nn


class SARProjector(nn.Module):
    """
    Two-layer MLP projector: SAR feature space → LLM embedding space.

    Args:
        d_sar (int): Dimension of SAR encoder output features.
            For MaRS SwinV2-Base, f3 stage outputs 1024 channels → d_sar=1024.
        hidden_dim (int | None): Inner MLP dimension. Defaults to d_sar.
        llm_hidden_size (int): Target dimension (LLM hidden size).
            For Vicuna-7B, this is 4096. Read dynamically from model.config.hidden_size.
        dropout (float): Optional dropout between the two linear layers. Default 0.0.

    Input:  [B, N_visual, d_sar]          — spatially flattened SAR feature tokens
    Output: [B, N_visual, llm_hidden_size] — projected tokens ready for concatenation
    """

    def __init__(
        self,
        d_sar: int,
        llm_hidden_size: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_sar  # default: same width at both layers

        layers = [
            nn.Linear(d_sar, hidden_dim, bias=True),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, llm_hidden_size, bias=True))

        self.proj = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Small-scale init to avoid large initial loss."""
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar_features: FloatTensor [B, N_visual, d_sar]
        Returns:
            FloatTensor [B, N_visual, llm_hidden_size]
        """
        return self.proj(sar_features)
