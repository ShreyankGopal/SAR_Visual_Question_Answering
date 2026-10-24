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

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load weights from a checkpoint path (.pth file or directory containing projector.pth).
        Handles nested dicts ('projector_state_dict', 'projector', 'model', 'state_dict')
        and strips common prefixes ('projector.', 'mm_projector.', 'model.mm_projector.').
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Projector checkpoint path does not exist: {checkpoint_path}")

        if os.path.isdir(checkpoint_path):
            candidates = ["projector.pth", "sar_projector.pth"]
            file_path = None
            for cand in candidates:
                cand_path = os.path.join(checkpoint_path, cand)
                if os.path.exists(cand_path):
                    file_path = cand_path
                    break
            if file_path is None:
                pths = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.endswith(".pth")]
                if pths:
                    file_path = pths[0]
                else:
                    raise FileNotFoundError(f"No .pth file found in directory: {checkpoint_path}")
        else:
            file_path = checkpoint_path

        try:
            state = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(file_path, map_location="cpu", weights_only=False)

        # Unwrap if nested
        for key in ("projector_state_dict", "projector", "model", "state_dict"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break

        # Strip prefixes
        cleaned_state = {}
        for k, v in state.items():
            new_k = k
            for prefix in ("model.mm_projector.", "mm_projector.", "projector."):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    break
            cleaned_state[new_k] = v

        # Match dtype
        target_dtype = next(self.parameters()).dtype
        cleaned_state = {
            k: (v.to(dtype=target_dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in cleaned_state.items()
        }

        missing, unexpected = self.load_state_dict(cleaned_state, strict=False)
        return file_path, missing, unexpected
