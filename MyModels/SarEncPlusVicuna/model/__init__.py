"""
SAR-VQA model package.

Public exports:
    make_sar_visual_mask     — bidirectional attention mask factory
    HybridLlamaModel         — LlamaModel with SAR visual mask injected
    HybridLlamaForCausalLM   — drop-in Vicuna replacement, weight-compatible
    SARProjector             — MLP projector: [B, N_v, D_sar] → [B, N_v, H]
    SARVLM                   — full model wrapper (encoder + projector + LLM)
    SAREncoderPlaceholder    — offline test stub mimicking the real MaRS encoder
    build_sar_encoder        — factory to load real MaRS SwinV2 checkpoint
"""

from .hybrid_llama import (
    make_sar_visual_mask,
    HybridLlamaModel,
    HybridLlamaForCausalLM,
)
from .sar_projector import SARProjector
from .sar_vlm import SARVLM, SAREncoderPlaceholder, build_sar_encoder

__all__ = [
    "make_sar_visual_mask",
    "HybridLlamaModel",
    "HybridLlamaForCausalLM",
    "SARProjector",
    "SARVLM",
    "SAREncoderPlaceholder",
    "build_sar_encoder",
]
