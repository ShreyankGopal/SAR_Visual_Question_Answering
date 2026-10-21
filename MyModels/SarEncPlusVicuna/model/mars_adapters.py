"""
model/mars_adapters.py
----------------------
Trainable add-ons that live *inside* the frozen MaRS (SwinV2-Base) encoder:

    - LoRA on each block's attention `qkv` and `proj` weights.
    - WindowGeoRoPEBranch: a gated GeoRoPE attention branch that runs next to
      each block's own window attention (same windows, same shift).

Both are installed AFTER the MaRS checkpoint is loaded (install_mars_adapters),
so the checkpoint's keys still match. With everything at init the encoder's
output is exactly MaRS's: LoRA's B matrix and the branch gate both start at 0.

Per block, with the branch enabled:

    x = x + norm1(WindowAttn_LoRA(x)) + geo_gate * WindowGeoRoPEBranch(x, gsd)
    x = x + norm2(MLP(x))

The GeoRoPE maths (GCC + GFC, pairwise rotary phases) is the same as
model/georope_adapter.py, applied to the 64 tokens of each 8x8 window instead
of the 256-token f3 grid.
"""
import math
import os
import re
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

from .georope_adapter import _GeoFrequencyCalibration

# Substrings that identify the parameters this file adds to the encoder.
_ADAPTER_KEYS = ("lora_A", "lora_B", "geo_branch.", "geo_gate")


class _LoRAWeight(nn.Module):
    """
    LoRA as a weight parametrization: weight -> weight + (alpha/r) * B @ A.

    A parametrization (not a wrapper around the Linear) because timm's SwinV2
    attention reads `self.qkv.weight` directly via F.linear, so a wrapper's
    forward() -- including PEFT's -- would never be called.
    """

    # ponytail: no LoRA dropout (it acts on the layer input, which a weight
    # parametrization never sees). Wrap the Linear instead if dropout matters.
    def __init__(self, out_features: int, in_features: int, r: int, alpha: float):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + self.scale * (self.lora_B @ self.lora_A).to(weight.dtype)


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """[B, H, W, C] -> [B * num_windows, ws * ws, C], windows grouped per sample."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def _window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """Inverse of _window_partition: [B * num_windows, ws * ws, C] -> [B, H, W, C]."""
    C = windows.shape[-1]
    x = windows.view(-1, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, H, W, C)


class WindowGeoRoPEBranch(nn.Module):
    """
    GeoRoPE attention over each Swin window of a stage's feature map.
    Returns only the correction ([B, H, W, C]); the caller gates and adds it.

    GCC uses the same per-sample gsd_ratio at every stage. GFC is computed on
    the whole feature map (real local texture across window borders, real
    scene-level pooling) and then cut into windows.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        shift_size: int,
        num_heads: int = 4,
        rope_base: float = 100.0,
        gcc_alpha: float = 0.5,
        gfc_hidden_dim: int = 64,
    ):
        super().__init__()
        bottleneck = dim // 2
        if bottleneck % num_heads != 0 or (bottleneck // num_heads) % 4 != 0:
            raise ValueError(
                f"dim/2={bottleneck} must split into {num_heads} heads whose size is divisible by 4"
            )
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.head_dim = bottleneck // num_heads
        self.num_bands = self.head_dim // 4  # M
        self.gcc_alpha = gcc_alpha

        self.q_proj = nn.Linear(dim, bottleneck)
        self.k_proj = nn.Linear(dim, bottleneck)
        self.v_proj = nn.Linear(dim, bottleneck)
        self.out_proj = nn.Linear(bottleneck, dim)

        # grid_size is set per forward call from the incoming feature map.
        self.gfc = _GeoFrequencyCalibration(dim, 0, gfc_hidden_dim, self.num_bands)

        rows = torch.arange(window_size).repeat_interleave(window_size)
        cols = torch.arange(window_size).repeat(window_size)
        coords = torch.stack([rows, cols], dim=-1).float()                       # [N, 2]
        self.register_buffer("delta", coords.unsqueeze(1) - coords.unsqueeze(0), persistent=False)  # [N(q), N(k), 2]

        # rope_base defaults to 100 (not 10000 as in the f3-grid adapter):
        # offsets inside an 8x8 window are at most 7 tokens, so the lowest
        # frequencies of a 10000 base would never rotate.
        m = torch.arange(self.num_bands).float()
        self.register_buffer("omega", rope_base ** (-m / self.num_bands), persistent=False)      # [M]

    def forward(
        self,
        x: torch.Tensor,
        gsd_ratio: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """attn_mask: the host block's own shifted-window mask ([num_windows, N, N], 0 / -100), or None."""
        B, H, W, C = x.shape
        ws, shift, h, M = self.window_size, self.shift_size, self.num_heads, self.num_bands
        if H != W:
            raise ValueError(f"expected a square feature map, got {H}x{W}")

        # --- GFC on the full map: [B, H*W, M] -> [B, H, W, M] ---
        self.gfc.grid_size = H
        mu = self.gfc(x.reshape(B, H * W, C)).float().view(B, H, W, M)

        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
            mu = torch.roll(mu, shifts=(-shift, -shift), dims=(1, 2))
        xw = _window_partition(x, ws)                       # [Bw, N, C]
        mu = _window_partition(mu, ws)                      # [Bw, N, M]
        Bw, N, _ = xw.shape
        nW = Bw // B

        def split(t):                                       # -> 4 x [Bw, h, N, M]
            return t.view(Bw, N, h, M, 4).transpose(1, 2).unbind(-1)

        q_xr, q_xi, q_yr, q_yi = split(self.q_proj(xw))
        k_xr, k_xi, k_yr, k_yi = split(self.k_proj(xw))
        v = self.v_proj(xw).view(Bw, N, h, self.head_dim).transpose(1, 2)        # [Bw, h, N, d]

        # --- GCC: geo-calibrated offsets ---
        delta = self.delta.unsqueeze(0)                     # [1, N, N, 2]
        if gsd_ratio is not None:
            G = gsd_ratio.float().pow(1.0 - self.gcc_alpha).repeat_interleave(nW)  # [Bw]
            delta = delta * G.view(Bw, 1, 1, 1)

        # --- pairwise GeoRoPE phases, in fp32 ---
        mu_pair = torch.sqrt(mu.unsqueeze(2) * mu.unsqueeze(1))                  # [Bw, N, N, M]
        phase_x = delta[..., 0:1] * mu_pair * self.omega
        phase_y = delta[..., 1:2] * mu_pair * self.omega

        def rotary(qr, qi, kr, ki, phase):
            # q . rot(k, phase) = cos * (qr*kr + qi*ki) + sin * (qi*kr - qr*ki)
            cos = torch.cos(phase).unsqueeze(1).to(qr.dtype)                     # [Bw, 1, N, N, M]
            sin = torch.sin(phase).unsqueeze(1).to(qr.dtype)
            qr, qi = qr.unsqueeze(3), qi.unsqueeze(3)                            # [Bw, h, N, 1, M]
            kr, ki = kr.unsqueeze(2), ki.unsqueeze(2)                            # [Bw, h, 1, N, M]
            return ((qr * kr + qi * ki) * cos + (qi * kr - qr * ki) * sin).sum(-1)

        logits = rotary(q_xr, q_xi, k_xr, k_xi, phase_x) + rotary(q_yr, q_yi, k_yr, k_yi, phase_y)
        logits = logits / math.sqrt(self.head_dim)                               # [Bw, h, N, N]

        if attn_mask is not None:
            logits = logits + attn_mask.repeat(B, 1, 1).unsqueeze(1).to(logits.dtype)

        out = torch.matmul(torch.softmax(logits, dim=-1), v)                     # [Bw, h, N, d]
        out = self.out_proj(out.transpose(1, 2).reshape(Bw, N, h * self.head_dim))

        out = _window_reverse(out, ws, H, W)
        if shift:
            out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))
        return out


class MaRSBlockAdapter(nn.Module):
    """
    Wraps one SwinTransformerV2Block. Adds the gated GeoRoPE branch (if any)
    at the attention residual, and gradient-checkpoints the block so that
    backprop through MaRS does not keep every block's activations in memory.
    """

    def __init__(self, block: nn.Module, geo_branch: Optional[nn.Module], ctx, use_checkpoint: bool):
        super().__init__()
        self.block = block
        self.geo_branch = geo_branch
        if geo_branch is not None:
            # Per-channel gate, starts at 0: at init the block is exactly MaRS.
            self.geo_gate = nn.Parameter(torch.zeros(block.attn.qkv.in_features))
        self.ctx = ctx  # shared holder for the current batch's gsd_ratio
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor, gsd_ratio: Optional[torch.Tensor]) -> torch.Tensor:
        blk = self.block
        if self.geo_branch is None:
            return blk(x)
        # Same steps as timm's SwinTransformerV2Block.forward, with the branch
        # added next to the attention output.
        B, H, W, C = x.shape
        y = x + blk.drop_path1(blk.norm1(blk._attn(x)))
        y = y + self.geo_gate * self.geo_branch(x, gsd_ratio, blk.attn_mask).to(y.dtype)
        y = y.reshape(B, -1, C)
        y = y + blk.drop_path2(blk.norm2(blk.mlp(y)))
        return y.reshape(B, H, W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gsd_ratio = self.ctx.gsd_ratio
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, gsd_ratio, use_reentrant=False)
        return self._forward(x, gsd_ratio)


def install_mars_adapters(
    encoder: nn.Module,
    lora: bool = False,
    georope: bool = False,
    stages: Sequence[int] = (0, 1, 2, 3),
    lora_r: int = 16,
    lora_alpha: float = 32,
    georope_num_heads: int = 4,
    georope_rope_base: float = 100.0,
    gcc_alpha: float = 0.5,
    gfc_hidden_dim: int = 64,
    grad_checkpointing: bool = True,
    stage_a_checkpoint: Optional[str] = None,
) -> nn.Module:
    """
    Install LoRA and/or GeoRoPE branches on the SwinV2 blocks of `encoder`
    (the object returned by build_sar_encoder) in the given stages. Call this
    after the MaRS checkpoint is loaded and before building SARVLM.

    lora=True, georope=False is the "MaRS LoRA only" control.
    Both False: no-op, the encoder is returned untouched.

    stage_a_checkpoint: path to a stage-A LoRA file (StageA_ScaleAware's
    mars_lora.pth). The LoRA is loaded from it and kept FROZEN: SARVLM then
    runs MaRS without gradients, exactly as with no adapters. Requires
    lora=True, georope=False, and the same stages / lora_r / lora_alpha as stage A.
    """
    if stage_a_checkpoint and (georope or not lora):
        raise ValueError("mars.stage_a_checkpoint needs mars.lora: true and mars.georope: false")
    if not (lora or georope):
        return encoder

    blocks = [(n, m) for n, m in encoder.named_modules() if type(m).__name__ == "SwinTransformerV2Block"]
    if not blocks:
        raise RuntimeError("install_mars_adapters: no SwinTransformerV2Block found in the encoder")

    ctx = SimpleNamespace(gsd_ratio=None)
    encoder.mars_ctx = ctx
    installed = 0
    for name, blk in blocks:
        stage = int(re.search(r"layers[._](\d+)", name).group(1))
        if stage not in stages:
            continue

        if lora:
            for lin in (blk.attn.qkv, blk.attn.proj):
                parametrize.register_parametrization(
                    lin, "weight", _LoRAWeight(lin.out_features, lin.in_features, lora_r, lora_alpha)
                )

        branch = None
        if georope:
            branch = WindowGeoRoPEBranch(
                dim=blk.attn.qkv.in_features,
                window_size=blk.window_size[0],
                shift_size=blk.shift_size[0],
                num_heads=georope_num_heads,
                rope_base=georope_rope_base,
                gcc_alpha=gcc_alpha,
                gfc_hidden_dim=gfc_hidden_dim,
            )

        parent_name, _, child = name.rpartition(".")
        encoder.get_submodule(parent_name)._modules[child] = MaRSBlockAdapter(
            blk, branch, ctx, grad_checkpointing
        )
        installed += 1

    if stage_a_checkpoint:
        load_mars_adapters(encoder, stage_a_checkpoint)
        encoder.mars_frozen = True   # SARVLM keeps these params frozen
        print(f"[install_mars_adapters] loaded frozen stage-A LoRA from {stage_a_checkpoint}")

    n_new = sum(p.numel() for p in mars_adapter_params(encoder).values())
    print(
        f"[install_mars_adapters] lora={lora} georope={georope} stages={list(stages)}: "
        f"{installed}/{len(blocks)} blocks adapted, {n_new:,} new parameters"
    )
    return encoder


def mars_adapter_params(encoder: nn.Module) -> Dict[str, nn.Parameter]:
    """The parameters this file added to the encoder (empty dict if none)."""
    return {n: p for n, p in encoder.named_parameters() if any(k in n for k in _ADAPTER_KEYS)}


def save_mars_adapters(encoder: nn.Module, path: str) -> bool:
    """Save the adapter weights to `path`. Returns False (and writes nothing) if there are none."""
    params = mars_adapter_params(encoder)
    if not params:
        return False
    torch.save({n: p.detach().cpu() for n, p in params.items()}, path)
    return True


def load_mars_adapters(encoder: nn.Module, path: str, map_location="cpu") -> None:
    """
    Load adapter weights saved by save_mars_adapters. Safe to call always:
    it does nothing when no adapters are installed and `path` does not exist.

    Raises if the file and the installed adapters disagree in either
    direction, so a benchmark can never run with branches silently left at
    their random init, nor silently ignore trained ones.
    """
    params = mars_adapter_params(encoder)
    exists = os.path.exists(path)
    if not params and not exists:
        return
    if not exists:
        raise FileNotFoundError(
            f"{path} not found, but the `mars:` config enables in-MaRS adapters. "
            f"If this checkpoint was trained without them, set mars.lora and mars.georope to false."
        )
    if not params:
        raise RuntimeError(
            f"{path} exists, so this checkpoint was trained with in-MaRS adapters, but the "
            f"`mars:` config has them disabled. Use the config the run was trained with."
        )
    state = torch.load(path, map_location=map_location)
    if set(state) != set(params):
        missing, extra = sorted(set(params) - set(state)), sorted(set(state) - set(params))
        raise RuntimeError(
            f"MaRS adapter checkpoint does not match the installed adapters "
            f"(check the `mars:` config). Missing from file: {missing[:3]} ({len(missing)} total); "
            f"not in model: {extra[:3]} ({len(extra)} total)"
        )
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state[n])
