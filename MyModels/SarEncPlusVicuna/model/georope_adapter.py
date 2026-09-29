"""
model/georope_adapter.py
-------------------------
GeoRoPEVisualAdapter: a lightweight attention-parallel adapter that gives
the SAR visual tokens genuine 2D, ground-aware rotary position information,
instead of the sequential 1D RoPE positions Vicuna would otherwise assign
them once they're concatenated with text (see hybrid_llama.py).

Ported from GeoRoPE (Luo et al., "Ground-Aware Rotary Adaptation for
Remote Sensing Foundation Models"):
    - Geo-Coordinate Calibration (GCC): rescales grid offsets by a
      per-sample ground-distance ratio. Defaults to identity (G=1) since
      the current data pipeline has no per-sample GSD metadata; passing
      `gsd_ratio` later activates it with no other code changes required.
    - Geo-Frequency Calibration (GFC): a small conv+MLP network predicts a
      bounded, content-driven modulation of the rotary frequency per token
      pair, adapting positional sensitivity to local SAR scene texture.

Because GFC's pairwise factor depends jointly on both tokens in a pair
(mu_u,v = sqrt(mu_u * mu_v)), it cannot be decomposed into independent
per-token phases the way standard RoPE is (rotate Q and K separately, then
dot product). This module instead builds the explicit [B, heads, N, N, M]
pairwise phase tensor and rotates K per-query directly. N (visual token
count) is small and fixed (256 in real runs), so this stays cheap.
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class _GeoFrequencyCalibration(nn.Module):
    """adapts spatial attention to intra-scene spatial heterogeneity"""
    """
    GFC: predicts a per-token, per-frequency-band modulation factor
    mu_u,m in (1/e, e) from local texture (depthwise conv) and global
    scene context (pooled + MLP), per Eq. 12-14 of the GeoRoPE paper.
    """

    def __init__(self, hidden_size: int, grid_size: int, gfc_hidden_dim: int, num_bands: int):
        super().__init__()
        self.grid_size = grid_size

        """Part 1: local tecxture descriptor (depthwise conv -> GeLu in forward function)"""
        """First reduce dim for compute efficiency - not there in GeoRoPE paper"""
        self.local_proj = nn.Conv2d(hidden_size, gfc_hidden_dim, kernel_size=1)
        self.local_conv = nn.Conv2d(
            gfc_hidden_dim, gfc_hidden_dim, kernel_size=3, padding=1, groups=gfc_hidden_dim
        )

        """ Part 2: Global landscape prior - MLP on pooled tokens - pooling in forward"""
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_size, gfc_hidden_dim),
            nn.GELU(),
            nn.Linear(gfc_hidden_dim, gfc_hidden_dim),
        )
        self.merge = nn.Conv2d(gfc_hidden_dim * 2, num_bands, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, H]/[Batch, no. of visual tokens, hidden size/embedding vector dim] -> mu: [B, N, M]"""
        B, N, H = tokens.shape
        g = self.grid_size

        spatial = tokens.transpose(1, 2).reshape(B, H, g, g) 
        #Local texture descriptor         # [B, H, g, g]
        local = F.gelu(self.local_conv(self.local_proj(spatial)))     # [B, gfc, g, g]

        pooled = tokens.mean(dim=1)    
        #global landscape prior                                # [B, H]
        global_feat = self.global_mlp(pooled)[:, :, None, None]        # [B, gfc, 1, 1]
        global_feat = global_feat.expand(-1, -1, g, g)  # [B, gfc, g, g]
        """projected via a 1 x 1 convolution into an $M$-channel frequency
                -modulating latent space H"""
        """Here, M = d_r / 4 represents the number of frequency bands per spatial axis"""               
        bands = self.merge(torch.cat([local, global_feat], dim=1))     # [B, M, g, g]
        bands = bands.flatten(2).transpose(1, 2)  # [B, N, M]
        """ GFC modulation bounded scaling factor"""                
        """alpha(u, m) - u is the position of the token, m is the frequency band index"""
        return torch.exp(torch.tanh(bands))

"""SARVLM.from_vicuna() will construct this adapter
 internally, reading hyperparameters from a new georope: section of 
 train_config.yaml"""
class GeoRoPEVisualAdapter(nn.Module):
    """
    Self-attention adapter over the visual-token subsequence only.
    Returns `sar_tokens + correction` (owns its own residual add).
    """
    #inputs 
    #llm_hidden_size: hidden dimensions of vicuna embedding - 4096 - config.hidden_size
    #config.hidden_size used in sar ablation projector as well
    #n_visual: number of visual tokens - 256 - config["n_visual"]
    #bottleneck_dim: bottleneck dimension of georope adapter
    #num_heads: number of attention heads in georope adapter
    #rope_base: rope frequency base for 2D rope standard is 10000
    #gcc_alpha: alpha parameter for geo-coordinate calibration
    #gfc_hidden_dim: hidden dimension for geo-frequency calibration
    #zero_init_output: whether to zero-initialize the output later

    def __init__(
        self,
        llm_hidden_size: int,
        n_visual: int,
        bottleneck_dim: int = 256,
        num_heads: int = 4,
        rope_base: float = 10000.0,
        gcc_alpha: float = 0.5,
        gfc_hidden_dim: int = 64,
        zero_init_output: bool = False,
    ):
        super().__init__()

        grid_size = int(round(math.sqrt(n_visual)))
        if grid_size * grid_size != n_visual:
            raise ValueError(
                f"n_visual={n_visual} must be a perfect square (got {grid_size}x{grid_size})"
            )
        if bottleneck_dim % num_heads != 0:
            raise ValueError("bottleneck_dim must be divisible by num_heads")
        head_dim = bottleneck_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError("bottleneck_dim / num_heads must be divisible by 4 for 2D RoPE")

        self.n_visual = n_visual
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        #Each head's 64 channels need to split evenly into "bands of 4" for georope adapter
        self.num_bands = head_dim // 4  # M
        self.gcc_alpha = gcc_alpha
        #standard Q, K, V projections 
        #Compress from llm_hidden_size of 4096 to bottleneck_dim of 256 
        #makes georope adaption cheaper
        self.q_proj = nn.Linear(llm_hidden_size, bottleneck_dim)
        self.k_proj = nn.Linear(llm_hidden_size, bottleneck_dim)
        self.v_proj = nn.Linear(llm_hidden_size, bottleneck_dim)
        #bring it back to llm_hidden_size of 4096 for vicuna to process
        self.out_proj = nn.Linear(bottleneck_dim, llm_hidden_size)
        #can set zero_init_output to True for training stability
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        #initialise GFC module
        self.gfc = _GeoFrequencyCalibration(llm_hidden_size, grid_size, gfc_hidden_dim, self.num_bands)
        #rows:repeat_interleave(grid_size) gives [0,0,0,...,0, 1,1,...,1, ...]
        #  (row index repeated grid_size times each) — the row of token i. 
        # cols gives [0,1,2,...,15, 0,1,2,...,15, ...]
        #  (cycling) — the column of token i. 
        # Stacked: grid_coords[i] = (row_i, col_i).
        # grid_coords[0] = (rows[0], cols[0]) = (0, 0)
        # grid_coords[1] = (rows[1], cols[1]) = (0, 1)
        rows = torch.arange(grid_size).repeat_interleave(grid_size)
        cols = torch.arange(grid_size).repeat(grid_size)
        grid_coords = torch.stack([rows, cols], dim=-1).float()  # [N, 2] (row, col)
        #Registered as a buffer (not a trainable parameter — moves with .to(device)/.to(dtype),
        # never gets gradients) with persistent=False
        # (not saved to state_dict, since it's always trivially recomputable from n_visual).
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        #The standard RoPE frequency schedule (Eq 4 in paper). 
        # Produces 16 frequencies, geometrically spaced from ω_0 = 1 
        # (fastest-changing, sensitive to a 1-token offset) 
        # down to a tiny ω_15 (slowest-changing, only responds to large offsets)
        m = torch.arange(self.num_bands).float()
        omega = rope_base ** (-4.0 * m / head_dim)  # [M]
        self.register_buffer("omega", omega, persistent=False) #again not trainable but pushed to GPU

    def forward(self, sar_tokens: torch.Tensor, gsd_ratio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sar_tokens: [B, N_visual, H] — projector output, in LLM space.
            gsd_ratio: optional [B] tensor of (ground_distance_per_token /
                reference_ground_distance) for GCC. None => identity (G=1),
                which is the only mode currently exercised by SARVLM since
                the data pipeline has no per-sample GSD yet.
        Returns:
            [B, N_visual, H] — sar_tokens plus the geo-aware correction.
        """
        B, N, H = sar_tokens.shape
        if N != self.n_visual:
            raise ValueError(f"expected {self.n_visual} visual tokens, got {N}")
        #Project [B,N,4096] → [B,N,256], reshape the 256 into 
        # (4 heads, 64 dims), then swap dims 1↔2 → [B,4,N,64].
        q = self.q_proj(sar_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(sar_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(sar_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # --- GCC: geo-calibrated coordinates (Eq. 10-11) ---
        coords = self.grid_coords.float()  # [N, 2]
        if gsd_ratio is None:
            calib = coords.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2] #Duplication
        else:
            G = gsd_ratio.float().pow(1.0 - self.gcc_alpha)  # [B]
            calib = coords.unsqueeze(0) * G.view(B, 1, 1) #[B, N, 2]

        #calib.unsqueeze(2) → [B,N,1,2] (query's coord, broadcast across every key); 
        # calib.unsqueeze(1) → [B,1,N,2] (key's coord, broadcast across every query). 
        # Subtracting gives [B,N,N,2] where entry [b,u,v,:] = calib[u] - calib[v] = Δu,v
        delta = calib.unsqueeze(2) - calib.unsqueeze(1)  # [B, N(q), N(k), 2]
        delta_x, delta_y = delta[..., 0], delta[..., 1] #[B, N(q), N(k)] each 

        # --- GFC: per-token modulation, symmetric pairwise aggregation (Eq. 12-14) ---
        mu = self.gfc(sar_tokens).float()  # [B, N, M]
        mu_pair = torch.sqrt(mu.unsqueeze(2) * mu.unsqueeze(1))  # [B, N(q), N(k), M]

        # --- combined GeoRoPE phase per pair/band/axis (Eq. 8), computed in fp32 ---
        phase_x = delta_x.unsqueeze(-1) * mu_pair * self.omega  # [B, N, N, M]
        phase_y = delta_y.unsqueeze(-1) * mu_pair * self.omega

        cos_x = torch.cos(phase_x).unsqueeze(1).to(q.dtype)  # [B, 1, N(q), N(k), M]
        sin_x = torch.sin(phase_x).unsqueeze(1).to(q.dtype)
        cos_y = torch.cos(phase_y).unsqueeze(1).to(q.dtype)
        sin_y = torch.sin(phase_y).unsqueeze(1).to(q.dtype)

        # Each head channel splits into M bands x 2 axes x (re, im).
        q_x_re, q_x_im, q_y_re, q_y_im = q.view(B, self.num_heads, N, self.num_bands, 4).unbind(-1)
        k_x_re, k_x_im, k_y_re, k_y_im = k.view(B, self.num_heads, N, self.num_bands, 4).unbind(-1)

        # Rotate K per (query, key) pair — cannot be shared across queries
        # because the GFC modulation is itself pair-dependent.
        k_x_re_b, k_x_im_b = k_x_re.unsqueeze(2), k_x_im.unsqueeze(2)  # [B, h, 1, N(k), M]
        k_y_re_b, k_y_im_b = k_y_re.unsqueeze(2), k_y_im.unsqueeze(2)

        k_x_re_rot = k_x_re_b * cos_x - k_x_im_b * sin_x
        k_x_im_rot = k_x_re_b * sin_x + k_x_im_b * cos_x
        k_y_re_rot = k_y_re_b * cos_y - k_y_im_b * sin_y
        k_y_im_rot = k_y_re_b * sin_y + k_y_im_b * cos_y

        q_x_re_b, q_x_im_b = q_x_re.unsqueeze(3), q_x_im.unsqueeze(3)  # [B, h, N(q), 1, M]
        q_y_re_b, q_y_im_b = q_y_re.unsqueeze(3), q_y_im.unsqueeze(3)

        rotary_logits = (
            q_x_re_b * k_x_re_rot + q_x_im_b * k_x_im_rot
            + q_y_re_b * k_y_re_rot + q_y_im_b * k_y_im_rot
        ).sum(dim=-1)  # [B, h, N(q), N(k)]

        attn = torch.softmax(rotary_logits / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attn, v)  # [B, h, N, d]

        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return sar_tokens + self.out_proj(out)
