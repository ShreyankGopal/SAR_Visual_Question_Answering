"""
tests/test_georope_adapter.py
------------------------------
Standalone unit tests for GeoRoPEVisualAdapter, independent of SARVLM/Vicuna.

Covers:
    - Output shape matches input shape.
    - Gradients flow to every adapter parameter after backward.
    - GFC's mu stays inside its bounded range (1/e, e).
    - gsd_ratio=None (default) vs. an explicit all-ones ratio produce
      identical output (both mean "no calibration").
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math

import pytest
import torch

from model.georope_adapter import GeoRoPEVisualAdapter, _GeoFrequencyCalibration

B = 2
N_VISUAL = 16  # 4x4 grid
H = 32         # llm_hidden_size (tiny, for a fast test)


@pytest.fixture
def adapter():
    return GeoRoPEVisualAdapter(
        llm_hidden_size=H,
        n_visual=N_VISUAL,
        bottleneck_dim=16,
        num_heads=2,   # head_dim = 8, num_bands = 2
        gfc_hidden_dim=8,
    )


class TestShape:
    def test_output_shape_matches_input(self, adapter):
        sar_tokens = torch.randn(B, N_VISUAL, H)
        out = adapter(sar_tokens)
        assert out.shape == sar_tokens.shape

    def test_non_square_n_visual_rejected(self):
        with pytest.raises(ValueError):
            GeoRoPEVisualAdapter(llm_hidden_size=H, n_visual=15)

    def test_bottleneck_not_divisible_by_heads_rejected(self):
        with pytest.raises(ValueError):
            GeoRoPEVisualAdapter(llm_hidden_size=H, n_visual=N_VISUAL, bottleneck_dim=15, num_heads=2)

    def test_head_dim_not_divisible_by_4_rejected(self):
        # bottleneck_dim=12, num_heads=2 -> head_dim=6, not divisible by 4
        with pytest.raises(ValueError):
            GeoRoPEVisualAdapter(llm_hidden_size=H, n_visual=N_VISUAL, bottleneck_dim=12, num_heads=2)

    def test_wrong_token_count_rejected(self, adapter):
        sar_tokens = torch.randn(B, N_VISUAL + 1, H)
        with pytest.raises(ValueError):
            adapter(sar_tokens)


class TestGradients:
    def test_all_params_get_gradients(self, adapter):
        sar_tokens = torch.randn(B, N_VISUAL, H, requires_grad=True)
        out = adapter(sar_tokens)
        out.sum().backward()

        for name, param in adapter.named_parameters():
            assert param.grad is not None, f"{name} got no gradient"

    def test_input_gets_gradient(self, adapter):
        sar_tokens = torch.randn(B, N_VISUAL, H, requires_grad=True)
        out = adapter(sar_tokens)
        out.sum().backward()
        assert sar_tokens.grad is not None
        assert not torch.all(sar_tokens.grad == 0)


class TestGFCBounds:
    def test_mu_bounded(self):
        gfc = _GeoFrequencyCalibration(hidden_size=H, grid_size=4, gfc_hidden_dim=8, num_bands=2)
        tokens = torch.randn(B, N_VISUAL, H) * 10  # large-ish values to stress the bound
        mu = gfc(tokens)
        assert torch.all(mu > 1.0 / math.e - 1e-4)
        assert torch.all(mu < math.e + 1e-4)


class TestGCCIdentity:
    def test_none_equals_all_ones_ratio(self, adapter):
        torch.manual_seed(0)
        sar_tokens = torch.randn(B, N_VISUAL, H)

        adapter.eval()
        with torch.no_grad():
            out_none = adapter(sar_tokens, gsd_ratio=None)
            out_ones = adapter(sar_tokens, gsd_ratio=torch.ones(B))

        assert torch.allclose(out_none, out_ones, atol=1e-5)
