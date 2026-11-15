"""Tests for the optional 2D sin/cos position code in model/sar_vlm.py (no Vicuna, no GPU)."""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")   # model/sar_vlm.py imports it (training server)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.sar_vlm import SARVLM, position_scale, sincos_2d  # noqa: E402


def test_code_shape_range_and_layout():
    c = sincos_2d(256, 1024)
    assert c.shape == (256, 1024) and c.abs().max() <= 1
    # token t = row * 16 + col: first half depends only on the row, second half only on the column
    assert torch.equal(c[16 * 3 + 5, :512], c[16 * 3 + 9, :512])     # same row 3
    assert torch.equal(c[16 * 2 + 7, 512:], c[16 * 11 + 7, 512:])    # same column 7
    assert len({tuple(r.tolist()) for r in c}) == 256                # every token's code is different


def test_off_by_default_and_from_config():
    assert position_scale({}) == 0.0
    assert position_scale({"position_encoding": {"enable": False, "scale": 3}}) == 0.0
    assert position_scale({"position_encoding": {"enable": True, "scale": 0.5}}) == 0.5
    feats = torch.randn(2, 256, 1024)
    off = SimpleNamespace(pos_encoding_scale=0.0, _pos_code=None)
    assert SARVLM._add_position(off, feats) is feats                 # off: exactly unchanged


def test_on_adds_scaled_code():
    feats = torch.randn(2, 256, 1024)
    on = SimpleNamespace(pos_encoding_scale=0.5, _pos_code=None)
    out = SARVLM._add_position(on, feats)
    assert torch.allclose(out - feats, 0.5 * sincos_2d(256, 1024).expand(2, -1, -1), atol=1e-6)
