"""Tests for Benchmarking/sarlang_images.py (CPU, synthetic arrays)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Benchmarking"))
import sarlang_images as si  # noqa: E402


def test_stretch_range_and_nodata():
    x = np.linspace(1, 100, 10000).reshape(100, 100)
    x[:10] = 0                                    # no-data border
    u = si.stretch(x)
    assert u.dtype == np.uint8 and u.max() == 255 and (u[:10] == 0).all()
    assert 100 < u[10:].mean() < 155              # roughly centred


def test_spacenet_takes_first_band_channels_first():
    a = np.zeros((4, 50, 50), np.float32)
    a[0] = np.linspace(1, 50, 2500).reshape(50, 50)   # HH has the signal
    a[1:] = 7.0
    u = si.spacenet6_to_uint8(a)
    assert u.shape == (50, 50) and u.max() == 255 and u.min() == 0


def test_dfc_log_then_stretch_and_name():
    a = np.full((20, 20), 0.5, np.float32)
    a[:, 10:] = 5.0                               # 10 dB apart
    u = si.dfc2023_to_uint8(a)
    assert u[:, :10].max() < u[:, 10:].min()
    assert si.sarlang_name("SN6_Train_AOI_11_Rotterdam_SAR-Intensity_2019_x_tile_1.tif") == \
        "SN6_Train_AOI_11_Rotterdam_2019_x_tile_1.png"
