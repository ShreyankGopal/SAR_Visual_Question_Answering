"""
Tests for train_stage_a.py on a small real timm SwinV2 (random weights, CPU).

    python -m pytest test_stage_a.py -q
"""
import math
import random

import pytest
import torch

timm = pytest.importorskip("timm")

from train_stage_a import (  # noqa: E402
    Predictor, install_mars_adapters, make_head, make_views, mars_adapter_params, multilook,
    pooled, random_flip_rot, scale_losses, single_dataset_batches, train_mode,
)


# --- views ------------------------------------------------------------------

def test_multilook_averages_power_not_amplitude():
    # one bright scatterer (250) next to dark sea (10): power average keeps it visible
    x = torch.tensor([[[250.0, 10.0], [10.0, 10.0]]])
    out = multilook(x, 2)
    expected = math.sqrt((250**2 + 3 * 10**2) / 4)          # ~125.3, plain average would be 70
    assert out.shape == x.shape
    assert torch.allclose(out, torch.full_like(x, expected), atol=1e-3)


def test_make_views_shapes_and_original_untouched():
    img = torch.rand(1, 64, 64) * 255
    v = make_views(img, [2, 4])
    assert v.shape == (3, 1, 64, 64)
    assert torch.equal(v[0], img)


def test_flip_rot_commutes_with_multilook():
    # augmenting before making views == views of the augmented image, so all views stay the same place
    img = torch.rand(1, 32, 32) * 255
    random.seed(3)
    a = make_views(random_flip_rot(img), [2, 4])
    random.seed(3)
    b = random_flip_rot(make_views(img, [2, 4]))
    assert torch.allclose(a, b, atol=1e-4)


# --- batches ------------------------------------------------------------------

def test_single_dataset_batches_are_pure_and_proportional():
    items = [("big", f"b{i}") for i in range(160)] + [("small", f"s{i}") for i in range(32)]
    batches = single_dataset_batches(items, 16, random.Random(0))
    assert all(len(b) == 16 for b in batches)
    assert all(len({items[i][0] for i in b}) == 1 for b in batches)
    counts = {"big": 0, "small": 0}
    for b in batches:
        counts[items[b[0]][0]] += 1
    assert counts == {"big": 10, "small": 2}


# --- loss and accuracy ----------------------------------------------------------

def test_invariant_encoder_gets_zero_term_b_accuracy():
    # if every scale gives the same embedding, Term B must NOT look solved
    torch.manual_seed(0)
    same = torch.randn(1, 8, 32).expand(3, 8, 32).contiguous()
    m = scale_losses(same, Predictor(d=32), [1, 2, 4], tau=0.1)
    assert m["acc_b"] == 0.0
    assert abs(float(m["loss_b"]) - math.log(2)) < 1e-4       # stuck at log 2, as designed


def test_losses_finite_and_differentiable():
    p = torch.randn(3, 8, 32, requires_grad=True)
    pred = Predictor(d=32)
    m = scale_losses(p, pred, [1, 2, 4], tau=0.1)
    (m["loss_a"] + m["loss_b"]).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert all(q.grad is not None for q in pred.parameters())


# --- end to end on a small SwinV2 ----------------------------------------------------

class _Encoder(torch.nn.Module):
    """Same contract as sar_vlm._SAREncoderWrapper."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone, self.mars_ctx = backbone, None

    def forward(self, x, gsd_ratio=None):
        f3 = self.backbone(x)[-1]
        return f3.reshape(f3.shape[0], -1, f3.shape[-1])


def test_training_steps_reduce_loss_and_touch_only_lora():
    torch.manual_seed(0)
    bb = timm.create_model("swinv2_tiny_window8_256", pretrained=False, features_only=True,
                           in_chans=1, img_size=64, drop_path_rate=0.1)
    for m in bb.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
    bb.requires_grad_(False)
    enc = install_mars_adapters(_Encoder(bb), lora=True, georope=False, grad_checkpointing=False)
    lora = list(mars_adapter_params(enc).values())
    for p in lora:
        p.requires_grad_(True)
    train_mode(enc)
    assert not any(m.training for m in enc.modules() if type(m).__name__ == "DropPath")

    frozen_before = {n: p.clone() for n, p in enc.named_parameters() if not p.requires_grad}
    d = bb.feature_info.channels()[-1]
    head, pred = make_head(d_in=d, d_hidden=64, d_out=32), Predictor(d=32, d_hidden=64)
    opt = torch.optim.AdamW(lora + list(head.parameters()) + list(pred.parameters()), lr=1e-3)

    imgs = torch.rand(6, 1, 64, 64) * 255
    views = torch.stack([make_views(im, [2, 4]) for im in imgs], dim=1)    # [3, 6, 1, 64, 64]
    losses = []
    for _ in range(15):
        m = scale_losses(head(pooled(enc, views)), pred, [1, 2, 4], tau=0.1)
        loss = m["loss_a"] + 0.5 * m["loss_b"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert all(torch.equal(p, frozen_before[n]) for n, p in enc.named_parameters() if n in frozen_before)
