"""
tests/test_mars_adapters.py
----------------------------
Tests for model/mars_adapters.py on a small real timm SwinV2 (random weights,
no checkpoint needed, CPU, a few seconds).

Covers:
    - At init (LoRA B = 0, gate = 0) the adapted encoder's output is exactly
      the original encoder's.
    - LoRA and the GeoRoPE branch each change the output once non-zero
      (guards against timm reading qkv.weight directly and skipping LoRA).
    - The branch responds to gsd_ratio.
    - Every adapter parameter gets a non-zero gradient after one step.
    - The windowed branch's maths equals GeoRoPEVisualAdapter on one window.
    - Save / load round trip, and a mismatch raises.
    - SARVLM keeps adapter params trainable and the rest of MaRS frozen.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy

import pytest
import torch

timm = pytest.importorskip("timm")

from model.georope_adapter import GeoRoPEVisualAdapter
from model.mars_adapters import (
    MaRSBlockAdapter,
    WindowGeoRoPEBranch,
    install_mars_adapters,
    load_mars_adapters,
    mars_adapter_params,
    save_mars_adapters,
)

B = 2
IMG = 128  # -> stage grids 32, 16, 8, 4 with window 8 (shifted and unshifted blocks)


class _Encoder(torch.nn.Module):
    """Same shape contract as sar_vlm._SAREncoderWrapper, without importing transformers."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.mars_ctx = None

    def forward(self, x, gsd_ratio=None):
        if self.mars_ctx is not None:
            self.mars_ctx.gsd_ratio = gsd_ratio
        f3 = self.backbone(x)[-1]
        return f3.reshape(f3.shape[0], -1, f3.shape[-1])


def _make_encoder():
    torch.manual_seed(0)
    # drop_path_rate=0: stochastic depth would make train-mode outputs random,
    # and these tests compare outputs exactly.
    backbone = timm.create_model(
        "swinv2_tiny_window8_256", pretrained=False, features_only=True, in_chans=1,
        img_size=IMG, drop_path_rate=0.0,
    )
    # timm zero-initialises SwinV2's post-attention / post-MLP norm weights, so
    # a random-init model's attention and MLP outputs are all zero and nothing
    # inside them could affect the result. Trained weights (MaRS) are non-zero.
    for m in backbone.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
    backbone.requires_grad_(False)
    return _Encoder(backbone)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, 1, IMG, IMG)


@pytest.fixture
def gsd():
    return torch.tensor([1.0, 15.0])


def _randomise(encoder, keys):
    """Give the named adapter params non-zero values (as if trained)."""
    torch.manual_seed(2)
    with torch.no_grad():
        for n, p in mars_adapter_params(encoder).items():
            if any(k in n for k in keys):
                p.normal_(std=0.05)


class TestIdentityAtInit:
    @pytest.mark.parametrize("train_mode", [False, True])
    def test_output_equals_plain_encoder(self, x, gsd, train_mode):
        plain = _make_encoder().eval()
        with torch.no_grad():
            ref = plain(x)

        adapted = install_mars_adapters(_make_encoder(), lora=True, georope=True)
        adapted.train(train_mode)
        with torch.no_grad():
            out = adapted(x, gsd_ratio=gsd)
        assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()

    def test_disabled_is_a_noop(self):
        enc = install_mars_adapters(_make_encoder(), lora=False, georope=False)
        assert mars_adapter_params(enc) == {}
        assert not any(isinstance(m, MaRSBlockAdapter) for m in enc.modules())


class TestModulesTakeEffect:
    def test_lora_changes_output(self, x):
        enc = install_mars_adapters(_make_encoder(), lora=True, georope=False).eval()
        with torch.no_grad():
            before = enc(x)
            _randomise(enc, ("lora_B",))
            after = enc(x)
        assert not torch.allclose(before, after, atol=1e-4)

    def test_lora_only_has_no_georope_params(self):
        enc = install_mars_adapters(_make_encoder(), lora=True, georope=False)
        assert not any("geo_" in n for n in mars_adapter_params(enc))

    def test_branch_changes_output_and_uses_gsd(self, x, gsd):
        enc = install_mars_adapters(_make_encoder(), lora=False, georope=True).eval()
        with torch.no_grad():
            before = enc(x, gsd_ratio=gsd)
            _randomise(enc, ("geo_gate",))
            after = enc(x, gsd_ratio=gsd)
            other_gsd = enc(x, gsd_ratio=gsd * 3.0)
        assert not torch.allclose(before, after, atol=1e-4)
        assert not torch.allclose(after, other_gsd, atol=1e-5)

    def test_stage_selection(self):
        enc = install_mars_adapters(_make_encoder(), lora=True, georope=True, stages=[2, 3])
        names = list(mars_adapter_params(enc))
        assert names and all(("layers_2" in n or "layers_3" in n or "layers.2" in n or "layers.3" in n) for n in names)


class TestGradients:
    def test_every_adapter_param_gets_gradient(self, x, gsd):
        enc = install_mars_adapters(_make_encoder(), lora=True, georope=True).train()
        params = mars_adapter_params(enc)
        for p in params.values():
            p.requires_grad_(True)
        opt = torch.optim.SGD(params.values(), lr=0.5)

        # Step 1: only the zero-initialised gate / LoRA-B receive gradient
        # (everything behind them is multiplied by 0). Step 2: everything.
        for _ in range(2):
            opt.zero_grad()
            enc(x, gsd_ratio=gsd).pow(2).mean().backward()
            opt.step()

        dead = [n for n, p in params.items() if p.grad is None or p.grad.abs().sum() == 0]
        assert not dead, f"no gradient for: {dead[:5]} ({len(dead)} total)"

    def test_original_mars_weights_untouched(self, x, gsd):
        enc = install_mars_adapters(_make_encoder(), lora=True, georope=True).train()
        params = mars_adapter_params(enc)
        for p in params.values():
            p.requires_grad_(True)
        enc(x, gsd_ratio=gsd).pow(2).mean().backward()
        frozen_with_grad = [
            n for n, p in enc.named_parameters() if n not in params and p.grad is not None
        ]
        assert not frozen_with_grad


class TestBranchMaths:
    def test_matches_georope_adapter_on_single_window(self):
        """An unshifted 8x8 map is one window == the f3-grid adapter on an 8x8 grid."""
        torch.manual_seed(3)
        dim, ws = 32, 8
        adapter = GeoRoPEVisualAdapter(
            llm_hidden_size=dim, n_visual=ws * ws, bottleneck_dim=dim // 2,
            num_heads=4, rope_base=100.0, gfc_hidden_dim=8,
        )
        branch = WindowGeoRoPEBranch(dim, ws, 0, num_heads=4, rope_base=100.0, gfc_hidden_dim=8)
        branch.load_state_dict(adapter.state_dict())

        tokens = torch.randn(B, ws * ws, dim)
        gsd = torch.tensor([0.5, 7.0])
        with torch.no_grad():
            expected = adapter(tokens, gsd_ratio=gsd) - tokens
            got = branch(tokens.view(B, ws, ws, dim), gsd_ratio=gsd).reshape(B, ws * ws, dim)
        assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()


class TestSaveLoad:
    def test_round_trip(self, x, gsd, tmp_path):
        path = str(tmp_path / "mars_adapters.pth")
        src = install_mars_adapters(_make_encoder(), lora=True, georope=True).eval()
        _randomise(src, ("lora_", "geo_"))
        assert save_mars_adapters(src, path)

        dst = install_mars_adapters(_make_encoder(), lora=True, georope=True).eval()
        load_mars_adapters(dst, path)
        with torch.no_grad():
            assert torch.allclose(src(x, gsd_ratio=gsd), dst(x, gsd_ratio=gsd), atol=1e-6)

    def test_mismatch_raises(self, tmp_path):
        path = str(tmp_path / "mars_adapters.pth")
        save_mars_adapters(install_mars_adapters(_make_encoder(), lora=True, georope=True), path)
        lora_only = install_mars_adapters(_make_encoder(), lora=True, georope=False)
        with pytest.raises(RuntimeError, match="does not match"):
            load_mars_adapters(lora_only, path)

    def test_nothing_to_save(self, tmp_path):
        path = str(tmp_path / "mars_adapters.pth")
        assert not save_mars_adapters(_make_encoder(), path)
        assert not os.path.exists(path)


class TestSARVLMIntegration:
    """Needs transformers + peft (as on the training server)."""

    def test_freeze_contract_and_identity_loss(self, x, gsd):
        pytest.importorskip("peft")
        transformers = pytest.importorskip("transformers")
        from model.hybrid_llama import HybridLlamaForCausalLM
        from model.sar_projector import SARProjector
        from model.sar_vlm import SARVLM, _SAREncoderWrapper

        cfg = transformers.LlamaConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=200,
        )
        cfg._attn_implementation = "eager"
        torch.manual_seed(4)
        llm = HybridLlamaForCausalLM(cfg)
        backbone = _make_encoder().backbone
        d_sar = backbone.feature_info.channels()[-1]
        projector = SARProjector(d_sar=d_sar, llm_hidden_size=64)

        def build(encoder):
            return SARVLM(
                sar_encoder=encoder, projector=copy.deepcopy(projector),
                hybrid_vicuna=copy.deepcopy(llm), lora_r=4, lora_alpha=8, lora_dropout=0.0,
            )

        ids = torch.randint(0, 200, (B, 8))
        plain = build(_SAREncoderWrapper(copy.deepcopy(backbone))).eval()
        adapted_enc = install_mars_adapters(
            _SAREncoderWrapper(copy.deepcopy(backbone)), lora=True, georope=True
        )
        adapted = build(adapted_enc)

        # Freeze contract: adapter params trainable, original MaRS weights frozen.
        new = mars_adapter_params(adapted.sar_encoder)
        assert new and all(p.requires_grad for p in new.values())
        assert not any(
            p.requires_grad for n, p in adapted.sar_encoder.named_parameters() if n not in new
        )

        # Untrained adapters: same loss as the model without them.
        with torch.no_grad():
            ref = plain(x, ids, labels=ids, gsd_ratio=gsd).loss
            got = adapted.eval()(x, ids, labels=ids, gsd_ratio=gsd).loss
        assert torch.allclose(ref, got, atol=1e-5)

        # Gradients reach the encoder adapters through the LLM loss.
        adapted.train()
        adapted(x, ids, labels=ids, gsd_ratio=gsd).loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in new.values())
