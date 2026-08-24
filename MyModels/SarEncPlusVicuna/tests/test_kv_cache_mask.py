"""
tests/test_kv_cache_mask.py
---------------------------
TEST 2 — KV CACHE (autoregressive decode step)

Scenario:
    Prefill: V1 V2 V3 T1 T2 T3  (3 SAR visual + 3 text tokens cached)
    Decode:  generate T4 (position 6)

Expected behaviour for T4:
    T4 can attend to: V1 V2 V3 T1 T2 T3 T4  → 7 positions, all 0 (attend)
    T4 is NOT bidirectional: it cannot see T5, T6, ... (not generated yet)

We test both:
    (a) The mask value for T4's row is all-zeros (full attend to cached tokens).
    (b) The visual mask function does NOT make T4 bidirectional (the visual
        mask only fires when q_idx < N_VISUAL; T4 has q_idx=6 >= N_VISUAL=3).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from model.hybrid_llama import HybridLlamaModel, make_sar_visual_mask


N_VISUAL = 3
N_PREFILL_TEXT = 3
PREFILL_LEN = N_VISUAL + N_PREFILL_TEXT  # 6
HIDDEN_SIZE = 64


@pytest.fixture
def eager_config():
    cfg = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=100,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _build_fake_cache(config, prefill_len: int, device="cpu"):
    """
    Populate a DynamicCache by running a real prefill forward on a
    randomly-initialised HybridLlamaModel.  Returns the populated cache.
    """
    model = HybridLlamaModel(config).eval()
    model._num_visual_tokens = N_VISUAL

    with torch.no_grad():
        embeds = torch.randn(1, prefill_len, config.hidden_size)
        cache_pos = torch.arange(prefill_len)
        out = model(
            inputs_embeds=embeds,
            cache_position=cache_pos,
            use_cache=True,
        )
    return out.past_key_values


class TestKVCacheMask:

    def test_decode_mask_shape(self, eager_config):
        """
        After caching 6 tokens, generating T4 should produce a mask of
        shape [1, 1, 1, 7] — one query (T4) attending over 7 kv slots.
        """
        cache = _build_fake_cache(eager_config, PREFILL_LEN)

        inputs_embeds = torch.randn(1, 1, eager_config.hidden_size)
        cache_position = torch.tensor([PREFILL_LEN])  # position 6

        visual_mask_fn = make_sar_visual_mask(N_VISUAL)
        mask = create_causal_mask(
            config=eager_config,
            input_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=cache_position.unsqueeze(0),
            or_mask_function=visual_mask_fn,
        )
        assert mask is not None
        # Shape: [B=1, heads=1, q_len=1, kv_len=PREFILL_LEN+1]
        assert mask.shape == (1, 1, 1, PREFILL_LEN + 1), (
            f"Expected (1, 1, 1, {PREFILL_LEN + 1}), got {mask.shape}"
        )

    def test_t4_attends_to_all_cached_tokens(self, eager_config):
        """T4's mask row should be all zeros (no -inf) — full attend."""
        cache = _build_fake_cache(eager_config, PREFILL_LEN)
        inputs_embeds = torch.randn(1, 1, eager_config.hidden_size)
        cache_position = torch.tensor([PREFILL_LEN])

        visual_mask_fn = make_sar_visual_mask(N_VISUAL)
        mask = create_causal_mask(
            config=eager_config,
            input_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=cache_position.unsqueeze(0),
            or_mask_function=visual_mask_fn,
        )
        t4_row = mask[0, 0, 0, :]  # [kv_len]
        assert torch.all(t4_row == 0), (
            f"T4 should attend to all {PREFILL_LEN+1} cached tokens, "
            f"but mask row has -inf at: {(torch.isinf(t4_row)).nonzero().squeeze().tolist()}"
        )

    def test_visual_mask_does_not_make_t4_bidirectional(self, eager_config):
        """
        Sanity check: visual mask function returns False for T4's position.
        q_idx = PREFILL_LEN = 6 ≥ N_VISUAL = 3, so (q_idx < N_VISUAL) = False.
        The OR with causal mask must not change behaviour.
        """
        visual_mask_fn = make_sar_visual_mask(N_VISUAL)
        q_idx = torch.tensor(PREFILL_LEN)  # T4 index = 6

        for kv_idx in range(PREFILL_LEN + 1):
            kv = torch.tensor(kv_idx)
            result = visual_mask_fn(0, 0, q_idx, kv)
            assert not result.item(), (
                f"visual_mask should be False for q_idx={PREFILL_LEN} "
                f"(T4 is a text token, not SAR), but got True at kv_idx={kv_idx}"
            )

    def test_full_model_decode_step(self, eager_config):
        """
        End-to-end decode step: run model forward for T4 with the KV cache
        from the prefill.  Verify the output shape and that no exception is raised.
        """
        cache = _build_fake_cache(eager_config, PREFILL_LEN)
        model = HybridLlamaModel(eager_config).eval()
        # During decode, visual tokens are already in the cache.
        # Keep _num_visual_tokens set (safe — visual mask has no effect for T4).
        model._num_visual_tokens = N_VISUAL

        with torch.no_grad():
            t4_embed = torch.randn(1, 1, eager_config.hidden_size)
            cache_pos = torch.tensor([PREFILL_LEN])
            out = model(
                inputs_embeds=t4_embed,
                cache_position=cache_pos,
                past_key_values=cache,
                use_cache=True,
            )

        # Hidden state for T4: [B=1, q_len=1, hidden]
        assert out.last_hidden_state.shape == (1, 1, eager_config.hidden_size), (
            f"Unexpected decode output shape: {out.last_hidden_state.shape}"
        )
        assert out.past_key_values is not None, "KV cache should be returned"
        print(f"\n[PASS] Decode step output shape: {out.last_hidden_state.shape}")
