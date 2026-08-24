"""
tests/test_hybrid_mask.py
-------------------------
TEST 1 — PREFILL MASK

Verifies that with N_visual=3 visual tokens and N_text=3 text tokens the
create_causal_mask + make_sar_visual_mask combination produces the exact
6×6 attention pattern:

    SAR→SAR : 1 1 1  (bidirectional)
    SAR→txt : 0 0 0  (SAR tokens cannot see future text)
    txt→SAR : 1 1 1  (text tokens always see visual prefix)
    txt→txt : causal lower-triangular

In terms of the float mask (0 = attend, -inf = block):
    Row V1: [0, 0, 0, -inf, -inf, -inf]
    Row V2: [0, 0, 0, -inf, -inf, -inf]
    Row V3: [0, 0, 0, -inf, -inf, -inf]
    Row T1: [0, 0, 0, 0, -inf, -inf]
    Row T2: [0, 0, 0, 0, 0, -inf]
    Row T3: [0, 0, 0, 0, 0, 0]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from transformers import LlamaConfig
from transformers.masking_utils import create_causal_mask

from model.hybrid_llama import make_sar_visual_mask


N_VISUAL = 3
N_TEXT = 3
TOTAL = N_VISUAL + N_TEXT
HIDDEN_SIZE = 64  # tiny for speed; mask shape is independent of hidden size


@pytest.fixture
def eager_config():
    """Tiny LlamaConfig with eager attention (required for float-mask output)."""
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


def build_prefill_mask(config):
    """Call create_causal_mask for the full prefill (no cache)."""
    inputs_embeds = torch.zeros(1, TOTAL, HIDDEN_SIZE)
    cache_position = torch.arange(TOTAL)
    visual_mask_fn = make_sar_visual_mask(N_VISUAL)

    mask = create_causal_mask(
        config=config,
        input_embeds=inputs_embeds,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=cache_position.unsqueeze(0),
        or_mask_function=visual_mask_fn,
    )
    return mask


class TestPrefillMask:

    def test_mask_shape(self, eager_config):
        """Mask should be [1, 1, TOTAL, TOTAL] for eager attention."""
        mask = build_prefill_mask(eager_config)
        assert mask is not None, "create_causal_mask returned None for eager mode"
        assert mask.shape == (1, 1, TOTAL, TOTAL), (
            f"Expected shape (1, 1, {TOTAL}, {TOTAL}), got {mask.shape}"
        )

    def test_sar_block_is_bidirectional(self, eager_config):
        """All entries in the SAR×SAR sub-block should be 0 (attend)."""
        mask = build_prefill_mask(eager_config)
        sar_block = mask[0, 0, :N_VISUAL, :N_VISUAL]
        assert torch.all(sar_block == 0), (
            f"SAR×SAR block should be all 0 (attend), got:\n{sar_block}"
        )

    def test_sar_cannot_see_text(self, eager_config):
        """SAR rows should have large negative values for all text columns."""
        mask = build_prefill_mask(eager_config)
        sar_to_text = mask[0, 0, :N_VISUAL, N_VISUAL:]
        assert torch.all(sar_to_text < -1e30), (
            f"SAR→text block should be large negative (blocked), got:\n{sar_to_text}"
        )

    def test_text_can_see_all_sar(self, eager_config):
        """Every text row should have 0 for all SAR columns."""
        mask = build_prefill_mask(eager_config)
        text_to_sar = mask[0, 0, N_VISUAL:, :N_VISUAL]
        assert torch.all(text_to_sar == 0), (
            f"text→SAR block should be all 0 (attend), got:\n{text_to_sar}"
        )

    def test_text_block_is_causal(self, eager_config):
        """
        The text×text sub-block should be a lower-triangular causal mask:
          - diagonal and below: 0  (attend)
          - above diagonal: < -1e30  (block)
        """
        mask = build_prefill_mask(eager_config)
        text_block = mask[0, 0, N_VISUAL:, N_VISUAL:]  # [N_TEXT, N_TEXT]

        for row in range(N_TEXT):
            # Positions up to and including `row` should be 0
            assert torch.all(text_block[row, : row + 1] == 0), (
                f"text block row {row}: causal positions should be 0, "
                f"got {text_block[row, :row+1]}"
            )
            # Positions after `row` should be blocked
            if row < N_TEXT - 1:
                assert torch.all(text_block[row, row + 1:] < -1e30), (
                    f"text block row {row}: future positions should be blocked, "
                    f"got {text_block[row, row+1:]}"
                )

    def test_full_pattern_exact(self, eager_config):
        """Verify the complete 6×6 mask against the expected binary pattern."""
        mask = build_prefill_mask(eager_config)
        m = mask[0, 0]  # [6, 6]

        # Expected: 1 = attend (0 in float mask), 0 = block (< 0)
        expected_attend = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ], dtype=torch.bool)

        actual_attend = (m == 0)  # True where mask == 0 (attend)
        assert torch.all(actual_attend == expected_attend), (
            f"Full mask pattern mismatch.\n"
            f"Expected attend:\n{expected_attend.int()}\n"
            f"Actual attend:\n{actual_attend.int()}\n"
            f"Raw mask:\n{m}"
        )
        print(f"\n[PASS] Full 6×6 attend pattern:\n{actual_attend.int()}")
