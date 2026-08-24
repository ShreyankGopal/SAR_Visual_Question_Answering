"""
tests/test_sar_vlm_forward.py
------------------------------
TEST 4 — SAR-VLM FORWARD PASS
TEST 5 — GRADIENT FREEZING

Uses a tiny random-weight model (no Vicuna download needed).

Test 4 verifies:
    - No shape errors during forward pass
    - logits.shape == [B, N_visual + N_text, vocab_size]
    - Labels aligned correctly (loss computable)
    - attention_mask dimensions correct

Test 5 verifies the gradient freeze contract:
    - SAR encoder gradients = None (frozen)
    - SARProjector gradients ≠ None (trainable)
    - Vicuna base weight gradients = None (frozen by PEFT)
    - LoRA adapter gradients ≠ None (trainable)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from torch import nn
from transformers import LlamaConfig

from model.hybrid_llama import HybridLlamaForCausalLM
from model.sar_projector import SARProjector
from model.sar_vlm import SARVLM, SAREncoderPlaceholder


# ---------------------------------------------------------------------------
# Tiny config — avoids downloading Vicuna, runs in seconds on CPU.
# ---------------------------------------------------------------------------
VOCAB_SIZE = 200
HIDDEN_SIZE = 64
N_LAYERS = 2
N_HEADS = 2

B = 2          # batch size
N_VISUAL = 16  # SAR visual tokens
N_TEXT = 8     # text tokens
D_SAR = 32     # tiny SAR feature dim (placeholder)


@pytest.fixture(scope="module")
def tiny_config():
    cfg = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        vocab_size=VOCAB_SIZE,
    )
    cfg._attn_implementation = "eager"
    return cfg


@pytest.fixture(scope="module")
def sarvlm(tiny_config):
    """Build a tiny SARVLM with random weights and LoRA applied."""
    encoder = SAREncoderPlaceholder(d_sar=D_SAR, n_visual=N_VISUAL)
    projector = SARProjector(d_sar=D_SAR, llm_hidden_size=HIDDEN_SIZE)
    hybrid = HybridLlamaForCausalLM(tiny_config)

    vlm = SARVLM(
        sar_encoder=encoder,
        projector=projector,
        hybrid_vicuna=hybrid,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        apply_lora=True,
    )
    return vlm


# ---------------------------------------------------------------------------
# TEST 4 — FORWARD PASS
# ---------------------------------------------------------------------------

class TestSARVLMForward:

    def test_no_exception(self, sarvlm, tiny_config):
        """Forward pass must not raise any exception."""
        sar_input = torch.zeros(B, 1, 512, 512)  # placeholder returns zeros anyway
        input_ids = torch.randint(0, VOCAB_SIZE, (B, N_TEXT))
        attention_mask = torch.ones(B, N_TEXT, dtype=torch.long)

        out = sarvlm(sar_input, input_ids, attention_mask)
        assert out is not None, "Forward returned None"

    def test_logits_shape(self, sarvlm, tiny_config):
        """logits must be [B, N_visual + N_text, vocab_size]."""
        sar_input = torch.zeros(B, 1, 512, 512)
        input_ids = torch.randint(0, VOCAB_SIZE, (B, N_TEXT))
        attention_mask = torch.ones(B, N_TEXT, dtype=torch.long)

        out = sarvlm(sar_input, input_ids, attention_mask)
        expected_seq = N_VISUAL + N_TEXT
        assert out.logits.shape == (B, expected_seq, VOCAB_SIZE), (
            f"Expected logits shape ({B}, {expected_seq}, {VOCAB_SIZE}), "
            f"got {out.logits.shape}"
        )
        print(f"\n[PASS] logits.shape = {out.logits.shape}")

    def test_loss_with_labels(self, sarvlm, tiny_config):
        """Loss should be a scalar when labels are provided."""
        sar_input = torch.zeros(B, 1, 512, 512)
        input_ids = torch.randint(0, VOCAB_SIZE, (B, N_TEXT))
        attention_mask = torch.ones(B, N_TEXT, dtype=torch.long)

        # Simulate: first half = question (-100), second half = answer tokens
        N_ANSWER = N_TEXT // 2
        labels = torch.full((B, N_TEXT), fill_value=-100)
        labels[:, -N_ANSWER:] = torch.randint(0, VOCAB_SIZE, (B, N_ANSWER))

        out = sarvlm(sar_input, input_ids, attention_mask, labels=labels)
        assert out.loss is not None, "Loss should not be None when labels are provided"
        assert out.loss.shape == (), f"Loss should be scalar, got shape {out.loss.shape}"
        assert not torch.isnan(out.loss), "Loss is NaN"
        assert not torch.isinf(out.loss), "Loss is Inf"
        print(f"\n[PASS] Loss = {out.loss.item():.4f}")

    def test_attention_mask_dimensions(self, sarvlm, tiny_config):
        """
        Internally the forward prepends N_visual ones to the attention mask.
        We can't directly check the internal mask here, but we verify that
        passing a valid text attention_mask does not raise and produces the
        right logit length.
        """
        sar_input = torch.zeros(B, 1, 512, 512)
        input_ids = torch.randint(0, VOCAB_SIZE, (B, N_TEXT))
        # Simulate padding: last 2 tokens masked
        attention_mask = torch.ones(B, N_TEXT, dtype=torch.long)
        attention_mask[:, -2:] = 0

        out = sarvlm(sar_input, input_ids, attention_mask)
        assert out.logits.shape[1] == N_VISUAL + N_TEXT, (
            f"Logit sequence length wrong: {out.logits.shape[1]}"
        )

    def test_num_visual_tokens_passed_correctly(self, sarvlm, tiny_config):
        """
        Verify that _num_visual_tokens on the inner HybridLlamaModel is set
        to N_VISUAL during the forward pass and is accessible.
        """
        sar_input = torch.zeros(B, 1, 512, 512)
        input_ids = torch.randint(0, VOCAB_SIZE, (B, N_TEXT))

        # Run forward so _num_visual_tokens gets set
        out = sarvlm(sar_input, input_ids)

        # After forward, _num_visual_tokens should still reflect the last call
        actual = sarvlm._llama_model_ref._num_visual_tokens
        assert actual == N_VISUAL, (
            f"Expected _num_visual_tokens={N_VISUAL}, got {actual}"
        )


# ---------------------------------------------------------------------------
# TEST 5 — GRADIENT FREEZING
# ---------------------------------------------------------------------------

class TestGradients:
    """
    Runs a forward + backward pass and checks gradient assignment:
        - SAR encoder: NO gradients (frozen)
        - Projector:   HAS gradients (trainable)
        - Vicuna base: NO gradients (frozen by PEFT)
        - LoRA weights: HAS gradients (trainable)
    """

    @pytest.fixture(autouse=True)
    def run_backward(self, sarvlm):
        """Run a forward+backward once for the whole class."""
        sar_input = torch.zeros(B, 1, 512, 512)
        input_ids = torch.randint(0, VOCAB_SIZE, (B, N_TEXT))
        labels = torch.randint(0, VOCAB_SIZE, (B, N_TEXT))  # all positions trainable

        out = sarvlm(sar_input, input_ids, labels=labels)
        out.loss.backward()
        yield
        # Zero grads after each test to avoid accumulation
        for p in sarvlm.parameters():
            if p.grad is not None:
                p.grad = None

    def test_encoder_has_no_gradients(self, sarvlm):
        """SAR encoder must be frozen — all .grad should be None."""
        for name, param in sarvlm.sar_encoder.named_parameters():
            assert param.grad is None, (
                f"SAR encoder parameter '{name}' has gradient — it should be frozen!\n"
                f"grad norm = {param.grad.norm().item()}"
            )
        print("\n[PASS] SAR encoder: no gradients ✓")

    def test_projector_has_gradients(self, sarvlm):
        """Projector must be trainable — at least one .grad should be non-None."""
        has_grad = [
            p.grad is not None
            for p in sarvlm.projector.parameters()
        ]
        assert any(has_grad), (
            "SARProjector has no gradients — it must be trainable!"
        )
        for name, param in sarvlm.projector.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Projector parameter '{name}' requires_grad=True but grad is None"
                )
        print("\n[PASS] SARProjector: gradients present ✓")

    def test_vicuna_base_has_no_gradients(self, sarvlm):
        """
        Original Vicuna base weights (non-LoRA) should be frozen.
        After PEFT wrapping, the original weight tensors have requires_grad=False,
        so their .grad is always None.
        """
        frozen_params = [
            (name, param)
            for name, param in sarvlm.hybrid_vicuna.named_parameters()
            if not param.requires_grad
        ]
        # There should be many frozen params (all original Vicuna weights)
        assert len(frozen_params) > 0, (
            "No frozen parameters found in hybrid_vicuna — base weights should be frozen by PEFT."
        )
        # None of the frozen params should have gradients
        for name, param in frozen_params:
            assert param.grad is None, (
                f"Frozen Vicuna parameter '{name}' received gradient — "
                f"PEFT base weight freezing may be broken."
            )
        print(f"\n[PASS] Vicuna base: {len(frozen_params)} frozen params, none have gradients ✓")

    def test_lora_has_gradients(self, sarvlm):
        """
        LoRA matrices (lora_A, lora_B) should be trainable and have gradients.
        """
        lora_params = [
            (name, param)
            for name, param in sarvlm.hybrid_vicuna.named_parameters()
            if "lora_" in name and param.requires_grad
        ]
        assert len(lora_params) > 0, (
            "No LoRA parameters found in hybrid_vicuna — "
            "check that PEFT was applied with the correct target_modules."
        )
        for name, param in lora_params:
            assert param.grad is not None, (
                f"LoRA parameter '{name}' requires_grad=True but grad is None after backward."
            )
        print(f"\n[PASS] LoRA: {len(lora_params)} trainable matrices, all have gradients ✓")
