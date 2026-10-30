"""
test_gated_lora.py
------------------
Unit and integration test for GSD-Aware Gated LoRA in SAR-VLM.

Tests the full Gated LoRA pipeline using a lightweight mini-Vicuna model:
  1. Module replacement & target layer inspection
  2. Parameter trainability (frozen base LLM + frozen MaRS encoder vs trainable LoRA + projector)
  3. GSD-dependent gating mechanics across resolutions (0.1m, 1.0m, 10.0m)
  4. End-to-end forward pass with real/placeholder MaRS encoder + small Vicuna
  5. Verification that `module.abs_gsd` is dynamically set during forward and cleanly cleared afterwards
  6. Backward pass and gradient flow through Gated LoRA parameters and MLP projector
  7. Numerical sensitivity: proof that varying GSD modulates the output logits
  8. Autoregressive generation with GSD context

Usage:
    python tests/test_gated_lora.py
"""

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import LlamaConfig

# Ensure parent directory is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.hybrid_llama import HybridLlamaForCausalLM
from model.sar_ablation_projector import SARAblationProjector
from model.sar_vlm import SARVLM, SAREncoderPlaceholder, build_sar_encoder
from model.LoraGatingGSD import GatedLoRA, GatedLoRALinear, apply_gated_lora


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------
def header(title: str):
    w = 72
    print("\n" + "=" * w)
    print(f"  {title.upper()}")
    print("=" * w)


def subheader(title: str):
    print(f"\n--- {title} ---")


def pass_fail(name: str, passed: bool, extra: str = ""):
    symbol = "✓ PASS" if passed else "✗ FAIL"
    extra_str = f" ({extra})" if extra else ""
    print(f"  [{symbol}] {name}{extra_str}")
    if not passed:
        raise AssertionError(f"Test failed: {name} {extra}")


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------
def run_all_tests():
    header("SAR-VLM Gated LoRA Verification Suite")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # =======================================================================
    # 1. Build Mini-Vicuna + MaRS Encoder + Projector
    # =======================================================================
    header("1. Model Setup (Mini-Vicuna + MaRS Encoder + Projector)")

    vocab_size = 256
    hidden_size = 4096
    intermediate_size = 256
    num_layers = 2
    num_heads = 4

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
    )
    config._attn_implementation = "eager"

    hybrid_vicuna = HybridLlamaForCausalLM(config)
    print(f"Created Mini-Vicuna: {num_layers} layers, hidden_size={hidden_size}, heads={num_heads}")

    # MaRS SAR Encoder (1024 channels)
    encoder_path = REPO_ROOT / "mars_base_sar_encoder_only.pth"
    if encoder_path.exists():
        print(f"Loading real MaRS encoder checkpoint: {encoder_path}")
        sar_encoder = build_sar_encoder(str(encoder_path), freeze=True, d_sar=1024)
    else:
        print("MaRS checkpoint not found, using SAREncoderPlaceholder for test.")
        sar_encoder = SAREncoderPlaceholder(d_sar=1024, n_visual=256)

    # Projector mapping 1024 -> 4096 (exact match to real architecture)
    projector = SARAblationProjector(
        d_sar=1024,
        llm_hidden_size=hidden_size,
        hidden_dim=4096,
    )
    print(f"Created SARAblationProjector: 1024 -> 4096 -> {hidden_size}")

    # Wire together in SARVLM with Gated LoRA
    rank = 16
    alpha = 32
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    vlm = SARVLM(
        sar_encoder=sar_encoder,
        projector=projector,
        hybrid_vicuna=hybrid_vicuna,
        lora_r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        lora_target_modules=target_modules,
        apply_gate_lora=True,
        apply_lora=False,
        georope_adapter=None,
    ).to(device)

    # =======================================================================
    # 2. Check Architecture & Parameter Status
    # =======================================================================
    header("2. Architecture & Trainability Check")

    pass_fail(
        "hasattr(vlm, 'gated_lora_mods')",
        hasattr(vlm, "gated_lora_mods") and len(vlm.gated_lora_mods) > 0,
        f"Found {len(getattr(vlm, 'gated_lora_mods', []))} gated LoRA modules",
    )

    expected_mod_count = num_layers * len(target_modules)
    pass_fail(
        f"Total GatedLoRALinear instances == {expected_mod_count}",
        len(vlm.gated_lora_mods) == expected_mod_count,
        f"{num_layers} layers x {len(target_modules)} projections = {expected_mod_count}",
    )

    # Inspect trainable parameters
    trainable_named = [(n, p) for n, p in vlm.named_parameters() if p.requires_grad]
    frozen_named = [(n, p) for n, p in vlm.named_parameters() if not p.requires_grad]

    trainable_numel = sum(p.numel() for _, p in trainable_named)
    frozen_numel = sum(p.numel() for _, p in frozen_named)
    total_numel = trainable_numel + frozen_numel

    print(f"\nParameter Summary:")
    print(f"  Trainable params : {trainable_numel:,} ({100 * trainable_numel / total_numel:.2f}%)")
    print(f"  Frozen params    : {frozen_numel:,} ({100 * frozen_numel / total_numel:.2f}%)")
    print(f"  Total params     : {total_numel:,}")

    # Check that base LLM weights are frozen
    base_llm_frozen = all(
        not p.requires_grad
        for n, p in vlm.hybrid_vicuna.named_parameters()
        if "gated_lora" not in n
    )
    pass_fail("Base Vicuna linear weights and embeddings are FROZEN", base_llm_frozen)

    # Check that Gated LoRA parameters are trainable
    lora_trainable = all(
        p.requires_grad
        for n, p in vlm.hybrid_vicuna.named_parameters()
        if "gated_lora" in n
    )
    pass_fail("Gated LoRA parameters (lora_A, lora_B, thresholds, beta) are TRAINABLE", lora_trainable)

    # Check that Projector is trainable
    projector_trainable = all(p.requires_grad for p in vlm.projector.parameters())
    pass_fail("MLP Projector parameters are TRAINABLE", projector_trainable)

    # Check that SAR Encoder is frozen
    encoder_frozen = all(not p.requires_grad for p in vlm.sar_encoder.parameters())
    pass_fail("MaRS SAR Encoder parameters are FROZEN", encoder_frozen)

    # =======================================================================
    # 3. GSD Gating Vector Behavior
    # =======================================================================
    header("3. GSD Gating Function Evaluation")
    sample_gated = vlm.gated_lora_mods[0].gated_lora

    print("Initial threshold values T_k across rank dimensions:")
    print(f"  gate_thresholds: {sample_gated.gate_thresholds.data.tolist()}")
    print(f"  gate_beta      : {sample_gated.gate_beta.item():.4f}")

    gsd_tests = [0.1, 0.5, 1.0, 3.0, 10.0]
    print("\nGating vector h(s) response for varying Ground Sample Distances (meters):")
    for gsd_val in gsd_tests:
        g = sample_gated.compute_gating(torch.tensor([gsd_val], device=device))
        mean_g = g.mean().item()
        min_g = g.min().item()
        max_g = g.max().item()
        print(f"  GSD = {gsd_val:5.1f}m (log10={math.log10(gsd_val):+.2f}) -> "
              f"Mean Gate = {mean_g:.4f}  [min={min_g:.4f}, max={max_g:.4f}]")

    # Verify batch support: [B] GSD tensor -> [B, rank] gates
    batch_gsds = torch.tensor([0.2, 5.0], device=device)
    batch_h = sample_gated.compute_gating(batch_gsds)
    pass_fail(
        "Batch GSD gating output shape",
        batch_h.shape == (2, rank),
        f"Expected (2, {rank}), got {list(batch_h.shape)}",
    )

    # =======================================================================
    # 4. Forward Pass & Temporary GSD Variable Lifecycle
    # =======================================================================
    header("4. Forward Pass & Dynamic GSD Lifecycle Check")

    batch_size = 2
    seq_len = 8
    n_visual = 256

    sar_input = torch.randn(batch_size, 1, 512, 512, device=device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    labels = input_ids.clone()
    test_gsd = torch.tensor([0.3, 2.5], device=device)

    # 4.1 Confirm abs_gsd is None before forward
    all_none_before = all(m.abs_gsd is None for m in vlm.gated_lora_mods)
    pass_fail("module.abs_gsd is None BEFORE forward pass", all_none_before)

    # 4.2 Use a forward pre-hook on a gated module to spy on abs_gsd DURING forward
    spied_gsd = []
    def spy_hook(module, args):
        spied_gsd.append(module.abs_gsd)

    hook_handle = vlm.gated_lora_mods[0].register_forward_pre_hook(spy_hook)

    # Run forward pass
    outputs = vlm(
        sar_input=sar_input,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        absolute_gsd=test_gsd,
    )
    hook_handle.remove()

    # 4.3 Verify that abs_gsd was present during execution
    pass_fail(
        "module.abs_gsd was set DURING forward execution",
        len(spied_gsd) > 0 and spied_gsd[0] is not None and torch.equal(spied_gsd[0], test_gsd),
        f"Captured GSD during forward: {spied_gsd[0]}",
    )

    # 4.4 Verify that abs_gsd is reset to None AFTER forward
    all_none_after = all(m.abs_gsd is None for m in vlm.gated_lora_mods)
    pass_fail("module.abs_gsd is reset to None AFTER forward pass (no leaks)", all_none_after)

    # 4.5 Verify logits and loss
    expected_seq_len = n_visual + seq_len
    pass_fail(
        "Output logits shape",
        outputs.logits.shape == (batch_size, expected_seq_len, vocab_size),
        f"Expected ({batch_size}, {expected_seq_len}, {vocab_size}), got {list(outputs.logits.shape)}",
    )

    loss_val = outputs.loss.item()
    pass_fail(
        "Cross-entropy loss calculation",
        not math.isnan(loss_val) and not math.isinf(loss_val) and loss_val > 0,
        f"Loss = {loss_val:.4f}",
    )

    # =======================================================================
    # 5. Backward Pass & End-to-End Gradient Flow
    # =======================================================================
    header("5. Backward Pass & Gradient Flow Verification")

    vlm.zero_grad()
    outputs.loss.backward()

    first_gated = vlm.gated_lora_mods[0].gated_lora

    # Standard LoRA initialization: B is initialized to 0, so at step 0:
    # dLoss/dB is non-zero (updating B in step 1), while dLoss/dA = 0 because B=0.
    pass_fail(
        "lora_B receives non-zero gradients at init (step 0)",
        first_gated.lora_B.grad is not None and first_gated.lora_B.grad.norm() > 0,
        f"||grad(lora_B)|| = {first_gated.lora_B.grad.norm().item():.6f}",
    )

    pass_fail(
        "Projector receives gradients from base pass at step 0",
        vlm.projector.proj[0].weight.grad is not None and vlm.projector.proj[0].weight.grad.norm() > 0,
        f"||grad(proj_w0)|| = {vlm.projector.proj[0].weight.grad.norm().item():.6f}",
    )

    # Now verify active gating backpropagation:
    # Initialize lora_B with small non-zero values (as happens after step 1 of training)
    # to verify that gradients flow through B into lora_A, gate_thresholds, and gate_beta.
    print("\nSimulating active LoRA adapter (non-zero B) to verify full gating backprop:")
    for mod in vlm.gated_lora_mods:
        nn.init.normal_(mod.gated_lora.lora_B, mean=0.0, std=0.02)

    vlm.zero_grad()
    active_outputs = vlm(
        sar_input=sar_input,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        absolute_gsd=test_gsd,
    )
    active_outputs.loss.backward()

    pass_fail(
        "lora_A receives non-zero gradients when adapter is active",
        first_gated.lora_A.grad is not None and first_gated.lora_A.grad.norm() > 0,
        f"||grad(lora_A)|| = {first_gated.lora_A.grad.norm().item():.6f}",
    )
    pass_fail(
        "lora_B receives non-zero gradients",
        first_gated.lora_B.grad is not None and first_gated.lora_B.grad.norm() > 0,
        f"||grad(lora_B)|| = {first_gated.lora_B.grad.norm().item():.6f}",
    )
    pass_fail(
        "gate_thresholds receives non-zero gradients (T_k is learnable)",
        first_gated.gate_thresholds.grad is not None and first_gated.gate_thresholds.grad.norm() > 0,
        f"||grad(gate_thresholds)|| = {first_gated.gate_thresholds.grad.norm().item():.6f}",
    )
    pass_fail(
        "gate_beta receives non-zero gradients (beta is learnable)",
        first_gated.gate_beta.grad is not None and first_gated.gate_beta.grad.abs().item() > 0,
        f"||grad(gate_beta)|| = {first_gated.gate_beta.grad.abs().item():.6f}",
    )

    # Verify that frozen base LLM weights have NO gradients
    sample_base_weight = vlm.gated_lora_mods[0].original_layer.weight
    pass_fail("Frozen base linear weight has NO gradients (grad is None)", sample_base_weight.grad is None)

    sample_embed_weight = vlm._llama_model_ref.embed_tokens.weight
    pass_fail("Frozen embedding table has NO gradients (grad is None)", sample_embed_weight.grad is None)

    # Verify that SAR Encoder has NO gradients
    encoder_no_grad = all(p.grad is None for p in vlm.sar_encoder.parameters())
    pass_fail("Frozen SAR Encoder has NO gradients (grad is None)", encoder_no_grad)

    print("\nActive Gradient Norm Summary:")
    print(f"  ||grad(lora_A)||          = {first_gated.lora_A.grad.norm().item():.6f}")
    print(f"  ||grad(lora_B)||          = {first_gated.lora_B.grad.norm().item():.6f}")
    print(f"  ||grad(gate_thresholds)|| = {first_gated.gate_thresholds.grad.norm().item():.6f}")
    print(f"  ||grad(gate_beta)||       = {first_gated.gate_beta.grad.abs().item():.6f}")
    print(f"  ||grad(projector_w0)||    = {vlm.projector.proj[0].weight.grad.norm().item():.6f}")

    # =======================================================================
    # 6. Sensitivity Check: Does changing GSD change the outputs?
    # =======================================================================
    header("6. GSD Sensitivity Check (Output Modulation)")

    vlm.eval()
    with torch.no_grad():
        out_high_res = vlm(
            sar_input=sar_input,
            input_ids=input_ids,
            attention_mask=attention_mask,
            absolute_gsd=torch.tensor([0.1, 0.1], device=device),
        ).logits

        out_low_res = vlm(
            sar_input=sar_input,
            input_ids=input_ids,
            attention_mask=attention_mask,
            absolute_gsd=torch.tensor([10.0, 10.0], device=device),
        ).logits

        out_no_gsd = vlm(
            sar_input=sar_input,
            input_ids=input_ids,
            attention_mask=attention_mask,
            absolute_gsd=None,
        ).logits

    # High res vs Low res diff
    diff_res = (out_high_res - out_low_res).abs().mean().item()
    print(f"Mean absolute difference (GSD=0.1m vs GSD=10.0m): {diff_res:.6f}")

    # With GSD vs Without GSD diff
    diff_nogsd = (out_high_res - out_no_gsd).abs().mean().item()
    print(f"Mean absolute difference (GSD=0.1m vs GSD=None) : {diff_nogsd:.6f}")

    pass_fail(
        "GSD change causes measurable shift in logits",
        diff_res > 1e-6,
        f"Diff = {diff_res:.6e}",
    )

    # =======================================================================
    # 7. Autoregressive Generation Check
    # =======================================================================
    header("7. Autoregressive Generation with GSD Context")

    prompt_ids = torch.randint(0, vocab_size, (batch_size, 4), device=device)
    prompt_mask = torch.ones((batch_size, 4), device=device)
    max_new_tokens = 5

    # Confirm module.abs_gsd before generation
    pass_fail("module.abs_gsd is None BEFORE generation", all(m.abs_gsd is None for m in vlm.gated_lora_mods))

    generated_ids = vlm.generate(
        sar_input=sar_input,
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        absolute_gsd=torch.tensor([0.5, 3.0], device=device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # When generating from inputs_embeds, HuggingFace generate() returns the new tokens of length max_new_tokens
    pass_fail(
        "Generated token IDs shape",
        generated_ids.shape == (batch_size, max_new_tokens),
        f"Expected ({batch_size}, {max_new_tokens}), got {list(generated_ids.shape)}",
    )

    pass_fail(
        "module.abs_gsd is cleanly reset to None AFTER generation",
        all(m.abs_gsd is None for m in vlm.gated_lora_mods),
    )

    print(f"Generated sample tokens: {generated_ids[0].tolist()}")

    # =======================================================================
    # All tests passed!
    # =======================================================================
    header("All Gated LoRA Tests Passed Successfully!")
    print("\nSummary of Verified Capabilities:")
    print("  1. Mini-Vicuna + MaRS Encoder + Projector correctly integrates Gated LoRA.")
    print("  2. GatedLoRALinear successfully replaces q_proj, k_proj, v_proj, o_proj.")
    print("  3. Base LLM and MaRS encoder remain strictly frozen.")
    print("  4. LoRA weights, gating thresholds, beta, and projector remain fully trainable.")
    print("  5. GSD dynamically injects into GatedLoRALinear during forward and resets to None.")
    print("  6. Gradients flow end-to-end back through gating and projector weights.")
    print("  7. GSD values actively modulate model logits.")
    print("  8. Autoregressive generation functions smoothly with GSD context.\n")


if __name__ == "__main__":
    run_all_tests()
