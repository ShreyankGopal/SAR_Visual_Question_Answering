"""
tests/test_vicuna_weights.py
----------------------------
TEST 3 — VICUNA WEIGHT LOADING

Loads lmsys/vicuna-7b-v1.5 from HuggingFace (or a local path via
VICUNA_PATH env var) into HybridLlamaForCausalLM and verifies:

    1. missing_keys  == []   (no HybridLlama param not in Vicuna)
    2. unexpected_keys == [] (no Vicuna param not in HybridLlama)
    3. Spot-checked weights are bit-identical between original and hybrid.

Set VICUNA_PATH to skip the HuggingFace download:
    VICUNA_PATH=/path/to/vicuna-7b-v1.5 pytest tests/test_vicuna_weights.py

Set SKIP_VICUNA_TEST=1 to skip this test entirely (offline CI):
    SKIP_VICUNA_TEST=1 pytest tests/test_vicuna_weights.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

VICUNA_PATH = os.environ.get("VICUNA_PATH", "lmsys/vicuna-7b-v1.5")
SKIP_VICUNA = os.environ.get("SKIP_VICUNA_TEST", "0") == "1"

pytestmark = pytest.mark.skipif(
    SKIP_VICUNA,
    reason="Set SKIP_VICUNA_TEST=0 and VICUNA_PATH=<path> to run weight tests.",
)


# Spot-check these keys — they exercise different parts of the model.
SPOT_CHECK_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
]


@pytest.fixture(scope="module")
def vicuna_and_hybrid():
    """Load Vicuna once and build a hybrid model from its config + state_dict."""
    from transformers import AutoModelForCausalLM, AutoConfig
    from model.hybrid_llama import HybridLlamaForCausalLM

    print(f"\n[test_vicuna_weights] Loading Vicuna from '{VICUNA_PATH}' ...")
    try:
        vicuna = AutoModelForCausalLM.from_pretrained(
            VICUNA_PATH,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
    except Exception as e:
        pytest.skip(f"Could not load Vicuna from '{VICUNA_PATH}': {e}")

    config = vicuna.config
    config._attn_implementation = "eager"

    print("[test_vicuna_weights] Building HybridLlamaForCausalLM ...")
    hybrid = HybridLlamaForCausalLM(config)

    vicuna_state = vicuna.state_dict()
    result = hybrid.load_state_dict(vicuna_state, strict=True)

    yield vicuna, hybrid, result, vicuna_state

    del vicuna, hybrid


class TestVicunaWeights:

    def test_no_missing_keys(self, vicuna_and_hybrid):
        """Every HybridLlamaForCausalLM parameter must be loadable from Vicuna."""
        _, _, result, _ = vicuna_and_hybrid
        assert result.missing_keys == [], (
            f"Missing keys in hybrid model:\n{result.missing_keys}\n\n"
            "This means HybridLlamaForCausalLM has parameters that Vicuna does not — "
            "check that no new parameters were accidentally added."
        )

    def test_no_unexpected_keys(self, vicuna_and_hybrid):
        """Every Vicuna parameter must have a matching key in HybridLlamaForCausalLM."""
        _, _, result, _ = vicuna_and_hybrid
        assert result.unexpected_keys == [], (
            f"Unexpected keys:\n{result.unexpected_keys}\n\n"
            "This means Vicuna has parameters that HybridLlamaForCausalLM does not — "
            "check that the model structure has not been altered."
        )

    @pytest.mark.parametrize("key", SPOT_CHECK_KEYS)
    def test_weights_are_identical(self, vicuna_and_hybrid, key):
        """Spot-check that loaded weights are bit-identical to Vicuna."""
        vicuna, hybrid, _, vicuna_state = vicuna_and_hybrid

        if key not in vicuna_state:
            pytest.skip(f"Key '{key}' not found in Vicuna state_dict (may vary by checkpoint).")

        vicuna_tensor = vicuna_state[key]
        hybrid_tensor = dict(hybrid.named_parameters())[key].data

        assert vicuna_tensor.shape == hybrid_tensor.shape, (
            f"Shape mismatch for '{key}': "
            f"Vicuna {vicuna_tensor.shape} vs Hybrid {hybrid_tensor.shape}"
        )
        assert torch.equal(vicuna_tensor.cpu(), hybrid_tensor.cpu()), (
            f"Weight mismatch for '{key}' — loading did not copy values correctly."
        )
        print(f"  [OK] {key}: {tuple(vicuna_tensor.shape)}")

    def test_parameter_count_matches(self, vicuna_and_hybrid):
        """Total parameter count should be identical."""
        vicuna, hybrid, _, _ = vicuna_and_hybrid
        n_vicuna = sum(p.numel() for p in vicuna.parameters())
        n_hybrid = sum(p.numel() for p in hybrid.parameters())
        assert n_vicuna == n_hybrid, (
            f"Parameter count mismatch: Vicuna={n_vicuna:,}, Hybrid={n_hybrid:,}"
        )
        print(f"\n[PASS] Both models have {n_vicuna:,} parameters.")
