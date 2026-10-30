"""
Unit test for load_gated_lora_weights in Benchmarking/run_benchmark.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from model.LoraGatingGSD import GatedLoRALinear
from Benchmarking.run_benchmark import load_gated_lora_weights

CHECKPOINT_DIR = (
    "/home/saishruti/Research1/Shreyank_20_credit/MyModels/SarEncPlusVicuna/"
    "Full_Model_AR_MLP_Pretrained_Gated_Lora_Data_4OBJ/checkpoints/step_44920"
)


class DummySelfAttn(nn.Module):
    def __init__(self, in_features=4096, out_features=4096, rank=16, alpha=32):
        super().__init__()
        base_q = nn.Linear(in_features, out_features, bias=False)
        base_k = nn.Linear(in_features, out_features, bias=False)
        base_v = nn.Linear(in_features, out_features, bias=False)
        base_o = nn.Linear(in_features, out_features, bias=False)

        self.q_proj = GatedLoRALinear(base_q, rank=rank, alpha=alpha)
        self.k_proj = GatedLoRALinear(base_k, rank=rank, alpha=alpha)
        self.v_proj = GatedLoRALinear(base_v, rank=rank, alpha=alpha)
        self.o_proj = GatedLoRALinear(base_o, rank=rank, alpha=alpha)


class DummyDecoderLayer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.self_attn = DummySelfAttn()


class DummyHybridVicunaModel(nn.Module):
    def __init__(self, num_layers=32):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(i) for i in range(num_layers)])


class DummyHybridVicuna(nn.Module):
    def __init__(self, num_layers=32):
        super().__init__()
        self.model = DummyHybridVicunaModel(num_layers)


class DummySARVLM(nn.Module):
    def __init__(self, num_layers=32):
        super().__init__()
        self.hybrid_vicuna = DummyHybridVicuna(num_layers)
        self._llama_model_ref = self.hybrid_vicuna.model


def test_load_gated_lora_weights():
    print("=" * 60)
    print("Testing load_gated_lora_weights...")
    print("=" * 60)

    # 1. Instantiate DummySARVLM with 32 layers
    vlm = DummySARVLM(num_layers=32)

    # Count total GatedLoRALinear modules before load
    initial_mods = [m for m in vlm.modules() if isinstance(m, GatedLoRALinear)]
    # Unique modules (since _llama_model_ref aliases self.hybrid_vicuna.model)
    unique_initial_mods = {id(m): m for m in initial_mods}
    print(f"Total unique GatedLoRALinear modules in model: {len(unique_initial_mods)}")
    assert len(unique_initial_mods) == 128, f"Expected 128 modules, got {len(unique_initial_mods)}"

    # Check weights before load (e.g. lora_B is initialized to 0)
    sample_mod = vlm.hybrid_vicuna.model.layers[0].self_attn.q_proj
    assert torch.all(sample_mod.gated_lora.lora_B == 0), "Initial lora_B should be 0"

    # 2. Load weights
    load_gated_lora_weights(vlm, CHECKPOINT_DIR)

    # 3. Verify against raw safetensors file
    safetensor_file = Path(CHECKPOINT_DIR) / "adapter_model.safetensors"
    raw_state = load_file(str(safetensor_file))

    for layer_idx in [0, 15, 31]:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mod = getattr(vlm.hybrid_vicuna.model.layers[layer_idx].self_attn, proj)
            prefix = f"_llama_model_ref.layers.{layer_idx}.self_attn.{proj}."

            # Verify lora_A (safetensors [16, 4096] -> internal [4096, 16])
            expected_A = raw_state[prefix + "lora_A.weight"].T
            assert torch.allclose(mod.gated_lora.lora_A, expected_A), (
                f"Mismatch in {prefix}lora_A"
            )

            # Verify lora_B (safetensors [4096, 16] -> internal [16, 4096])
            expected_B = raw_state[prefix + "lora_B.weight"].T
            assert torch.allclose(mod.gated_lora.lora_B, expected_B), (
                f"Mismatch in {prefix}lora_B"
            )

            # Verify gate_thresholds
            expected_thresh = raw_state[prefix + "gate_thresholds"]
            assert torch.allclose(mod.gated_lora.gate_thresholds, expected_thresh), (
                f"Mismatch in {prefix}gate_thresholds"
            )

            # Verify gate_beta
            expected_beta = raw_state[prefix + "gate_beta"]
            assert torch.allclose(mod.gated_lora.gate_beta, expected_beta), (
                f"Mismatch in {prefix}gate_beta"
            )

    print("\n[SUCCESS] All weights matched raw safetensors tensors exactly with correct transpose orientations!")


if __name__ == "__main__":
    test_load_gated_lora_weights()
