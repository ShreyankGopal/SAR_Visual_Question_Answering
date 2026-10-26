"""
Check which linear layers in the model match the target module names.
This verifies if the apply_gated_lora function will correctly identify target modules.
"""
import gc
import torch
from model.sar_vlm import SARVLM, build_sar_encoder
from model.sar_ablation_projector import SARAblationProjector
from model.LoraGatingGSD import apply_gated_lora

def check_lora_targets():
    """Check which linear layers match target module names."""
    
    # Clear cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared before loading")
    
    # Load SAR encoder
    print("Loading SAR encoder...")
    sar_encoder = build_sar_encoder(
        checkpoint_path="/home/saishruti/Research1/Shreyank_20_credit/MyModels/SarEncPlusVicuna/mars_base_sar_encoder_only.pth",
        freeze=True,
        d_sar=1024,
    )
    
    # Clear cache after SAR encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared after SAR encoder")
    
    # Create projector
    print("Creating projector...")
    projector = SARAblationProjector(
        d_sar=1024,
        llm_hidden_size=4096,
        hidden_dim=512,
    )
    
    # Clear cache before loading Vicuna
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared before Vicuna loading")
    
    # Create a minimal model structure for testing
    print("Creating minimal model structure...")
    from model.hybrid_llama import HybridLlamaForCausalLM
    from transformers import AutoConfig
    
    # Load full Vicuna to check actual module names
    print("Loading Vicuna model to check module names...")
    config = AutoConfig.from_pretrained("lmsys/vicuna-7b-v1.5")
    config._attn_implementation = "eager"
    
    # Load Vicuna (this will take time and memory)
    from transformers import AutoModelForCausalLM
    vicuna_base = AutoModelForCausalLM.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    # Clear cache after Vicuna loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared after Vicuna loading")
    
    # Create hybrid model
    hybrid_vicuna = HybridLlamaForCausalLM(config)
    result = hybrid_vicuna.load_state_dict(vicuna_base.state_dict(), strict=True)
    print(f"Vicuna loaded: {result.missing_keys} missing, {result.unexpected_keys} unexpected")
    
    # Clear cache after state dict copy
    del vicuna_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared after deleting vicuna_base")
    
    # Create SARVLM
    vlm = SARVLM(
        sar_encoder=sar_encoder,
        projector=projector,
        hybrid_vicuna=hybrid_vicuna,
    )
    
    # Clear cache after model creation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("CUDA cache cleared after model creation")
    
    # Target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    print("\n" + "="*80)
    print("Checking Linear Layers in Hybrid Vicuna")
    print("="*80)
    
    # Find all linear layers
    linear_layers = []
    for name, module in vlm.hybrid_vicuna.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = name.split('.')[-1]
            linear_layers.append((name, module_name, module.shape))
    
    print(f"\nTotal linear layers found: {len(linear_layers)}")
    
    # Check which match target modules
    print("\n--- Layers matching target modules ---")
    matched = []
    for full_name, module_name, shape in linear_layers:
        if module_name in target_modules:
            matched.append((full_name, module_name, shape))
            print(f"✓ {full_name} | {module_name} | {shape}")
    
    print(f"\nMatched layers: {len(matched)}")
    
    # Show non-matched linear layers
    print("\n--- Linear layers NOT matching target modules ---")
    non_matched = []
    for full_name, module_name, shape in linear_layers:
        if module_name not in target_modules:
            non_matched.append((full_name, module_name, shape))
            print(f"✗ {full_name} | {module_name} | {shape}")
    
    print(f"\nNon-matched layers: {len(non_matched)}")
    
    # Test the apply_gated_lora function
    print("\n" + "="*80)
    print("Testing apply_gated_lora function")
    print("="*80)
    
    try:
        gated_lora_modules = apply_gated_lora(
            model=vlm.hybrid_vicuna,
            target_modules=target_modules,
            rank=16,
            alpha=32,
            dropout=0.05,
        )
        print(f"\n✓ Successfully applied Gated LoRA to {len(gated_lora_modules)} modules")
        
        print("\n--- Applied Gated LoRA to these modules ---")
        for i, gated_lora in enumerate(gated_lora_modules):
            print(f"{i+1}. {type(gated_lora).__name__}")
            print(f"   Original layer: {gated_lora.original_layer}")
            print(f"   Gate thresholds: {gated_lora.gated_lora.gate_thresholds.data}")
            print(f"   Gate beta: {gated_lora.gated_lora.gate_beta.data}")
        
        # Clear cache after applying LoRA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("CUDA cache cleared after applying Gated LoRA")
            
    except Exception as e:
        print(f"\n✗ Error applying Gated LoRA: {e}")
        import traceback
        traceback.print_exc()
        
        # Clear cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("CUDA cache cleared after error")

if __name__ == "__main__":
    check_lora_targets()
