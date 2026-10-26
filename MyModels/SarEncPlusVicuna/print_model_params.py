"""
Print all parameters and their names in the SARVLM model.
"""
import torch
from model.sar_vlm import SARVLM, build_sar_encoder
from model.sar_ablation_projector import SARAblationProjector

def print_model_parameters():
    """Print all parameters with their names and shapes."""
    
    # Load SAR encoder
    print("Loading SAR encoder...")
    sar_encoder = build_sar_encoder(
        checkpoint_path="/home/saishruti/Research1/Shreyank_20_credit/MyModels/SarEncPlusVicuna/mars_base_sar_encoder_only.pth",
        freeze=True,
        d_sar=1024,
    )
    
    # Create projector
    print("Creating projector...")
    projector = SARAblationProjector(
        in_features=1024,
        out_features=4096,
        hidden_features=512,
    )
    
    # Create a minimal SARVLM instance (without loading full Vicuna)
    print("Creating SARVLM instance...")
    from model.hybrid_llama import HybridLlamaForCausalLM
    from transformers import AutoConfig
    
    # Load config only (not full model) to check parameters
    config = AutoConfig.from_pretrained("lmsys/vicuna-7b-v1.5")
    
    print("\n" + "="*80)
    print("SARVLM Model Parameters")
    print("="*80)
    
    # Print SAR encoder parameters
    print("\n--- SAR Encoder Parameters ---")
    for name, param in sar_encoder.named_parameters():
        print(f"{name}: {param.shape} | requires_grad={param.requires_grad} | numel={param.numel():,}")
    
    # Print projector parameters
    print("\n--- Projector Parameters ---")
    for name, param in projector.named_parameters():
        print(f"{name}: {param.shape} | requires_grad={param.requires_grad} | numel={param.numel():,}")
    
    # Print total parameters
    total_encoder = sum(p.numel() for p in sar_encoder.parameters())
    total_projector = sum(p.numel() for p in projector.parameters())
    
    print("\n" + "="*80)
    print("Parameter Summary")
    print("="*80)
    print(f"SAR Encoder: {total_encoder:,} parameters")
    print(f"Projector: {total_projector:,} parameters")
    print(f"Total (excluding Vicuna): {total_encoder + total_projector:,} parameters")
    print("\nNote: Vicuna parameters not included (would require loading full model)")

if __name__ == "__main__":
    print_model_parameters()
