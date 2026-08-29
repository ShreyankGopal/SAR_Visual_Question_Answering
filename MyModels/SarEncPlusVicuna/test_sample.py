import os
import re
import yaml
import torch
from transformers import AutoTokenizer
from model.sar_vlm import SARVLM, build_sar_encoder
from dataset import SARVLMDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_root):
    candidates = []

    for name in os.listdir(checkpoint_root):
        match = re.match(r"step_(\d+)$", name)

        if match:
            step = int(match.group(1))
            path = os.path.join(checkpoint_root, name)

            if os.path.isdir(path):
                candidates.append((step, path))

    if not candidates:
        raise RuntimeError(
            f"No step_* checkpoints found in {checkpoint_root}"
        )

    candidates.sort(key=lambda x: x[0])

    return candidates[-1]


def main():

    config = load_config("train_config.yaml")

    c_data = config["data"]
    c_model = config["model"]
    c_lora = config["lora"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------------------------------------
    # Find latest checkpoint
    # ---------------------------------------------------------

    checkpoint_root = "./checkpoints"

    step, checkpoint_dir = find_latest_checkpoint(
        checkpoint_root
    )

    print("=" * 70)
    print(f"Latest checkpoint: step_{step}")
    print(f"Path: {checkpoint_dir}")
    print("=" * 70)

    projector_path = os.path.join(
        checkpoint_dir,
        "projector.pth"
    )

    if not os.path.exists(projector_path):
        raise FileNotFoundError(projector_path)

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        c_model["vicuna_path"],
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # ---------------------------------------------------------
    # SAR encoder
    # ---------------------------------------------------------

    print("Loading SAR encoder...")

    sar_encoder = build_sar_encoder(
        checkpoint_path=c_model["encoder_checkpoint"],
        freeze=True,
        d_sar=c_model["d_sar"]
    )

    # ---------------------------------------------------------
    # Build full VLM
    # ---------------------------------------------------------

    print("Loading Vicuna + LoRA architecture...")

    vlm = SARVLM.from_vicuna(
        vicuna_path=c_model["vicuna_path"],
        sar_encoder=sar_encoder,
        d_sar=c_model["d_sar"],
        n_visual=c_model["n_visual"],
        lora_r=c_lora["r"],
        lora_alpha=c_lora["alpha"],
        lora_dropout=c_lora["dropout"],
        lora_target_modules=c_lora["target_modules"],
        apply_lora=True,
        torch_dtype=torch.float32
    )

    # ---------------------------------------------------------
    # Load LoRA checkpoint
    # ---------------------------------------------------------

    print("Loading LoRA checkpoint...")

    # Your training code used:
    #
    # vlm.hybrid_vicuna.save_pretrained(checkpoint_dir)
    #
    # Therefore the adapter is loaded using PEFT's
    # from_pretrained().

    from peft import PeftModel

    base_model = vlm.hybrid_vicuna

    # If from_vicuna() already applied LoRA, we need to
    # load the saved adapter weights into the existing model.
    #
    # PeftModel.from_pretrained() expects the base model,
    # so replace hybrid_vicuna with the loaded adapter model.

    vlm.hybrid_vicuna = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
        is_trainable=False
    )

    # ---------------------------------------------------------
    # Load projector
    # ---------------------------------------------------------

    print("Loading projector...")

    projector_state = torch.load(
        projector_path,
        map_location="cpu",
        weights_only=True
    )

    vlm.projector.load_state_dict(
        projector_state,
        strict=True
    )

    # ---------------------------------------------------------
    # Move to GPU
    # ---------------------------------------------------------

    vlm = vlm.to(device)
    vlm.eval()

    print("Model loaded successfully.")
    print(f"Using checkpoint: step_{step}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    val_dataset = SARVLMDataset(
        c_data["val_jsonl"],
        c_data["data_root"],
        tokenizer,
        max_length=config["training"]["max_length"]
    )

    # ---------------------------------------------------------
    # Take ONE sample
    # ---------------------------------------------------------

    sample = val_dataset[0]

    sar_input = sample["sar_input"].unsqueeze(0).to(
        device,
        dtype=torch.float32
    )

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    # ---------------------------------------------------------
    # Reconstruct prompt
    # ---------------------------------------------------------

    prompt_mask = labels == -100

    prompt_ids = input_ids[
        prompt_mask
    ].unsqueeze(0).to(device)

    prompt_attention_mask = torch.ones_like(
        prompt_ids
    ).to(device)

    print("\nPrompt:")
    print(
        tokenizer.decode(
            prompt_ids[0],
            skip_special_tokens=False
        )
    )

    # ---------------------------------------------------------
    # Generate
    # ---------------------------------------------------------

    print("\nGenerating...")

    with torch.no_grad():

        output_ids = vlm.generate(
            sar_input=sar_input,
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=50,
            do_sample=False
        )

    # ---------------------------------------------------------
    # Decode
    # ---------------------------------------------------------

    generated_ids = output_ids[0]

    print("\nRaw generated token IDs:")
    print(generated_ids.tolist())

    print("\nRaw decoded output:")
    print(
        repr(
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=False
            )
        )
    )

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )
    print("Prompt shape:", prompt_ids.shape)
    print("Output shape:", output_ids.shape)

    print("Prompt length:", prompt_ids.shape[1])
    print("Output length:", output_ids.shape[1])

    print("Raw output IDs:")
    print(output_ids[0].tolist())

    print("\n" + "=" * 70)
    print("GENERATED ANSWER:")
    print(repr(generated_text))
    print("=" * 70)


if __name__ == "__main__":
    main()