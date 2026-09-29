"""
train.py
--------
Main training loop for SAR-VLM.

Checkpointing:
    - Save every 20 training batches (CHECKPOINT_EVERY), and once more
      at the end of every epoch.

Validation:
    - Runs once, after all epochs finish: full validation pass + 2
      generated samples, logged. (Previously ran every 1000 batches
      during training -- moved to only run at the end since the
      periodic version was slow.)

Precision:
    - Model weights loaded in FP32. Forward/backward run under the
      default (FP16) torch.cuda.amp.autocast, with GradScaler handling
      loss scaling -- this matches the original, stable numeric setup
      (train2.py), not the later bf16-autocast/no-scaler experiment.

Config:
    - Config file is selectable via --config/-c (defaults to
      train_config.yaml), so different experiments can point at
      different YAML files without editing this script.

Resuming:
    - Set training.checkpoint_path in the config to the checkpoint
      directory you want to resume from. Then pass --load-checkpoint
      on the command line to actually use it; without that flag, the
      path in the config is ignored and training starts from scratch.
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import time

from model.sar_vlm import SARVLM, build_sar_encoder
from dataset import SARVLMDataset, collate_fn, count_jsonl_records
from Loss_functions.Centered_Kernel_Allign import RBFCKALoss


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log(msg: str, log_file: str = None):
    out = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(out)

    if log_file:
        with open(log_file, "a") as f:
            f.write(out + "\n")


def save_checkpoint(vlm, save_dir, global_step, log_file=None):
    """
    Save LoRA adapters and projector.

    This function is intentionally independent of validation/sampling,
    so a failure during validation cannot prevent checkpoint creation.
    """

    checkpoint_dir = os.path.join(
        save_dir,
        f"step_{global_step}"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    log(
        f"Saving checkpoint at global step {global_step} -> "
        f"{checkpoint_dir}",
        log_file
    )

    # Save LoRA adapters
    vlm.hybrid_vicuna.save_pretrained(checkpoint_dir)

    # Save SAR -> Vicuna projector
    torch.save(
        vlm.projector.state_dict(),
        os.path.join(
            checkpoint_dir,
            "projector.pth"
        )
    )

    # Save the global step so training can be resumed/debugged.
    torch.save(
        {
            "global_step": global_step
        },
        os.path.join(
            checkpoint_dir,
            "training_state.pth"
        )
    )

    log(
        f"Checkpoint saved successfully at step {global_step}.",
        log_file
    )

    return checkpoint_dir


def load_checkpoint_into_vlm(vlm, checkpoint_path, device, log_file=None):
    """
    Load a checkpoint saved by save_checkpoint() into an already-built
    SARVLM: the LoRA adapter weights and the SAR -> Vicuna projector.

    ASSUMPTION: vlm.hybrid_vicuna is a PEFT-wrapped model using the
    default adapter name "default" (PEFT's default when none is given,
    which is what save_checkpoint's `vlm.hybrid_vicuna.save_pretrained(...)`
    saves under). If SARVLM wraps LoRA some other way, this call may need
    adjusting -- this hasn't been verified against model/sar_vlm.py.

    Returns the global_step recorded in the checkpoint's
    training_state.pth (0 if that file isn't present), so the caller can
    resume the step counter for checkpoint-naming continuity.
    """

    log(f"dLoading checkpoint from: {checkpoint_path}", log_file)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint_path does not exist or is not a directory: "
            f"{checkpoint_path}"
        )

    # LoRA adapter weights.
    vlm.hybrid_vicuna.load_adapter(
        checkpoint_path,
        adapter_name="default",
        is_trainable=True,
    )
    vlm.hybrid_vicuna.set_adapter("default")

    # SAR -> Vicuna projector.
    projector_path = os.path.join(checkpoint_path, "projector.pth")

    if os.path.exists(projector_path):
        vlm.projector.load_state_dict(
            torch.load(projector_path, map_location=device)
        )
    else:
        log(
            f"WARNING: no projector.pth found at {projector_path} -- "
            f"projector weights were NOT restored from the checkpoint.",
            log_file
        )

    # Resume the global step counter, if it was recorded.
    state_path = os.path.join(checkpoint_path, "training_state.pth")
    start_step = 0

    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        start_step = state.get("global_step", 0)

    log(
        f"Checkpoint loaded. Resuming global_step counter from {start_step}.",
        log_file
    )

    return start_step


@torch.no_grad()
def run_validation(
    vlm,
    val_loader,
    device,
    log_file=None,
    intermediate_outputs=None,
    cka_loss_fn=None,
    is_CKA=False,
    lambda_CKA=0.01
):
    """
    Run validation over the complete validation loader.
    """

    vlm.eval()

    total_val_loss = 0.0
    num_batches = 0

    val_pbar = tqdm(
        val_loader,
        desc="Validation"
    )

    for batch in val_pbar:

        # IMPORTANT:
        # Keep SAR encoder input in FP32.
        sar_input = batch["sar_input"].to(
            device,
            dtype=torch.float32
        )

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Periodic CUDA cache cleanup during validation
        if num_batches % 50 == 0:
            torch.cuda.empty_cache()

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available(),
            dtype=torch.float32
        ):
            outputs = vlm(
                sar_input=sar_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        loss = outputs.loss

        total_val_loss += loss.item()
        num_batches += 1

        val_pbar.set_postfix(
            {"loss": f"{loss.item():.4f}"}
        )

        # Access intermediate outputs if hooks are active
        if intermediate_outputs is not None:
            encoder_output = intermediate_outputs['encoder_output']
            penultimate_hidden = intermediate_outputs['penultimate_hidden']
            final_hidden = intermediate_outputs['final_hidden']

            # Compute CKA loss during validation if enabled
            if is_CKA and cka_loss_fn is not None:
                # Pool encoder output: [B, N_visual, d_sar] -> [B, d_sar]
                encoder_pooled = encoder_output.mean(dim=1)  # [B, d_sar]

                # Pool LLM penultimate layer: [B, N_visual+N_text, hidden_size] -> [B, hidden_size]
                llm_penultimate_pooled = penultimate_hidden.mean(dim=1)  # [B, hidden_size]

                # Compute CKA loss between encoder and penultimate LLM layer
                val_cka_loss, val_cka_value = cka_loss_fn(encoder_pooled, llm_penultimate_pooled)

                # Log CKA statistics during validation
                if num_batches % 10 == 0:
                    log(
                        f"Validation batch {num_batches} - CKA value: {val_cka_value.item():.4f}, "
                        f"CKA loss: {val_cka_loss.item():.4f}",
                        log_file
                    )

        # Clear intermediate outputs after validation
        if intermediate_outputs is not None:
            intermediate_outputs.clear()

    if num_batches == 0:
        return 0.0

    avg_val_loss = total_val_loss / num_batches

    log(
        f"Validation Loss: {avg_val_loss:.4f}",
        log_file
    )

    # Clean up CUDA cache after validation
    torch.cuda.empty_cache()

    return avg_val_loss


@torch.no_grad()
def generate_samples(
    vlm,
    val_dataset,
    tokenizer,
    device,
    num_samples=2,
    log_file=None
):
    """
    Generate a small number of outputs from the validation set.

    Only used periodically so generation does not slow down training.
    """

    vlm.eval()

    num_samples = min(
        num_samples,
        len(val_dataset)
    )

    log(
        f"Generating {num_samples} validation samples...",
        log_file
    )

    for i in range(num_samples):

        try:
            sample = val_dataset[i]

            # Keep SAR input FP32 because the frozen SAR encoder
            # has FP32 weights.
            sar_input = sample["sar_input"].unsqueeze(0).to(
                device,
                dtype=torch.float32
            )

            # Clear CUDA cache before generation to prevent memory buildup
            torch.cuda.empty_cache()

            input_ids = sample["input_ids"]

            labels = sample["labels"]

            # Keep only the prompt tokens.
            prompt_mask = labels == -100

            prompt_ids = input_ids[
                prompt_mask
            ].unsqueeze(0).to(device)

            prompt_attention_mask = torch.ones_like(
                prompt_ids
            ).to(device)

            output_ids = vlm.generate(
                sar_input=sar_input,
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=50,
                do_sample=False
            )

            generated_text = tokenizer.decode(
                output_ids[0][prompt_ids.shape[1]:],
                skip_special_tokens=True
            )

            log(
                f"Sample {i + 1}: "
                f"{generated_text.strip()}",
                log_file
            )

            # Clean up memory after each sample
            torch.cuda.empty_cache()

        except Exception as e:

            # IMPORTANT:
            # Do not allow one bad sample to kill training.
            log(
                f"Sampling failed for sample {i + 1}: "
                f"{type(e).__name__}: {e}",
                log_file
            )

            # Clean up memory after failed sample
            torch.cuda.empty_cache()

#####
# building dataset here
#####
def build_dataset(c_data, split: str, tokenizer, max_length):
    """
    Build a (possibly multi-source) SARVLMDataset for the given split
    ("train" or "val").

    If a second dataset is configured (data.train_jsonl_2 / val_jsonl_2),
    it is randomly subsampled down to the record count of the primary
    dataset for that split, then concatenated with it. This keeps the two
    datasets balanced 1:1 rather than letting the size of the second
    dataset dominate.
    """

    primary_key = f"{split}_jsonl"
    secondary_key = f"{split}_jsonl_2"
    secondary_root_key = "data_root_2"

    primary_path = c_data[primary_key]
    primary_root = c_data["data_root"]

    sources = [
        {"jsonl_path": primary_path, "data_root": primary_root, "sample_size": None}
    ]

    secondary_path = c_data.get(secondary_key)

    if secondary_path:
        base_count = count_jsonl_records(primary_path)
        secondary_root = c_data.get(secondary_root_key, primary_root)

        sources.append(
            {
                "jsonl_path": secondary_path,
                "data_root": secondary_root,
                "sample_size": base_count,
            }
        )

    return SARVLMDataset(
        sources=sources,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=c_data.get("sample_seed", 42),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAR-VLM using a given config file."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="train_config.yaml",
        help=(
            "Path to the training config YAML file "
            "(default: train_config.yaml in the current directory). "
            "Lets you run multiple experiments by pointing each run at "
            "a different config, e.g. --config train_config_v2.yaml"
        ),
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help=(
            "Resume from the checkpoint directory given by "
            "training.checkpoint_path in the config file. The path itself "
            "always comes from the config -- this flag only controls "
            "whether it gets used. Omit this flag to start from scratch."
        ),
    )
    return parser.parse_args()


def main():

    args = parse_args()

    config = load_config(args.config)

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------

    c_data = config["data"]
    c_model = config["model"]
    c_lora = config["lora"]
    c_train = config["training"]

    os.makedirs(
        c_train["save_dir"],
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log(
        f"Starting training on device: {device}",
        c_train["log_file"]
    )

    # Clear CUDA cache at the very start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log("Initial CUDA cache cleared", c_train["log_file"])

        # Log initial memory state
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_free = (torch.cuda.get_device_properties(device).total_memory -
                      torch.cuda.memory_allocated(device)) / 1024**3
        log(
            f"Initial GPU Memory - Allocated: {memory_allocated:.2f} GB, "
            f"Reserved: {memory_reserved:.2f} GB, Free: {memory_free:.2f} GB",
            c_train["log_file"]
        )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    # Clear CUDA cache before loading tokenizer
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        c_model["vicuna_path"],
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    log(
        "Loading SAR Encoder and Hybrid Vicuna Model...",
        c_train["log_file"]
    )

    # Aggressive CUDA cache clearing before model loading
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    log("Cleared CUDA cache and garbage collected before model loading", c_train["log_file"])

    # Log memory before loading
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        log(
            f"Memory before model loading - Allocated: {memory_allocated:.2f} GB, "
            f"Reserved: {memory_reserved:.2f} GB",
            c_train["log_file"]
        )

    # Load SAR encoder first (smaller, clears cache after)
    log("Loading SAR encoder first...", c_train["log_file"])
    sar_encoder = build_sar_encoder(
        checkpoint_path=c_model["encoder_checkpoint"],
        freeze=True,
        d_sar=c_model["d_sar"]
    )

    log("SAR encoder loaded", c_train["log_file"])

    # Clear cache after encoder loading to free memory for Vicuna
    torch.cuda.empty_cache()
    gc.collect()

    # Log memory after encoder
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        log(
            f"Memory after encoder - Allocated: {memory_allocated:.2f} GB, "
            f"Reserved: {memory_reserved:.2f} GB",
            c_train["log_file"]
        )

    # Now load Vicuna with aggressive memory management
    log("Loading Vicuna model...", c_train["log_file"])

    # Get load_cpu flag from config (default to False if not specified)
    load_cpu = c_model.get("load_cpu", False)
    log(f"Vicuna loading mode: {'CPU-first' if load_cpu else 'Direct to GPU'}", c_train["log_file"])

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
        torch_dtype=torch.float32,  # Use FP32 for model stability
        load_cpu=load_cpu  # CPU-first or direct GPU loading
    )

    log("Vicuna model loaded", c_train["log_file"])

    # Clear cache after Vicuna loading
    torch.cuda.empty_cache()
    gc.collect()

    # Gradient checkpointing
    vlm.hybrid_vicuna.gradient_checkpointing_enable()

    # Clear cache before moving to device
    torch.cuda.empty_cache()
    gc.collect()

    log("Moving model to device...", c_train["log_file"])

    vlm = vlm.to(device)

    log("Model moved to device", c_train["log_file"])

    # Aggressive cache clearing after moving model to device
    torch.cuda.empty_cache()
    gc.collect()

    # Log memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_free = (torch.cuda.get_device_properties(device).total_memory -
                      torch.cuda.memory_allocated(device)) / 1024**3
        log(
            f"GPU Memory after moving - Allocated: {memory_allocated:.2f} GB, "
            f"Reserved: {memory_reserved:.2f} GB, Free: {memory_free:.2f} GB",
            c_train["log_file"]
        )

    # ---------------------------------------------------------
    # CKA Loss initialization
    # ---------------------------------------------------------

    cka_loss_fn = None
    if c_train["is_CKA"]:
        cka_loss_fn = RBFCKALoss()
        log(
            f"CKA loss enabled with lambda={c_train['lambda_CKA']}",
            c_train["log_file"]
        )

    # ---------------------------------------------------------
    # Optionally resume from a saved checkpoint
    # ---------------------------------------------------------
    #
    # --load-checkpoint (CLI flag) turns this on or off; the actual path
    # always comes from training.checkpoint_path in the config file.

    global_step = 0

    if args.load_checkpoint:

        checkpoint_path = c_train.get("checkpoint_path")

        if not checkpoint_path:
            raise ValueError(
                "--load-checkpoint was passed but training.checkpoint_path "
                "is empty in the config -- set it to a checkpoint directory."
            )

        # Clear cache before loading checkpoint
        torch.cuda.empty_cache()

        global_step = load_checkpoint_into_vlm(
            vlm=vlm,
            checkpoint_path=checkpoint_path,
            device=device,
            log_file=c_train["log_file"]
        )

        # Clear cache after loading checkpoint
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    log(
        "Loading datasets...",
        c_train["log_file"]
    )

    # Clear CUDA cache before loading datasets
    torch.cuda.empty_cache()
    gc.collect()

    # Log memory before dataset loading
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        log(
            f"Memory before datasets - Allocated: {memory_allocated:.2f} GB, "
            f"Reserved: {memory_reserved:.2f} GB",
            c_train["log_file"]
        )

    train_dataset = build_dataset(
        c_data, "train", tokenizer, c_train["max_length"]
    )

    val_dataset = build_dataset(
        c_data, "val", tokenizer, c_train["max_length"]
    )

    log(
        f"Train dataset size: {len(train_dataset)} | "
        f"Val dataset size: {len(val_dataset)}",
        c_train["log_file"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=c_train["micro_batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=c_train["micro_batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=4,
        pin_memory=True
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    trainable_params = [
        p
        for p in vlm.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(c_train["learning_rate"]),
        weight_decay=c_train["weight_decay"]
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------

    grad_acc_steps = c_train[
        "gradient_accumulation_steps"
    ]

    CHECKPOINT_EVERY = 200
    NUM_SAMPLES = 2

    # ---------------------------------------------------------
    # Setup hooks for intermediate outputs
    # ---------------------------------------------------------

    intermediate_outputs = {}
    hook_counter = [0]  # Use list to make it mutable in closure

    def encoder_hook(module, input, output):
        """Capture SAR encoder output"""
        intermediate_outputs['encoder_output'] = output.detach()

    def penultimate_hook(module, input, output):
        """Capture penultimate layer hidden states"""
        # The output is directly the hidden states tensor, not a tuple
        # It has shape [batch_size, seq_len, hidden_size]
        hidden_states = output.detach()
        intermediate_outputs['penultimate_hidden'] = hidden_states

        # Debug: log shape periodically
        hook_counter[0] += 1
        if hook_counter[0] % 10 == 0:
            log(f"Penultimate hook captured shape: {hidden_states.shape}", c_train["log_file"])

    def final_hook(module, input, output):
        """Capture final layer hidden states"""
        # The output is directly the hidden states tensor, not a tuple
        # It has shape [batch_size, seq_len, hidden_size]
        hidden_states = output.detach()
        intermediate_outputs['final_hidden'] = hidden_states

        # Debug: log shape periodically
        if hook_counter[0] % 10 == 0:
            log(f"Final hook captured shape: {hidden_states.shape}", c_train["log_file"])

    # Register encoder hook
    encoder_handle = vlm.sar_encoder.register_forward_hook(encoder_hook)

    # Register penultimate layer hook
    num_layers = vlm._llama_model_ref.config.num_hidden_layers
    penultimate_layer = vlm._llama_model_ref.layers[num_layers - 2]
    penultimate_handle = penultimate_layer.register_forward_hook(penultimate_hook)

    # Register final layer hook
    final_layer = vlm._llama_model_ref.layers[num_layers - 1]
    final_handle = final_layer.register_forward_hook(final_hook)

    log(
        "Registered hooks for encoder output, penultimate and final LLM layers",
        c_train["log_file"]
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(
        1,
        c_train["epochs"] + 1
    ):

        log(
            f"--- Epoch {epoch}/{c_train['epochs']} ---",
            c_train["log_file"]
        )

        vlm.train()

        total_train_loss = 0.0
        num_train_batches = 0

        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}"
        )

        for step, batch in enumerate(pbar):

            # -------------------------------------------------
            # Move inputs to GPU
            # -------------------------------------------------

            # IMPORTANT:
            # SAR encoder weights are FP32.
            # Do NOT explicitly convert SAR images to FP16 here.
            sar_input = batch["sar_input"].to(
                device,
                dtype=torch.float32
            )

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            # Use FP32 autocast to maintain consistency with FP32 SAR encoder
            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available(),
                dtype=torch.float32
            ):

                outputs = vlm(
                    sar_input=sar_input,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = (
                    outputs.loss /
                    grad_acc_steps
                )

            # -------------------------------------------------
            # Access intermediate outputs from hooks
            # -------------------------------------------------

            # SAR encoder output: [B, N_visual, d_sar]
            encoder_output = intermediate_outputs['encoder_output']

            # Penultimate layer hidden states: [B, N_visual+N_text, hidden_size]
            penultimate_hidden = intermediate_outputs['penultimate_hidden']

            # Final layer hidden states: [B, N_visual+N_text, hidden_size]
            final_hidden = intermediate_outputs['final_hidden']

            # Debug: Check original tensor shapes
            if global_step % 10 == 0:
                log(
                    f"Original shapes - encoder_output: {encoder_output.shape}, "
                    f"penultimate_hidden: {penultimate_hidden.shape}, "
                    f"final_hidden: {final_hidden.shape}",
                    c_train["log_file"]
                )

            # Clear intermediate outputs to free memory
            intermediate_outputs.clear()

            # -------------------------------------------------
            # CKA Loss computation (if enabled)
            # -------------------------------------------------

            cka_loss = torch.tensor(0.0, device=device, dtype=loss.dtype)
            cka_value = torch.tensor(0.0, device=device, dtype=loss.dtype)

            if c_train["is_CKA"] and cka_loss_fn is not None:
                # Pool encoder output: [B, N_visual, d_sar] -> [B, d_sar]
                # Using mean pooling over visual tokens
                encoder_pooled = encoder_output.mean(dim=1)  # [B, d_sar]

                # Pool LLM penultimate layer: [B, N_visual+N_text, hidden_size] -> [B, hidden_size]
                # Using mean pooling over all tokens
                llm_penultimate_pooled = penultimate_hidden.mean(dim=1)  # [B, hidden_size]

                # Pool LLM final layer: [B, N_visual+N_text, hidden_size] -> [B, hidden_size]
                llm_final_pooled = final_hidden.mean(dim=1)  # [B, hidden_size]

                # Debug: Check tensor shapes
                if global_step % 10 == 0:
                    log(
                        f"CKA tensor shapes - encoder_pooled: {encoder_pooled.shape}, "
                        f"llm_penultimate_pooled: {llm_penultimate_pooled.shape}, "
                        f"encoder_pooled dim: {encoder_pooled.dim()}, "
                        f"llm_penultimate_pooled dim: {llm_penultimate_pooled.dim()}",
                        c_train["log_file"]
                    )

                # Compute CKA loss between encoder and penultimate LLM layer
                cka_loss, cka_value = cka_loss_fn(encoder_pooled, llm_penultimate_pooled)

                # Optional: Also compute CKA between encoder and final LLM layer
                # You can uncomment if you want to use final layer as well
                # cka_loss_final, cka_value_final = cka_loss_fn(encoder_pooled, llm_final_pooled)
                # cka_loss = (cka_loss + cka_loss_final) / 2

            # -------------------------------------------------
            # Total loss computation
            # -------------------------------------------------

            if c_train["is_CKA"]:
                # Total loss = original loss + lambda * CKA loss
                total_loss = loss + (c_train["lambda_CKA"] * cka_loss)
            else:
                total_loss = loss

            # Example: log some statistics
            if global_step % 10 == 0:
                log(
                    f"Step {global_step} - Encoder output shape: {encoder_output.shape}, "
                    f"Penultimate hidden shape: {penultimate_hidden.shape}, "
                    f"Final hidden shape: {final_hidden.shape}",
                    c_train["log_file"]
                )
                if c_train["is_CKA"]:
                    log(
                        f"Step {global_step} - CKA value: {cka_value.item():.4f}, "
                        f"CKA loss: {cka_loss.item():.4f}, "
                        f"Total loss: {total_loss.item():.4f}",
                        c_train["log_file"]
                    )

            # Log memory usage periodically to detect memory leaks
            if global_step % 50 == 0 and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                log(
                    f"Step {global_step} - Memory: Allocated {memory_allocated:.2f} GB, "
                    f"Reserved {memory_reserved:.2f} GB",
                    c_train["log_file"]
                )

            # -------------------------------------------------
            # Backward
            # -------------------------------------------------

            scaler.scale(total_loss).backward()

            # -------------------------------------------------
            # Optimizer step
            # -------------------------------------------------

            if (
                (step + 1) % grad_acc_steps == 0
                or
                (step + 1) == len(train_loader)
            ):

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    1.0
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                # Periodic CUDA cache cleanup to prevent memory buildup
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            loss_val = (
                total_loss.item() *
                grad_acc_steps
            )

            total_train_loss += loss_val
            num_train_batches += 1

            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "step": global_step
                }
            )

            # =================================================
            # CHECKPOINT EVERY 20 BATCHES
            # =================================================

            if global_step % CHECKPOINT_EVERY == 0:

                # Save BEFORE validation/sampling.
                #
                # This is critical: even if validation crashes,
                # the trained model up to this point is already
                # safely stored.
                try:

                    save_checkpoint(
                        vlm=vlm,
                        save_dir=c_train["save_dir"],
                        global_step=global_step,
                        log_file=c_train["log_file"]
                    )

                except Exception as e:

                    log(
                        f"CHECKPOINT FAILED at step "
                        f"{global_step}: "
                        f"{type(e).__name__}: {e}",
                        c_train["log_file"]
                    )

                # Clean up CUDA cache after checkpointing
                torch.cuda.empty_cache()

            # NOTE: validation + sampling used to run here every
            # EVAL_EVERY batches. That's been moved to run once, after
            # all epochs finish (see below main training loop) since
            # running a full validation pass this often was slow.
            # Checkpointing above is untouched and still happens every
            # CHECKPOINT_EVERY batches during training.

        # -----------------------------------------------------
        # Epoch statistics
        # -----------------------------------------------------

        avg_train_loss = (
            total_train_loss /
            max(num_train_batches, 1)
        )

        log(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f}",
            c_train["log_file"]
        )

        # Clean up CUDA cache at end of each epoch
        torch.cuda.empty_cache()

        # -----------------------------------------------------
        # End-of-epoch checkpoint
        # -----------------------------------------------------

        try:

            save_checkpoint(
                vlm=vlm,
                save_dir=c_train["save_dir"],
                global_step=global_step,
                log_file=c_train["log_file"]
            )

        except Exception as e:

            log(
                f"END-OF-EPOCH CHECKPOINT FAILED: "
                f"{type(e).__name__}: {e}",
                c_train["log_file"]
            )

    # ---------------------------------------------------------
    # Final validation + sampling (all epochs complete)
    # ---------------------------------------------------------
    #
    # Moved here from inside the training loop so it runs exactly once,
    # after all training is done, instead of every EVAL_EVERY batches --
    # the periodic mid-training evaluation was taking a lot of time.

    log(
        "========== FINAL EVALUATION (all epochs complete) ==========",
        c_train["log_file"]
    )

    try:

        run_validation(
            vlm=vlm,
            val_loader=val_loader,
            device=device,
            log_file=c_train["log_file"],
            intermediate_outputs=intermediate_outputs,
            cka_loss_fn=cka_loss_fn,
            is_CKA=c_train["is_CKA"],
            lambda_CKA=c_train["lambda_CKA"]
        )

    except Exception as e:

        log(
            f"FINAL VALIDATION FAILED: {type(e).__name__}: {e}",
            c_train["log_file"]
        )

    try:

        generate_samples(
            vlm=vlm,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            num_samples=NUM_SAMPLES,
            log_file=c_train["log_file"]
        )

    except Exception as e:

        log(
            f"FINAL SAMPLING FAILED: {type(e).__name__}: {e}",
            c_train["log_file"]
        )
    # --------------
    # cleanup hooks
    # --------------
    encoder_handle.remove()
    penultimate_handle.remove()
    final_handle.remove()
    log(
        "Removed intermediate output hooks",
        c_train["log_file"]
    )

    # Final cleanup
    torch.cuda.empty_cache()
    log(
        "Training complete. CUDA cache cleared.",
        c_train["log_file"]
    ) 

    # No extra checkpoint save here -- the end-of-epoch save above for
    # the final epoch already captured the model at this global_step.


if __name__ == "__main__":
    main()
