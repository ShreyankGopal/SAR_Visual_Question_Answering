"""
train.py
--------
Main training loop for SAR-VLM.

Checkpointing:
    - Save every 20 training batches (CHECKPOINT_EVERY), and once more
      at the end of every epoch.

Validation:
    - Runs once, after all epochs finish: full validation pass + 2
      generated samples, logged.

Precision:
    - Model weights loaded in FP32. Forward/backward run under the
      default (FP16) torch.cuda.amp.autocast, with GradScaler handling
      loss scaling.

Config:
    - Config file is selectable via --config/-c (defaults to
      train_config.yaml), so different experiments can point at
      different YAML files without editing this script.

Resuming:
    - Automatic: on startup, training resumes from training.checkpoint_path
      (if set in the config) or otherwise the latest step_N checkpoint under
      training.save_dir. Model weights, optimizer, GradScaler and the
      epoch/step counters are all restored. Pass --no-resume to start from
      scratch instead.
"""

import argparse
import functools
import os
import re
import time
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

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


def find_latest_checkpoint(save_dir):
    """Return the step_N checkpoint dir with the highest N under save_dir, or None."""
    if not os.path.isdir(save_dir):
        return None

    steps = []
    for name in os.listdir(save_dir):
        match = re.fullmatch(r"step_(\d+)", name)
        if match and os.path.isdir(os.path.join(save_dir, name)):
            steps.append((int(match.group(1)), name))

    return os.path.join(save_dir, max(steps)[1]) if steps else None


def save_checkpoint(vlm, optimizer, scaler, save_dir, global_step, epoch, log_file=None):
    """
    Save LoRA adapters, the projector, optimizer/scaler state, and the
    global_step/epoch counters, so training can resume exactly. Kept
    independent of validation/sampling so a failure there can't block
    checkpointing.
    """
    checkpoint_dir = os.path.join(save_dir, f"step_{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log(f"Saving checkpoint at global step {global_step} -> {checkpoint_dir}", log_file)

    vlm.hybrid_vicuna.save_pretrained(checkpoint_dir)
    torch.save(vlm.projector.state_dict(), os.path.join(checkpoint_dir, "projector.pth"))
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        os.path.join(checkpoint_dir, "training_state.pth"),
    )

    log(f"Checkpoint saved successfully at step {global_step}.", log_file)
    return checkpoint_dir


def load_model_weights(vlm, checkpoint_path, device, log_file=None):
    """Load LoRA adapter weights and the SAR -> Vicuna projector from checkpoint_path."""
    log(f"Loading model weights from: {checkpoint_path}", log_file)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint_path does not exist or is not a directory: {checkpoint_path}"
        )

    # "default" is PEFT's default adapter name -- matches what
    # save_checkpoint()'s save_pretrained() saves under.
    vlm.hybrid_vicuna.load_adapter(checkpoint_path, adapter_name="default", is_trainable=True)
    vlm.hybrid_vicuna.set_adapter("default")

    projector_path = os.path.join(checkpoint_path, "projector.pth")
    if os.path.exists(projector_path):
        vlm.projector.load_state_dict(torch.load(projector_path, map_location=device))
    else:
        log(
            f"WARNING: no projector.pth found at {projector_path} -- "
            f"projector weights were NOT restored from the checkpoint.",
            log_file,
        )


def load_training_state(optimizer, scaler, checkpoint_path, log_file=None):
    """
    Load optimizer/scaler state and the global_step/epoch counters from
    checkpoint_path. Must be called after optimizer/scaler are constructed.
    Returns (global_step, start_epoch), defaulting to (0, 1) if no
    training_state.pth is found (e.g. an older checkpoint).

    Note: start_epoch is the epoch to (re-)run next -- a mid-epoch checkpoint
    restarts that same epoch (no per-batch dataloader position is saved), an
    end-of-epoch checkpoint moves on to the next one.
    """
    state_path = os.path.join(checkpoint_path, "training_state.pth")
    if not os.path.exists(state_path):
        log(
            f"WARNING: no training_state.pth found at {state_path} -- "
            f"resuming with a fresh optimizer and epoch counter.",
            log_file,
        )
        return 0, 1

    state = torch.load(state_path, map_location="cpu")

    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    global_step = state.get("global_step", 0)
    start_epoch = state.get("epoch", 1)

    log(
        f"Checkpoint loaded. Resuming from epoch {start_epoch}, global_step {global_step}.",
        log_file,
    )
    return global_step, start_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAR-VLM using a given config file.")
    parser.add_argument(
        "--config", "-c", type=str, default="train_config.yaml",
        help=(
            "Path to the training config YAML file (default: train_config.yaml "
            "in the current directory). Lets you run multiple experiments by "
            "pointing each run at a different config, e.g. --config train_config_v2.yaml"
        ),
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help=(
            "Start from scratch even if a checkpoint is found. By default, "
            "training auto-resumes from training.checkpoint_path in the config "
            "(if set) or otherwise the latest checkpoint under training.save_dir."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def run_validation(vlm, val_loader, device, log_file=None):
    """Run one full pass over val_loader, returning the average loss."""
    vlm.eval()

    total_val_loss = 0.0
    num_batches = 0
    val_pbar = tqdm(val_loader, desc="Validation")

    for batch in val_pbar:
        # sar_input stays FP32 (the frozen SAR encoder has FP32 weights);
        # autocast still mixes precision for the rest of the forward pass.
        sar_input = batch["sar_input"].to(device, dtype=torch.float32)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = vlm(
                sar_input=sar_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = outputs.loss
        total_val_loss += loss.item()
        num_batches += 1
        val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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
    log(f"Validation Loss: {avg_val_loss:.4f}", log_file)
    return avg_val_loss


@torch.no_grad()
def generate_samples(vlm, val_dataset, tokenizer, device, num_samples=2, log_file=None):
    """Generate a few sample outputs from the validation set (spot-checking, not full eval)."""
    vlm.eval()
    num_samples = min(num_samples, len(val_dataset))
    log(f"Generating {num_samples} validation samples...", log_file)

    for i in range(num_samples):
        try:
            sample = val_dataset[i]

            # Keep SAR input FP32 -- the frozen SAR encoder has FP32 weights.
            sar_input = sample["sar_input"].unsqueeze(0).to(device, dtype=torch.float32)

            # Clear CUDA cache before generation to prevent memory buildup
            torch.cuda.empty_cache()

            input_ids = sample["input_ids"]
            labels = sample["labels"]

            # Keep only the prompt tokens (labels == -100 marks the prompt).
            prompt_mask = labels == -100
            prompt_ids = input_ids[prompt_mask].unsqueeze(0).to(device)
            prompt_attention_mask = torch.ones_like(prompt_ids).to(device)

            output_ids = vlm.generate(
                sar_input=sar_input,
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=50,
                do_sample=False,
            )

            generated_text = tokenizer.decode(
                output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True
            )
            log(f"Sample {i + 1}: {generated_text.strip()}", log_file)

            # Clean up memory after each sample
            torch.cuda.empty_cache()

        except Exception as e:
            # Don't let one bad sample kill training.
            log(f"Sampling failed for sample {i + 1}: {type(e).__name__}: {e}", log_file)

            # Clean up memory after failed sample
            torch.cuda.empty_cache()

#####
# building dataset here
#####
def build_dataset(c_data, split: str, tokenizer, max_length):
    """
    Build a SARVLMDataset for the given split ("train"/"val"). If a second
    dataset is configured (data.{split}_jsonl_2), it's subsampled down to the
    primary dataset's record count and concatenated with it 1:1, so the two
    stay balanced instead of the second dominating by size.
    """
    # TODO: hardcoded to 2 sources; generalize to N sources if a 3rd
    # dataset actually shows up.
    primary_key = f"{split}_jsonl"
    secondary_key = f"{split}_jsonl_2"
    secondary_root_key = "data_root_2"

    primary_path = c_data[primary_key]
    primary_root = c_data["data_root"]

    sources = [{"jsonl_path": primary_path, "data_root": primary_root, "sample_size": None}]

    secondary_path = c_data.get(secondary_key)
    if secondary_path:
        base_count = count_jsonl_records(primary_path)
        secondary_root = c_data.get(secondary_root_key, primary_root)
        sources.append({
            "jsonl_path": secondary_path,
            "data_root": secondary_root,
            "sample_size": base_count,
        })

    return SARVLMDataset(
        sources=sources,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=c_data.get("sample_seed", 42),
    )


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

    os.makedirs(c_train["save_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting training on device: {device}", c_train["log_file"])

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
    tokenizer = AutoTokenizer.from_pretrained(c_model["vicuna_path"], use_fast=False)

    if tokenizer.pad_token is None:
        # LLaMA/Vicuna tokenizers ship with no pad_token; reuse unk_token
        # instead of adding a new special token (which would require
        # resizing the model's embedding matrix). Safe because padded
        # positions are excluded via attention_mask when computing loss.
        tokenizer.pad_token = tokenizer.unk_token

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    log("Loading SAR Encoder and Hybrid Vicuna Model...", c_train["log_file"])

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
        d_sar=c_model["d_sar"],
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
        torch_dtype=torch.float32,
    )

    vlm.hybrid_vicuna.gradient_checkpointing_enable()
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
    # Resume: locate a checkpoint (unless --no-resume) and restore weights
    # ---------------------------------------------------------
    # Explicit training.checkpoint_path wins; otherwise auto-pick the latest
    # step_N checkpoint under training.save_dir. Optimizer/scaler state is
    # restored later, once they exist.
    resume_path = None
    if not args.no_resume:
        resume_path = c_train.get("checkpoint_path") or find_latest_checkpoint(c_train["save_dir"])

    if resume_path:
        load_model_weights(vlm, resume_path, device, c_train["log_file"])

        # Clear cache after loading checkpoint
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    log("Loading datasets...", c_train["log_file"])

    train_dataset = build_dataset(c_data, "train", tokenizer, c_train["max_length"])
    val_dataset = build_dataset(c_data, "val", tokenizer, c_train["max_length"])

    log(
        f"Train dataset size: {len(train_dataset)} | Val dataset size: {len(val_dataset)}",
        c_train["log_file"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=c_train["micro_batch_size"],
        shuffle=True,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=c_train["micro_batch_size"],
        shuffle=False,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
        num_workers=4,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    trainable_params = [p for p in vlm.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(c_train["learning_rate"]),
        weight_decay=c_train["weight_decay"],
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    global_step, start_epoch = (0, 1)
    if resume_path:
        global_step, start_epoch = load_training_state(
            optimizer, scaler, resume_path, c_train["log_file"]
        )

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------
    grad_acc_steps = c_train["gradient_accumulation_steps"]
    CHECKPOINT_EVERY = c_train.get("checkpoint_every", 20)
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
    for epoch in range(start_epoch, c_train["epochs"] + 1):
        log(f"--- Epoch {epoch}/{c_train['epochs']} ---", c_train["log_file"])

        vlm.train()
        total_train_loss = 0.0
        num_train_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

        for step, batch in enumerate(pbar):
            sar_input = batch["sar_input"].to(device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = vlm(
                    sar_input=sar_input,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_acc_steps

            scaler.scale(total_loss).backward()

            if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = loss.item() * grad_acc_steps
            total_train_loss += loss_val
            num_train_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "step": global_step})

            if global_step % CHECKPOINT_EVERY == 0:
                # Save BEFORE validation/sampling. epoch=epoch (not epoch+1):
                # this is a mid-epoch checkpoint, so resuming re-runs this epoch.
                try:
                    save_checkpoint(
                        vlm=vlm,
                        optimizer=optimizer,
                        scaler=scaler,
                        save_dir=c_train["save_dir"],
                        global_step=global_step,
                        epoch=epoch,
                        log_file=c_train["log_file"],
                    )
                except Exception as e:
                    log(
                        f"CHECKPOINT FAILED at step {global_step}: {type(e).__name__}: {e}",
                        c_train["log_file"],
                    )

        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        log(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}", c_train["log_file"])

        # End-of-epoch checkpoint. epoch=epoch+1: this epoch is fully done,
        # so resuming should move on to the next one.
        try:
            save_checkpoint(
                vlm=vlm,
                optimizer=optimizer,
                scaler=scaler,
                save_dir=c_train["save_dir"],
                global_step=global_step,
                epoch=epoch + 1,
                log_file=c_train["log_file"],
            )
        except Exception as e:
            log(f"END-OF-EPOCH CHECKPOINT FAILED: {type(e).__name__}: {e}", c_train["log_file"])

    # Final validation + sampling (all epochs complete)
    log("========== FINAL EVALUATION (all epochs complete) ==========", c_train["log_file"])

    try:
        run_validation(vlm=vlm, val_loader=val_loader, device=device, log_file=c_train["log_file"])
    except Exception as e:
        log(f"FINAL VALIDATION FAILED: {type(e).__name__}: {e}", c_train["log_file"])

    try:
        generate_samples(
            vlm=vlm,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            num_samples=NUM_SAMPLES,
            log_file=c_train["log_file"],
        )
    except Exception as e:
        log(f"FINAL SAMPLING FAILED: {type(e).__name__}: {e}", c_train["log_file"])


if __name__ == "__main__":
    main()
