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
import csv
import functools
import gc
import os
import re
import shutil
import time
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from model.sar_vlm import SARVLM, build_sar_encoder
from model.mars_adapters import install_mars_adapters, load_mars_adapters, save_mars_adapters
from model.LoraGatingGSD import save_gated_lora
from dataset import SARVLMDataset, SARCaptioningDs, collate_fn, count_jsonl_records
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


LOSS_CSV_FIELDS = ["epoch", "global_step", "split", "answer_loss", "cka_loss", "cka_value", "total_loss"]


def append_loss_row(path, row):
    """
    Append one row to the losses CSV (header written when the file is new).
    Columns: answer_loss = the model's loss on answer tokens; cka_loss /
    cka_value = the CKA term (blank when CKA is off or skipped); total_loss =
    the value the training log prints for that step.
    """
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


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


def save_checkpoint(vlm, optimizer, scaler, save_dir, global_step, epoch, log_file=None, keep_last=2, only_mlp=False):
    """
    Save LoRA adapters, the projector, optimizer/scaler state, and the
    global_step/epoch counters, so training can resume exactly. Kept
    independent of validation/sampling so a failure there can't block
    checkpointing.

    keep_last: after saving, deletes all but the most recent `keep_last`
    step_N checkpoint dirs under save_dir, so disk usage doesn't grow
    unbounded over a long run.

    only_mlp: if True, only save the projector weights (no LoRA adapters).
    This is used when training only the MLP projector with frozen encoder
    and no LoRA on Vicuna.
    """
    # Never save (or prune good checkpoints for) a model whose weights went NaN/inf.
    # SystemExit, not an Exception: the training loop catches Exceptions around
    # save_checkpoint and would otherwise just keep training a broken model.
    bad = [n for n, p in vlm.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
    if bad:
        log(f"STOPPING at step {global_step}: {len(bad)} trainable tensors are NaN/inf "
            f"(e.g. {bad[:3]}). Not saving. Resume from the last checkpoint in {save_dir}.", log_file)
        raise SystemExit(1)

    checkpoint_dir = os.path.join(save_dir, f"step_{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log(f"Saving checkpoint at global step {global_step} -> {checkpoint_dir}", log_file)

    if only_mlp:
        log("only_mlp=True: saving only projector weights", log_file)
    else:
        # Check if Gated LoRA is activated
        if hasattr(vlm, 'gated_lora_mods') and vlm.gated_lora_mods:
            log("Gated LoRA activated: saving Gated LoRA adapters", log_file)
            try:
                save_gated_lora(vlm, checkpoint_dir, base_model_name="lmsys/vicuna-7b-v1.5")
                log("Gated LoRA adapters saved successfully", log_file)
            except Exception as e:
                log(f"ERROR saving Gated LoRA adapters: {e}", log_file)
                raise
        else:
            # Save standard PEFT LoRA adapters
            vlm.hybrid_vicuna.save_pretrained(checkpoint_dir)

    torch.save(vlm.projector.state_dict(), os.path.join(checkpoint_dir, "projector.pth"))
    if vlm.georope_adapter is not None:
        torch.save(
            vlm.georope_adapter.state_dict(),
            os.path.join(checkpoint_dir, "georope_adapter.pth"),
        )
    # In-MaRS LoRA / GeoRoPE branches (no-op when none are installed).
    save_mars_adapters(vlm.sar_encoder, os.path.join(checkpoint_dir, "mars_adapters.pth"))
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

    # Prune old checkpoints, keeping only the most recent `keep_last`.
    steps = []
    for name in os.listdir(save_dir):
        match = re.fullmatch(r"step_(\d+)", name)
        if match and os.path.isdir(os.path.join(save_dir, name)):
            steps.append((int(match.group(1)), name))
    steps.sort()
    for _, name in steps[:-keep_last]:
        shutil.rmtree(os.path.join(save_dir, name))
        log(f"Removed old checkpoint: {name}", log_file)

    return checkpoint_dir


def load_model_weights(vlm, checkpoint_path, device, log_file=None, only_mlp=False):
    """Load LoRA adapter weights and the SAR -> Vicuna projector from checkpoint_path."""
    log(f"Loading model weights from: {checkpoint_path}", log_file)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint_path does not exist or is not a directory: {checkpoint_path}"
        )

    # Only load LoRA adapter if only_mlp is False (i.e., LoRA was saved)
    if not only_mlp:
        # Check if adapter_config.json exists (indicates LoRA was saved)
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            # Check if it's Gated LoRA (has custom_method field)
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)

            if adapter_config.get("custom_method") == "GatedLoRA":
                log("Gated LoRA adapter found - loading logic not implemented yet, skipping", log_file)
                log("NOTE: Gated LoRA loading will be implemented separately", log_file)
            else:
                # Standard PEFT LoRA
                vlm.hybrid_vicuna.load_adapter(checkpoint_path, adapter_name="default", is_trainable=True)
                vlm.hybrid_vicuna.set_adapter("default")
                log("Standard PEFT LoRA adapter loaded", log_file)
        else:
            log("No LoRA adapter found in checkpoint (only_mlp mode or no LoRA saved)", log_file)
    else:
        log("only_mlp=True: skipping LoRA adapter loading", log_file)

    projector_path = os.path.join(checkpoint_path, "projector.pth")
    if os.path.exists(projector_path):
        vlm.projector.load_state_dict(torch.load(projector_path, map_location=device))
        log("Projector weights loaded", log_file)
    else:
        log(
            f"WARNING: no projector.pth found at {projector_path} -- "
            f"projector weights were NOT restored from the checkpoint.",
            log_file,
        )

    if vlm.georope_adapter is not None:
        georope_path = os.path.join(checkpoint_path, "georope_adapter.pth")
        if os.path.exists(georope_path):
            vlm.georope_adapter.load_state_dict(torch.load(georope_path, map_location=device))
        else:
            log(
                f"WARNING: no georope_adapter.pth found at {georope_path} -- "
                f"GeoRoPE adapter weights were NOT restored (starting from fresh init).",
                log_file,
            )

    # In-MaRS adapters: no-op if none are configured and none were saved;
    # raises if the checkpoint and the `mars:` config disagree.
    load_mars_adapters(vlm.sar_encoder, os.path.join(checkpoint_path, "mars_adapters.pth"), device)


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
    parser.add_argument(
        "--mlp-checkpoint", "--mlp_checkpoint", type=str, default=None,
        help=(
            "Path to pre-trained MLP projector checkpoint (.pth file or directory "
            "containing projector.pth) to load before training."
        ),
    )
    parser.add_argument(
        "--checkpoint-path", "--checkpoint_path", "--resume", type=str, default=None,
        help=(
            "Explicit path to a step checkpoint directory (e.g. step_11940) to resume training from."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Run 5 training steps from scratch, log the peak GPU memory, and exit "
            "without saving a checkpoint. Use it to check that a config fits on "
            "the GPU at the chosen micro_batch_size before a real launch."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def run_validation(
    vlm, val_loader, device, log_file=None,
    intermediate_outputs=None, cka_loss_fn=None, is_CKA=False, lambda_CKA=0.01,
):
    """
    Run one full pass over val_loader, returning the average loss.

    intermediate_outputs/cka_loss_fn/is_CKA/lambda_CKA: optional CKA
    diagnostics. intermediate_outputs is the same dict main()'s forward
    hooks write into; when is_CKA is True and it's populated, this logs the
    CKA value/loss between the SAR encoder output and the LLM's penultimate
    layer for each batch. Does not affect the returned validation loss.
    """
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
        gsd_ratio = batch["gsd_ratio"].to(device) if "gsd_ratio" in batch else None

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = vlm(
                sar_input=sar_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                gsd_ratio=gsd_ratio,
            )

        loss = outputs.loss
        total_val_loss += loss.item()
        num_batches += 1
        val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if intermediate_outputs:
            encoder_output = intermediate_outputs['encoder_output']
            penultimate_hidden = intermediate_outputs['penultimate_hidden']

            # CKA needs >= 2 samples; a short final batch can have just 1.
            if is_CKA and cka_loss_fn is not None and encoder_output.shape[0] >= 2:
                encoder_pooled = encoder_output.mean(dim=1)          # [B, d_sar]
                llm_penultimate_pooled = penultimate_hidden.mean(dim=1)  # [B, hidden_size]
                val_cka_loss, val_cka_value = cka_loss_fn(encoder_pooled, llm_penultimate_pooled)

                if num_batches % 10 == 0:
                    log(
                        f"Validation batch {num_batches} - CKA value: {val_cka_value.item():.4f}, "
                        f"CKA loss: {val_cka_loss.item():.4f}",
                        log_file,
                    )

            intermediate_outputs.clear()

    if num_batches == 0:
        return 0.0

    avg_val_loss = total_val_loss / num_batches
    log(f"Validation Loss: {avg_val_loss:.4f}", log_file)
    return avg_val_loss


@torch.no_grad()
def generate_samples(vlm, val_dataset, tokenizer, device, num_samples=2, log_file=None, ref_gsd_m=None):
    """Generate a few sample outputs from the validation set (spot-checking, not full eval)."""
    vlm.eval()
    num_samples = min(num_samples, len(val_dataset))
    log(f"Generating {num_samples} validation samples...", log_file)

    for i in range(num_samples):
        try:
            sample = val_dataset[i]

            # Keep SAR input FP32 -- the frozen SAR encoder has FP32 weights.
            sar_input = sample["sar_input"].unsqueeze(0).to(device, dtype=torch.float32)

            input_ids = sample["input_ids"]
            labels = sample["labels"]

            # Keep only the prompt tokens (labels == -100 marks the prompt).
            prompt_mask = labels == -100
            prompt_ids = input_ids[prompt_mask].unsqueeze(0).to(device)
            prompt_attention_mask = torch.ones_like(prompt_ids).to(device)

            gsd_ratio = None
            if ref_gsd_m is not None:
                sample_gsd_m = sample.get("gsd_m")
                
                ratio = (sample_gsd_m / ref_gsd_m) if sample_gsd_m is not None else 1.0
                gsd_ratio = torch.tensor([ratio], dtype=torch.float32, device=device)

            output_ids = vlm.generate(
                sar_input=sar_input,
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                gsd_ratio=gsd_ratio,
                absolute_gsd=sample_gsd_m,
                max_new_tokens=50,
                do_sample=False,
            )

            generated_text = tokenizer.decode(
                output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True
            )
            log(f"Sample {i + 1}: {generated_text.strip()}", log_file)

        except Exception as e:
            # Don't let one bad sample kill training.
            log(f"Sampling failed for sample {i + 1}: {type(e).__name__}: {e}", log_file)

#####
# building dataset here
#####
def build_dataset(c_data, split: str, tokenizer, max_length):
    """
    Build a dataset for the given split ("train"/"val").
    If train_caption is True, uses SARCaptioningDs for captioning data.
    Otherwise, builds a SARVLMDataset supporting any number of extra datasets
    via data.{split}_jsonl_2, _3, _4, ... (each with its own optional data_root_N,
    defaulting to data.data_root) -- stops at the first missing N.

    train: every extra source is subsampled down to the primary dataset's
    *effective* record count -- i.e. after train_sample_size caps it (e.g.
    50k/50k/50k...), or its full count if train_sample_size is null -- so no
    single source dominates by size.
    val: nothing is capped; every source contributes all of its records.
    """
    # Check if captioning dataset should be used
    train_caption = c_data.get("train_caption", False)

    if train_caption:
        # Use captioning dataset (SARCaptioningDs)
        caption_json_path = c_data.get(f"caption_{split}_json")
        caption_data_root = c_data.get("caption_data_root")

        if not caption_json_path or not caption_data_root:
            raise ValueError(
                f"train_caption is True but caption_{split}_json or caption_data_root not configured"
            )
        print("using captioing dataset")
        return SARCaptioningDs(
            json_path=caption_json_path,
            data_root=caption_data_root,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    # Use standard SARVLMDataset
    primary_path = c_data[f"{split}_jsonl"]
    primary_root = c_data["data_root"]
    primary_sample_size = c_data.get(f"{split}_sample_size") if split == "train" else None

    sources = [{"jsonl_path": primary_path, "data_root": primary_root, "sample_size": primary_sample_size}]

    base_count = None
    n = 2
    while True:
        secondary_path = c_data.get(f"{split}_jsonl_{n}")
        if not secondary_path:
            break
        if split == "train" and base_count is None:
            base_count = count_jsonl_records(primary_path)
            if primary_sample_size is not None:
                base_count = min(base_count, primary_sample_size)
        sources.append({
            "jsonl_path": secondary_path,
            "data_root": c_data.get(f"data_root_{n}", primary_root),
            "sample_size": base_count,
        })
        n += 1

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
    c_georope = config.get("georope", {})
    
    only_mlp = c_train.get("only_mlp", False)
    c_mars = config.get("mars") or {}

    os.makedirs(c_train["save_dir"], exist_ok=True)
    shutil.copy(args.config, os.path.join(c_train["save_dir"], os.path.basename(args.config)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting training on device: {device}", c_train["log_file"])

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

    # Clear cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    sar_encoder = build_sar_encoder(
        checkpoint_path=c_model["encoder_checkpoint"],
        freeze=True,
        d_sar=c_model["d_sar"],
    )
    # In-MaRS LoRA / GeoRoPE branches; a no-op unless enabled under `mars:`.
    install_mars_adapters(sar_encoder, **c_mars)

    log("SAR encoder loaded", c_train["log_file"])

    # Clear cache after SAR encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Get load_cpu flag from config (default to False if not specified)
    load_cpu = c_model.get("load_cpu", False)
    log(f"Vicuna loading mode: {'CPU-first' if load_cpu else 'Direct to GPU'}", c_train["log_file"])

    # If only_mlp is True, disable LoRA (only projector is trainable)
    if only_mlp:
        log("only_mlp=True: Disabling LoRA (only projector will be trainable)", c_train["log_file"])
        apply_lora = False
        apply_gate_lora = False
    else:
        apply_lora = c_lora.get("is_lora", True)
        apply_gate_lora = c_lora.get("is_gated_lora", False)

    vlm = SARVLM.from_vicuna(
        vicuna_path=c_model["vicuna_path"],
        sar_encoder=sar_encoder,
        d_sar=c_model["d_sar"],
        n_visual=c_model["n_visual"],
        lora_r=c_lora["r"],
        lora_alpha=c_lora["alpha"],
        lora_dropout=c_lora["dropout"],
        lora_target_modules=c_lora["target_modules"],
        apply_lora=apply_lora,
        apply_gate_lora=apply_gate_lora,
        torch_dtype=torch.float32,
        low_memory_load=True,
        load_cpu=load_cpu,
        use_georope_adapter=c_georope.get("enable", True),
        georope_bottleneck_dim=c_georope.get("bottleneck_dim", 256),
        georope_num_heads=c_georope.get("num_heads", 4),
        georope_gcc_alpha=c_georope.get("gcc_alpha", 0.5),
        georope_gfc_hidden_dim=c_georope.get("gfc_hidden_dim", 64),
        georope_zero_init_output=c_georope.get("zero_init_output", False),
    )

    vlm.hybrid_vicuna.gradient_checkpointing_enable()
    vlm = vlm.to(device)

    log("Model moved to device", c_train["log_file"])

    # ---------------------------------------------------------
    # Resume: locate a checkpoint (unless --no-resume) and restore weights
    # ---------------------------------------------------------
    # Explicit --checkpoint-path CLI argument or training.checkpoint_path in config wins;
    # otherwise auto-pick the latest step_N checkpoint under training.save_dir.
    resume_path = None
    if not args.no_resume and not args.dry_run:
        resume_path = args.checkpoint_path or c_train.get("checkpoint_path") or find_latest_checkpoint(c_train["save_dir"])

    # ---------------------------------------------------------
    # Pre-trained MLP Projector Loading (initial load for new runs)
    # ---------------------------------------------------------
    load_mlp = c_train.get("load_mlp", False) or (args.mlp_checkpoint is not None)
    mlp_checkpoint = args.mlp_checkpoint or c_train.get("mlp_checkpoint")

    if load_mlp and resume_path is None:
        if not mlp_checkpoint:
            raise ValueError(
                "load_mlp is True in config, but no MLP checkpoint was specified! "
                "Please pass --mlp-checkpoint /path/to/projector.pth (or directory) "
                "or set training.mlp_checkpoint in train_config.yaml."
            )
        file_path, missing, unexpected = vlm.load_projector_checkpoint(mlp_checkpoint)
        log(f"Pre-trained MLP projector loaded from: {file_path}", c_train["log_file"])
        if missing:
            log(f"WARNING: missing keys in MLP checkpoint: {missing}", c_train["log_file"])
        if unexpected:
            log(f"WARNING: unexpected keys in MLP checkpoint: {unexpected}", c_train["log_file"])
        if not missing and not unexpected:
            log("MLP projector weights matched and loaded successfully.", c_train["log_file"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    elif load_mlp and resume_path is not None:
        if args.mlp_checkpoint is not None:
            file_path, missing, unexpected = vlm.load_projector_checkpoint(args.mlp_checkpoint)
            log(f"Pre-trained MLP projector loaded from: {file_path}", c_train["log_file"])
        else:
            log(
                f"Resuming run from checkpoint {resume_path}; skipping separate MLP load "
                f"as projector weights are restored from the step checkpoint.",
                c_train["log_file"],
            )

    if resume_path:
        load_model_weights(vlm, resume_path, device, c_train["log_file"], only_mlp=only_mlp)
        # Clear cache after checkpoint loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            log("CUDA cache cleared after checkpoint loading", c_train["log_file"])

    trainable = 0
    total = 0

    for name, param in vlm.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            print("TRAINABLE:", name, param.numel())

    print(f"Trainable: {trainable:,} / {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.4f}%")
    # Clear CUDA cache after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log("CUDA cache cleared after model loading", c_train["log_file"])

    # ---------------------------------------------------------
    # CKA loss initialization (encoder-vs-LLM representation alignment)
    # ---------------------------------------------------------
    is_CKA = c_train.get("is_CKA", False)
    lambda_CKA = c_train.get("lambda_CKA", 0.01)
    cka_loss_fn = RBFCKALoss() if is_CKA else None
    if is_CKA:
        log(f"CKA loss enabled with lambda={lambda_CKA}", c_train["log_file"])

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    log("Loading datasets...", c_train["log_file"])

    train_caption = c_data.get("train_caption", False)
    dataset_type = "Captioning (SARCaptioningDs)" if train_caption else "VQA (SARVLMDataset)"
    log(f"Using {dataset_type}", c_train["log_file"])

    train_dataset = build_dataset(c_data, "train", tokenizer, c_train["max_length"])
    val_dataset = build_dataset(c_data, "val", tokenizer, c_train["max_length"])

    log(
        f"Train dataset size: {len(train_dataset)} | Val dataset size: {len(val_dataset)}",
        c_train["log_file"],
    )

    ref_gsd_m = c_georope.get("ref_gsd_m")

    train_loader = DataLoader(
        train_dataset,
        batch_size=c_train["micro_batch_size"],
        shuffle=True,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer, ref_gsd_m=ref_gsd_m),
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=c_train["micro_batch_size"],
        shuffle=False,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer, ref_gsd_m=ref_gsd_m),
        num_workers=4,
        pin_memory=True,
    )
    # logging numnber of samples
    log(
        f"Train dataloader: {len(train_loader)} batches ({len(train_dataset)} samples) | "
        f"Val dataloader: {len(val_loader)} batches ({len(val_dataset)} samples)",
        c_train["log_file"],
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
    # ponytail: append-only; after a resume, steps since the last checkpoint
    # appear twice (same global_step) -- keep the last occurrence when plotting.
    loss_csv = os.path.join(c_train["save_dir"], "losses.csv")
    log(f"Logging losses to {loss_csv}", c_train["log_file"])
    NUM_SAMPLES = 2

    # ---------------------------------------------------------
    # Hooks for CKA: capture SAR encoder output and the LLM's penultimate/
    # final layer hidden states during each forward pass.
    # ---------------------------------------------------------
    intermediate_outputs = {}
    encoder_handle = penultimate_handle = final_handle = None

    if is_CKA:
        def encoder_hook(module, input, output):
            intermediate_outputs['encoder_output'] = output.detach()

        def penultimate_hook(module, input, output):
            intermediate_outputs['penultimate_hidden'] = output[:256].detach()

        def final_hook(module, input, output):
            intermediate_outputs['final_hidden'] = output.detach()

        encoder_handle = vlm.sar_encoder.register_forward_hook(encoder_hook)
        num_layers = vlm._llama_model_ref.config.num_hidden_layers
        penultimate_handle = vlm._llama_model_ref.layers[num_layers - 2].register_forward_hook(penultimate_hook)
        final_handle = vlm._llama_model_ref.layers[num_layers - 1].register_forward_hook(final_hook)
        log("Registered hooks for encoder output, penultimate and final LLM layers", c_train["log_file"])

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
            gsd_ratio = batch["gsd_ratio"].to(device) if "gsd_ratio" in batch else None
            absolute_gsd = batch["absolute_gsd"].to(device) if "absolute_gsd" in batch else None

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = vlm(
                    sar_input=sar_input,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    gsd_ratio=gsd_ratio,
                    absolute_gsd = absolute_gsd
                    
                )
                loss = outputs.loss / grad_acc_steps

            cka_loss = torch.tensor(0.0, device=device, dtype=loss.dtype)
            cka_value = None
            # CKA needs >= 2 samples; a short final batch can have just 1.
            if (is_CKA and cka_loss_fn is not None and intermediate_outputs
                    and intermediate_outputs['encoder_output'].shape[0] >= 2):
                encoder_pooled = intermediate_outputs['encoder_output'].mean(dim=1)
                llm_penultimate_pooled = intermediate_outputs['penultimate_hidden'].mean(dim=1)
                cka_loss, cka_value = cka_loss_fn(encoder_pooled, llm_penultimate_pooled)
                intermediate_outputs.clear()

                if global_step % 10 == 0:
                    log(
                        f"Step {global_step} - CKA value: {cka_value.item():.4f}, "
                        f"CKA loss: {cka_loss.item():.4f}",
                        c_train["log_file"],
                    )

            total_loss = loss + (lambda_CKA * cka_loss) if is_CKA else loss

            scaler.scale(total_loss).backward()

            if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = total_loss.item() * grad_acc_steps
            total_train_loss += loss_val
            num_train_batches += 1
            global_step += 1

            if not args.dry_run:
                append_loss_row(loss_csv, {
                    "epoch": epoch,
                    "global_step": global_step,
                    "split": "train",
                    "answer_loss": outputs.loss.item(),
                    "cka_loss": cka_loss.item() if cka_value is not None else "",
                    "cka_value": cka_value.item() if cka_value is not None else "",
                    "total_loss": loss_val,
                })

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "step": global_step})

            if args.dry_run and global_step >= 5:
                if torch.cuda.is_available():
                    log(
                        f"DRY RUN: 5 steps OK at micro_batch_size={c_train['micro_batch_size']}. "
                        f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB allocated, "
                        f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved.",
                        c_train["log_file"],
                    )
                else:
                    log("DRY RUN: 5 steps OK (no GPU, so no memory figure).", c_train["log_file"])
                return

            # Periodic cache clearing during training
            if global_step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

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
                        only_mlp=only_mlp,
                    )
                except Exception as e:
                    log(
                        f"CHECKPOINT FAILED at step {global_step}: {type(e).__name__}: {e}",
                        c_train["log_file"],
                    )

        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        log(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}", c_train["log_file"])
        append_loss_row(loss_csv, {
            "epoch": epoch, "global_step": global_step, "split": "train_epoch_avg",
            "answer_loss": "", "cka_loss": "", "cka_value": "", "total_loss": avg_train_loss,
        })

        # End-of-epoch checkpoint. epoch=epoch+1: this epoch is fully done,
        # so resuming should move on to the next one.
        try:
            save_checkpoint(
                vlm=vlm,
                only_mlp=only_mlp,
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
        val_loss = run_validation(
            vlm=vlm, val_loader=val_loader, device=device, log_file=c_train["log_file"],
            intermediate_outputs=intermediate_outputs, cka_loss_fn=cka_loss_fn,
            is_CKA=is_CKA, lambda_CKA=lambda_CKA,
        )
        append_loss_row(loss_csv, {
            "epoch": c_train["epochs"], "global_step": global_step, "split": "val",
            "answer_loss": val_loss, "cka_loss": "", "cka_value": "", "total_loss": "",
        })
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
            ref_gsd_m=ref_gsd_m,
        )
    except Exception as e:
        log(f"FINAL SAMPLING FAILED: {type(e).__name__}: {e}", c_train["log_file"])

    if is_CKA:
        encoder_handle.remove()
        penultimate_handle.remove()
        final_handle.remove()
        log("Removed CKA hooks", c_train["log_file"])


if __name__ == "__main__":
    main()
