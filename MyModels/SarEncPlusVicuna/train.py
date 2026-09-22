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
    - Set training.checkpoint_path in the config to the checkpoint
      directory you want to resume from. Then pass --load-checkpoint
      on the command line to actually use it; without that flag, the
      path in the config is ignored and training starts from scratch.
"""

import argparse
import functools
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import time

from model.sar_vlm import SARVLM, build_sar_encoder
from dataset import SARVLMDataset, collate_fn, count_jsonl_records


"""Helper functions"""
def load_config(config_path: str):
    with open(config_path, "r") as f:
        """1. Opening a config file and returning the same. File input"""
        return yaml.safe_load(f)


def log(msg: str, log_file: str = None):
    """2. Prints a message on the terminal or writes it to a log file. To ensure the multiple intermediate training 
    steps are going on. File output """
    out = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(out)

    if log_file:
        with open(log_file, "a") as f:
            f.write(out + "\n")


def save_checkpoint(vlm, save_dir, global_step, log_file=None):
    """
    3. Save LoRA adapters (of hybrid vicuna - save_pretrained method) and projector (separately saved).
    Saves global step as well so that later on it can be found to continue from the same step

    This function is intentionally independent of validation/sampling,
    so a failure during validation cannot prevent checkpoint creation.
    """
    #Making directory for checkpoint
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
    4. (a) Load a checkpoint saved by save_checkpoint() into an already-built
    SARVLM: the LoRA adapter weights and the (b) SAR -> Vicuna projector.

    vlm.hybrid_vicuna is a PEFT-wrapped model using the
    default adapter name "default" (PEFT's default when none is given,
    which is what save_checkpoint's `vlm.hybrid_vicuna.save_pretrained(...)`
    saves under).

    Returns the (c) global_step recorded in the checkpoint's
    training_state.pth (0 if that file isn't present), so the caller can
    resume the step counter for checkpoint-naming continuity.
    """

    log(f"Loading checkpoint from: {checkpoint_path}", log_file)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint_path does not exist or is not a directory: "
            f"{checkpoint_path}"
        )

    # LoRA adapter weight - setting adapter is needed incase we are loading multiple adapters with different names.
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
    
"""Validation functions"""
@torch.no_grad()
def run_validation(
    vlm,
    val_loader,
    device,
    log_file=None
):
    """
    1. Run validation over the complete validation loader. Returns only avg loss.
    """

    vlm.eval()

    total_val_loss = 0.0
    num_batches = 0

    val_pbar = tqdm(
        val_loader,
        desc="Validation"
    )

    for batch in val_pbar:

        """It keeps the sar input in float32 but amp might act on top of it 
        Might only be a fix at the input level. torch.amp will operate in mixed precision of fp16 and fp32 
        So the intermediate outputs will adapt to this mixed precision irrespectibe of thiss dtype being 
        set."""
        sar_input = batch["sar_input"].to(
            device,
            dtype=torch.float32
        )

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
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

    if num_batches == 0:
        return 0.0

    avg_val_loss = total_val_loss / num_batches

    log(
        f"Validation Loss: {avg_val_loss:.4f}",
        log_file
    )

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
    2. Generate a small number of outputs from the validation set.

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

            input_ids = sample["input_ids"]

            labels = sample["labels"]

            # Keep only the prompt tokens.
            prompt_mask = labels == -100

            prompt_ids = input_ids[
                prompt_mask
            ].unsqueeze(0).to(device)
            
            #attend to everythign in the sample - used sometimes non trivially as well when prompts of different 
            #lengths are passed in the same batch so through padding so that the padded parts are ignored.
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

        except Exception as e:

            # IMPORTANT:
            # Do not allow one bad sample to kill training.
            log(
                f"Sampling failed for sample {i + 1}: "
                f"{type(e).__name__}: {e}",
                log_file
            )


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
    # Model
    # ---------------------------------------------------------

    log(
        "Loading SAR Encoder and Hybrid Vicuna Model...",
        c_train["log_file"]
    )

    sar_encoder = build_sar_encoder(
        checkpoint_path=c_model["encoder_checkpoint"],
        freeze=True,
        d_sar=c_model["d_sar"]
    )

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

    # Gradient checkpointing
    vlm.hybrid_vicuna.gradient_checkpointing_enable()

    vlm = vlm.to(device)

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

        global_step = load_checkpoint_into_vlm(
            vlm=vlm,
            checkpoint_path=checkpoint_path,
            device=device,
            log_file=c_train["log_file"]
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    log(
        "Loading datasets...",
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
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=c_train["micro_batch_size"],
        shuffle=False,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
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

    CHECKPOINT_EVERY = 20
    NUM_SAMPLES = 2

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

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
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
            # Backward
            # -------------------------------------------------

            scaler.scale(loss).backward()

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

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            loss_val = (
                loss.item() *
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
            log_file=c_train["log_file"]
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

    # No extra checkpoint save here -- the end-of-epoch save above for
    # the final epoch already captured the model at this global_step.


if __name__ == "__main__":
    main()
