
"""
train.py
--------
Main training loop for SAR-VLM.

Checkpointing:
    - Save every 20 training batches.
    - Every 1000 global training batches:
        1. Save checkpoint first.
        2. Run validation.
        3. Generate 2 validation samples.
        4. Save validation results to the log.

This ensures that a validation/sampling failure never causes
the latest trained model weights to be lost.
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import time

from model.sar_vlm import SARVLM, build_sar_encoder
from dataset import SARVLMDataset, collate_fn


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


@torch.no_grad()
def run_validation(
    vlm,
    val_loader,
    device,
    log_file=None
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

        except Exception as e:

            # IMPORTANT:
            # Do not allow one bad sample to kill training.
            log(
                f"Sampling failed for sample {i + 1}: "
                f"{type(e).__name__}: {e}",
                log_file
            )


def main():

    config = load_config("train_config.yaml")

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
    # Dataset
    # ---------------------------------------------------------

    log(
        "Loading datasets...",
        c_train["log_file"]
    )

    train_dataset = SARVLMDataset(
        c_data["train_jsonl"],
        c_data["data_root"],
        tokenizer,
        max_length=c_train["max_length"]
    )

    val_dataset = SARVLMDataset(
        c_data["val_jsonl"],
        c_data["data_root"],
        tokenizer,
        max_length=c_train["max_length"]
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

    CHECKPOINT_EVERY = 20
    EVAL_EVERY = 1000
    NUM_SAMPLES = 2

    # Global batch counter.
    global_step = 0

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

            # =================================================
            # VALIDATE + SAMPLE EVERY 1000 BATCHES
            # =================================================

            if global_step % EVAL_EVERY == 0:

                log(
                    f"========== EVALUATION AT "
                    f"STEP {global_step} ==========",
                    c_train["log_file"]
                )

                # ---------------------------------------------
                # 1. Save AGAIN before evaluation
                # ---------------------------------------------

                # This is redundant with the every-20 save,
                # but intentionally done here for safety.
                try:

                    save_checkpoint(
                        vlm=vlm,
                        save_dir=c_train["save_dir"],
                        global_step=global_step,
                        log_file=c_train["log_file"]
                    )

                except Exception as e:

                    log(
                        f"Pre-evaluation checkpoint failed: "
                        f"{type(e).__name__}: {e}",
                        c_train["log_file"]
                    )

                # ---------------------------------------------
                # 2. Validation
                # ---------------------------------------------

                try:

                    run_validation(
                        vlm=vlm,
                        val_loader=val_loader,
                        device=device,
                        log_file=c_train["log_file"]
                    )

                except Exception as e:

                    log(
                        f"VALIDATION FAILED at step "
                        f"{global_step}: "
                        f"{type(e).__name__}: {e}",
                        c_train["log_file"]
                    )

                # ---------------------------------------------
                # 3. Sampling
                # ---------------------------------------------

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
                        f"SAMPLING FAILED at step "
                        f"{global_step}: "
                        f"{type(e).__name__}: {e}",
                        c_train["log_file"]
                    )

                # Return to training mode.
                vlm.train()

                log(
                    f"========== RESUMING TRAINING "
                    f"AT STEP {global_step} ==========",
                    c_train["log_file"]
                )

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


if __name__ == "__main__":
    main()

