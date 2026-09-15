#!/usr/bin/env python3
"""
run_benchmark.py
================
Benchmarks SAR-VLM on the generated test.jsonl / val.jsonl datasets.

Metrics
-------
  - Exact Match (EM)   : case-insensitive, for single-word ground-truth answers
  - BLEU-1/2/3/4       : sentence BLEU via NLTK
  - ROUGE-1/2/L        : via rouge-score

Usage
-----
  python Benchmarking/run_benchmark.py --subset 10   # smoke-test
  python Benchmarking/run_benchmark.py --category "presence"
  python Benchmarking/run_benchmark.py --checkpoint checkpoints/step_1000

Outputs (all saved to Benchmarking/)
  results_val_<ts>.csv   – per-sample rows with all metric scores
  results_val_<ts>.json  – same, JSON format
  summary_val_<ts>.txt   – human-readable dataset-wide + per-category report
"""

# ===========================================================================
# Imports
# ===========================================================================
import argparse
import csv
import json
import os
import re
import sys
import time
import yaml
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from model.sar_vlm import SARVLM, build_sar_encoder
from dataset import SARVLMDataset, collate_fn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/saishruti/Research1/Shreyank_20_credit/DataGen")

# ===========================================================================
# Helpers
# ===========================================================================
def log(msg: str, log_file: str = None) -> None:
    out = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(out, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(out + "\n")


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_question_and_gt(record: dict):
    """Return (human_question, gt_answer) from a JSONL record."""
    convs = record.get("conversations", [])
    question, gt = "", ""
    for turn in convs:
        if turn["from"] == "human":
            question = turn["value"].replace("<image>\n", "").replace("<image>", "").strip()
        elif turn["from"] == "gpt":
            gt = turn["value"].strip()
    return question, gt


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def is_single_word_gt(gt: str) -> bool:
    """True if the ground truth normalises to a single token."""
    return len(normalise(gt).split()) == 1


def get_latest_checkpoint(checkpoints_dir: str):
    """Find the latest checkpoint directory based on step number."""
    checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, "step_*"))
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")
    
    step_dirs = []
    for dir_path in checkpoint_dirs:
        dir_name = os.path.basename(dir_path)
        step_num = int(dir_name.replace("step_", ""))
        step_dirs.append((step_num, dir_path))
    
    step_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_path = step_dirs[0]
    return latest_path, latest_step


# ===========================================================================
# Metric functions
# ===========================================================================
def exact_match(pred: str, gt: str) -> float:
    """Case-insensitive exact match after normalisation."""
    return 1.0 if normalise(pred) == normalise(gt) else 0.0


def compute_bleu(pred: str, gt: str) -> dict:
    """Sentence BLEU 1-4 using NLTK (smoothing method 1)."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoother = SmoothingFunction().method1
    ref   = [normalise(gt).split()]
    hyp   = normalise(pred).split()
    if not hyp:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    return {
        "bleu1": sentence_bleu(ref, hyp, weights=(1,0,0,0), smoothing_function=smoother),
        "bleu2": sentence_bleu(ref, hyp, weights=(0.5,0.5,0,0), smoothing_function=smoother),
        "bleu3": sentence_bleu(ref, hyp, weights=(1/3,1/3,1/3,0), smoothing_function=smoother),
        "bleu4": sentence_bleu(ref, hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother),
    }


def compute_rouge(pred: str, gt: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(normalise(gt), normalise(pred))
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def score_all(pred: str, gt: str) -> dict:
    """Compute all metrics for one (pred, gt) pair."""
    is_sw = is_single_word_gt(gt)
    em    = exact_match(pred, gt) if is_sw else None

    bleu  = compute_bleu(pred, gt)
    rouge = compute_rouge(pred, gt)

    return {
        "is_single_word": is_sw,
        "exact_match":    em,
        **bleu,
        **rouge,
    }


# ===========================================================================
# Model loading
# ===========================================================================
def load_model(config_path: str, checkpoint_path: str = None, use_lora: bool = False, projector_path: str = None):
    """Load SAR-VLM model with checkpoint.
    
    Args:
        config_path: Path to training config file
        checkpoint_path: Path to checkpoint directory (if use_lora=True)
        use_lora: Whether to load LoRA adapters (default: False)
        projector_path: Path to projector weights file (required if use_lora=False)
    """
    config = load_config(config_path)
    c_data = config["data"]
    c_model = config["model"]
    c_lora = config["lora"]
    c_train = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Running on device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(c_model["vicuna_path"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # Load SAR encoder
    log("Loading SAR Encoder...")
    sar_encoder = build_sar_encoder(
        checkpoint_path=c_model["encoder_checkpoint"],
        freeze=True,
        d_sar=c_model["d_sar"]
    )

    # Load model
    log(f"Loading SAR-VLM model (LoRA: {use_lora})...")
    vlm = SARVLM.from_vicuna(
        vicuna_path=c_model["vicuna_path"],
        sar_encoder=sar_encoder,
        d_sar=c_model["d_sar"],
        n_visual=c_model["n_visual"],
        lora_r=c_lora["r"],
        lora_alpha=c_lora["alpha"],
        lora_dropout=c_lora["dropout"],
        lora_target_modules=c_lora["target_modules"],
        apply_lora=use_lora,
        torch_dtype=torch.float16
    )

    vlm = vlm.to(device)
    # vlm.hybrid_vicuna.gradient_checkpointing_enable()  # Removed for inference - training only
    
    # Set model to eval mode for inference
    vlm.eval()

    # Load checkpoint weights
    if use_lora:
        if checkpoint_path is None:
            checkpoint_path, global_step = get_latest_checkpoint(c_train["save_dir"])
            log(f"Using latest checkpoint: {checkpoint_path} (step {global_step})")
        else:
            global_step = int(checkpoint_path.split("_")[-1])
            log(f"Using specified checkpoint: {checkpoint_path}")
        
        log(f"Loading LoRA adapter from {checkpoint_path}...")
        vlm.hybrid_vicuna.load_adapter(checkpoint_path, adapter_name="default")
        
        projector_ckpt_path = os.path.join(checkpoint_path, "projector.pth")
        log(f"Loading projector weights from {projector_ckpt_path}...")
        vlm.projector.load_state_dict(torch.load(projector_ckpt_path, map_location=device))
    else:
        # No LoRA - load only projector weights
        if projector_path is None:
            raise ValueError("projector_path must be specified when use_lora=False")
        
        # Extract step number from projector path filename
        try:
            global_step = int(projector_path.split("step")[-1].replace(".pth", ""))
        except (ValueError, IndexError):
            global_step = 0  # Fallback if step number can't be extracted
            log(f"Warning: Could not extract step number from path, using 0")
        
        log(f"Loading projector weights from {projector_path} (step {global_step})...")
        proj_state = torch.load(projector_path, map_location=device)
        
        # Debug: Check what we loaded
        log(f"Loaded checkpoint type: {type(proj_state)}")
        if isinstance(proj_state, dict):
            log(f"Loaded checkpoint keys: {list(proj_state.keys())}")
        
        # Unwrap projector weights from checkpoint dictionary
        for unwrap_key in ("model", "state_dict", "projector", "projector_state_dict"):
            if isinstance(proj_state, dict) and unwrap_key in proj_state:
                proj_state = proj_state[unwrap_key]
                log(f"Unwrapped using key: {unwrap_key}")
                break
        
        if isinstance(proj_state, dict):
            log(f"Final projector state keys: {list(proj_state.keys())}")
        log(f"Current projector architecture: {vlm.projector}")
        
        vlm.projector.load_state_dict(proj_state)
        log("Projector weights loaded successfully")

    log("Model loaded successfully")
    return vlm, tokenizer, device, global_step


# ===========================================================================
# Inference for one record (using SARVLMDataset pipeline)
# ===========================================================================
@torch.no_grad()
def run_one(batch: dict, vlm, tokenizer, device) -> list:
    """Generate predictions for a batch using the same pipeline as val.py."""
    # Move batch to device
    sar_input = batch["sar_input"].to(device, dtype=torch.float32)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    batch_size = input_ids.shape[0]
    predictions = []
    
    for i in range(batch_size):
        # Find where the answer starts (where labels are not -100)
        # The dataset masks the prompt with -100, so the first non -100 is the answer start
        labels = batch["labels"][i]
        first_answer_position = 0
        for j, label in enumerate(labels):
            if label != -100:
                first_answer_position = j
                break
        
        # Use the prompt up to the answer position for this sample
        prompt_ids = input_ids[i:i+1, :first_answer_position]
        prompt_attention_mask = attention_mask[i:i+1, :first_answer_position]
        sar_input_single = sar_input[i:i+1]
        
        # Generate (model already in eval mode)
        try:
            output_ids = vlm.generate(
                sar_input=sar_input_single,
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=64,
                do_sample=False
            )
            # vlm.generate() returns ONLY the generated tokens (not the full sequence)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predictions.append(generated_text.strip())
        except Exception as e:
            log(f"Generation error for sample {i}: {e}")
            predictions.append("")
    
    return predictions


# ===========================================================================
# Main benchmark loop (using SARVLMDataset)
# ===========================================================================
def run_benchmark(dataset, vlm, tokenizer, device, split_name: str, results_prefix: str = "Ablation_3") -> list:
    """Run benchmark with memory management and checkpointing."""
    results = []
    
    # Memory management constants
    CHECKPOINT_EVERY = 100  # Save intermediate results every 100 samples
    CLEANUP_EVERY = 50      # Clean GPU memory every 50 samples
    BATCH_SIZE = 1         # Batch size for processing (increase carefully to avoid OOM)
    
    # Create dataloader with configurable batch size
    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=4,
        pin_memory=True
    )
    
    n = len(dataset)
    total_processed = 0  # Track total samples processed
    
    for idx, batch in enumerate(val_loader):
        batch_size = batch["input_ids"].shape[0]
        batch_start_idx = idx * BATCH_SIZE
        
        # Process the entire batch
        t0 = time.time()
        try:
            predictions = run_one(batch, vlm, tokenizer, device)
            elapsed = time.time() - t0
        except torch.cuda.OutOfMemoryError as e:
            log(f"  CUDA OOM on batch {idx}: clearing cache and continuing...")
            torch.cuda.empty_cache()
            predictions = [""] * batch_size
            elapsed = time.time() - t0
        except Exception as e:
            log(f"  ERROR on batch {idx}: {e}")
            predictions = [""] * batch_size
            elapsed = time.time() - t0
        
        # Process each sample in the batch
        for i in range(batch_size):
            sample_idx = batch_start_idx + i
            if sample_idx >= len(dataset.records):
                break
                
            # Get ground truth from labels (convert -100 to actual token IDs)
            labels = batch["labels"][i].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            gt_text = tokenizer.decode(labels, skip_special_tokens=True)
            
            # Get question from the dataset record
            record = dataset.records[sample_idx]
            category = record.get("category", "unknown")
            question, _ = extract_question_and_gt(record)
            rec_id = record.get("id", f"idx_{sample_idx}")
            
            pred = predictions[i]
            metrics = score_all(pred, gt_text)
            error = None if pred else "Empty prediction"

            row = {
                "id":         rec_id,
                "split":      split_name,
                "category":   category,
                "image":      record.get("image", ""),
                "question":   question,
                "ground_truth": gt_text,
                "prediction": pred,
                "elapsed_s":  round(elapsed / batch_size, 2),  # Average time per sample
                "error":      error,
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
            }
            results.append(row)
            total_processed += 1

        # Periodic GPU memory cleanup
        if total_processed % CLEANUP_EVERY == 0:
            torch.cuda.empty_cache()
            gpu_alloc = torch.cuda.memory_allocated() / 1024**3
            gpu_res = torch.cuda.memory_reserved() / 1024**3
            log(f"  Memory cleanup at sample {total_processed}: {gpu_alloc:.2f}GB allocated / {gpu_res:.2f}GB reserved")

        # Periodic checkpointing
        if total_processed % CHECKPOINT_EVERY == 0:
            ts_partial = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = SCRIPT_DIR / f"{results_prefix}_checkpoint_{split_name}_{ts_partial}.json"
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)
            log(f"  Checkpoint saved: {checkpoint_path}")

        # Live progress line (show batch-level progress)
        if results:  # Only show progress if we have results
            latest_row = results[-1]
            em_str = f" EM={latest_row['exact_match']:.0f}" if latest_row.get('exact_match') is not None else ""
            print(
                f"\r  {total_processed}/{n} ({total_processed/n*100:.0f}%)"
                f"  bleu1={latest_row['bleu1']:.3f}"
                f"  rouge1={latest_row['rouge1']:.3f}"
                f"{em_str}"
                f"  [{latest_row.get('category', 'unknown')}]          ",
                end="", flush=True,
            )

    print()
    return results


# ===========================================================================
# Statistics helpers
# ===========================================================================
def _safe_mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _stats_block(rows, prefix="  ") -> list:
    """Return human-readable stat lines for a list of result rows."""
    valid = [r for r in rows if not r["error"]]
    if not valid:
        return [f"{prefix}No valid samples."]

    n = len(valid)
    sw  = [r for r in valid if r["is_single_word"]]
    mw  = [r for r in valid if not r["is_single_word"]]

    lines = []
    lines.append(f"{prefix}Samples          : {n}  (single-word GT: {len(sw)}, multi-word: {len(mw)})")

    # BLEU
    for k in ["bleu1", "bleu2", "bleu3", "bleu4"]:
        v = _safe_mean([r[k] for r in valid])
        lines.append(f"{prefix}{k.upper():<10}: {v:.4f}")

    # ROUGE
    for k in ["rouge1", "rouge2", "rougeL"]:
        v = _safe_mean([r[k] for r in valid])
        lines.append(f"{prefix}{k.upper():<10}: {v:.4f}")

    # Exact match (single-word only)
    if sw:
        em = _safe_mean([r["exact_match"] for r in sw])
        lines.append(f"{prefix}EXACT MATCH (single-word GT only): {em:.4f}  ({int(em*len(sw))}/{len(sw)} correct)")

    return lines


# ===========================================================================
# Save + Report
# ===========================================================================
def save_results(results: list, split_name: str, prefix: str = "Ablation_3") -> None:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = SCRIPT_DIR / f"{prefix}_results_{split_name}_{ts}"

    # CSV
    csv_path = base.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log(f"CSV  -> {csv_path}")

    # JSON
    json_path = base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"JSON -> {json_path}")

    # ── Summary report ────────────────────────────────────────────────────────
    summary_lines = []
    W = 64

    summary_lines += [
        "=" * W,
        f"  SAR-VLM Benchmark  --  {split_name.upper()} split",
        f"  Timestamp : {ts}",
        f"  Total samples : {len(results)}   Errors: {sum(1 for r in results if r['error'])}",
        "=" * W,
        "",
        "DATASET-WIDE STATISTICS",
        "-" * W,
    ]
    summary_lines += _stats_block(results)

    # Per-category
    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)

    summary_lines += ["", "PER-CATEGORY STATISTICS", "=" * W]
    for cat in sorted(cats):
        summary_lines += [
            "",
            f"  [{cat}]  (n={len(cats[cat])})",
            "-" * (W // 2),
        ]
        summary_lines += _stats_block(cats[cat], prefix="    ")

    summary_lines += ["", "=" * W]

    summary_text = "\n".join(summary_lines)
    summary_path = SCRIPT_DIR / f"{prefix}_summary_{split_name}_{ts}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    log(f"Summary -> {summary_path}")


# ===========================================================================
# CLI + Entry point
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark SAR-VLM on validation dataset")
    p.add_argument("--subset",   type=int, default=None,
                   help="Evaluate only the first N samples (smoke-test)")
    p.add_argument("--shuffle",  action="store_true",
                   help="Shuffle before taking --subset")
    p.add_argument("--category", type=str, default=None,
                   help="Filter to one question category")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to specific checkpoint (e.g., checkpoints/step_1000)")
    p.add_argument("--config", type=str, default="train_config.yaml",
                   help="Path to training config file")
    p.add_argument("--use_lora", action="store_true", default=False,
                   help="Load LoRA adapters from checkpoint (default: False)")
    p.add_argument("--projector_path", type=str, 
                   default="/home/saishruti/Research1/Shreyank_20_credit/MyModels/SarEncPlusVicuna/Benchmarking/projector_step70000.pth",
                   help="Path to projector weights file (required if use_lora=False)")
    p.add_argument("--results_prefix", type=str, default="Ablation_3",
                   help="Prefix for result files (default: Ablation_3)")
    
    return p.parse_args()


def main():
    args = parse_args()

    # Download NLTK punkt if needed
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    log("=" * 60)
    log("SAR-VLM Benchmark Runner")
    log("=" * 60)

    # Change to repo directory for imports
    os.chdir(REPO_DIR)

    # Load config
    config = load_config(args.config)
    c_data = config["data"]
    c_train = config["training"]

    vlm, tokenizer, device, global_step = load_model(
        args.config, 
        args.checkpoint, 
        use_lora=args.use_lora,
        projector_path=args.projector_path
    )

    gpu_alloc = torch.cuda.memory_allocated() / 1024**3
    gpu_res   = torch.cuda.memory_reserved()  / 1024**3
    log(f"GPU: {gpu_alloc:.2f} GB allocated / {gpu_res:.2f} GB reserved")

    # Load validation dataset using SARVLMDataset (same as val.py)
    log(f"\nLoading validation dataset from {c_data['val_jsonl']} ...")
    dataset = SARVLMDataset(
        c_data["val_jsonl"],
        c_data["data_root"],
        tokenizer,
        max_length=c_train["max_length"]
    )
    log(f"  Total samples: {len(dataset)}")

    if args.category:
        # Filter dataset by category
        filtered_records = [r for r in dataset.records if r.get("category") == args.category]
        dataset.records = filtered_records
        log(f"  After category filter '{args.category}': {len(dataset.records)}")

    if args.shuffle:
        import random
        random.shuffle(dataset.records)

    if args.subset is not None:
        dataset.records = dataset.records[: args.subset]
        log(f"  Subset: {len(dataset.records)}")

    log(f"\nStarting evaluation on {len(dataset.records)} samples ...\n")
    t0 = time.time()
    results = run_benchmark(dataset, vlm, tokenizer, device, "val", args.results_prefix)
    elapsed = time.time() - t0
    log(f"Done in {elapsed:.1f}s  ({elapsed/len(results):.2f}s/sample)")

    save_results(results, "val", prefix=args.results_prefix)

    log("\nAll done!")


if __name__ == "__main__":
    main()
