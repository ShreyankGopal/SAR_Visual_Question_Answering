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


def discover_val_datasets(c_data: dict) -> list:
    """
    Collect every val dataset defined in the config's `data` block.

    Looks for val_jsonl / data_root (primary), then val_jsonl_2 / data_root_2,
    val_jsonl_3 / data_root_3, etc. (same numbering scheme already used for
    train_jsonl_N / data_root_N). An optional dataset_name_N key gives it a
    label; otherwise the label is inferred from the jsonl path.
    """
    datasets = []

    def _name_for(key_suffix: str, jsonl_path: str) -> str:
        name_key = f"dataset_name{key_suffix}"
        if c_data.get(name_key):
            return c_data[name_key]
        return Path(jsonl_path).parent.parent.name or Path(jsonl_path).stem

    if c_data.get("val_jsonl"):
        datasets.append({
            "name": _name_for("", c_data["val_jsonl"]),
            "val_jsonl": c_data["val_jsonl"],
            "data_root": c_data["data_root"],
        })

    i = 2
    while c_data.get(f"val_jsonl_{i}"):
        datasets.append({
            "name": _name_for(f"_{i}", c_data[f"val_jsonl_{i}"]),
            "val_jsonl": c_data[f"val_jsonl_{i}"],
            "data_root": c_data[f"data_root_{i}"],
        })
        i += 1

    return datasets


def list_checkpoints_desc(checkpoints_dir: str) -> list:
    """All step_* checkpoint dirs under checkpoints_dir, sorted newest-first."""
    checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, "step_*"))
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")

    step_dirs = []
    for dir_path in checkpoint_dirs:
        dir_name = os.path.basename(dir_path)
        try:
            step_num = int(dir_name.replace("step_", ""))
        except ValueError:
            continue
        step_dirs.append((step_num, dir_path))

    step_dirs.sort(key=lambda x: x[0], reverse=True)
    return step_dirs


def get_latest_checkpoint(checkpoints_dir: str):
    """Find the latest checkpoint directory based on step number."""
    latest_step, latest_path = list_checkpoints_desc(checkpoints_dir)[0]
    return latest_path, latest_step


def checkpoint_is_healthy(vlm, checkpoint_path: str, health_batch: dict, device) -> bool:
    """
    Load adapter + projector weights from checkpoint_path into vlm, then run
    one real forward pass on health_batch and check the loss/logits are
    finite. Returns False (without raising) on any load or NaN/Inf failure,
    so the caller can fall back to the next-older checkpoint.
    """
    try:
        vlm.hybrid_vicuna.load_adapter(checkpoint_path, adapter_name="default")
        projector_path = os.path.join(checkpoint_path, "projector.pth")
        vlm.projector.load_state_dict(torch.load(projector_path, map_location=device))
    except Exception as e:
        log(f"    Failed to load weights from {checkpoint_path}: {e}")
        return False

    vlm.eval()
    try:
        with torch.no_grad():
            output = vlm(
                sar_input=health_batch["sar_input"].to(device, dtype=torch.float32),
                input_ids=health_batch["input_ids"].to(device),
                attention_mask=health_batch["attention_mask"].to(device),
                labels=health_batch["labels"].to(device),
            )
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            log(f"    Non-finite loss ({loss}) from {checkpoint_path}")
            return False
        if output.logits is not None and not torch.isfinite(output.logits).all():
            log(f"    Non-finite logits from {checkpoint_path}")
            return False
        return True
    except Exception as e:
        log(f"    Forward pass failed for {checkpoint_path}: {e}")
        return False


def build_health_check_batch(c_data: dict, c_train: dict, tokenizer) -> dict:
    """One-sample batch (from the primary val set) used to sanity-check a checkpoint."""
    health_dataset = SARVLMDataset(
        c_data["val_jsonl"], c_data["data_root"], tokenizer, max_length=c_train["max_length"]
    )
    health_dataset.records = health_dataset.records[:1]
    return collate_fn([health_dataset[0]], tokenizer)


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
def load_model(config_path: str, checkpoint_path: str = None):
    """Load SAR-VLM model with checkpoint."""
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
    log("Loading SAR-VLM model...")
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
        torch_dtype=torch.float16
    )

    vlm = vlm.to(device)
    vlm.hybrid_vicuna.gradient_checkpointing_enable()

    if checkpoint_path is not None:
        # User explicitly chose this checkpoint -- load it as-is, no health check.
        global_step = int(checkpoint_path.split("_")[-1])
        log(f"Loading specified checkpoint: {checkpoint_path}")
        vlm.hybrid_vicuna.load_adapter(checkpoint_path, adapter_name="default")
        projector_path = os.path.join(checkpoint_path, "projector.pth")
        vlm.projector.load_state_dict(torch.load(projector_path, map_location=device))
    else:
        # Walk step_* checkpoints newest-first; skip any that fail to load or
        # produce a NaN/Inf loss on a real sample, until a healthy one is found.
        candidates = list_checkpoints_desc(c_train["save_dir"])
        log(f"Found {len(candidates)} checkpoint(s) in {c_train['save_dir']}")
        log("Building 1-sample health-check batch from the primary val set...")
        health_batch = build_health_check_batch(c_data, c_train, tokenizer)

        chosen = None
        for step_num, cand_path in candidates:
            log(f"  Checking step_{step_num} ({cand_path}) ...")
            if checkpoint_is_healthy(vlm, cand_path, health_batch, device):
                log(f"  -> step_{step_num} is healthy (finite loss). Using it.")
                chosen = (cand_path, step_num)
                break
            log(f"  -> step_{step_num} failed health check, trying next-older checkpoint.")

        if chosen is None:
            raise RuntimeError(
                f"No healthy checkpoint found in {c_train['save_dir']} "
                f"(all {len(candidates)} candidate(s) failed to load or produced NaN/Inf)"
            )
        checkpoint_path, global_step = chosen

    log("Model loaded successfully")
    return vlm, tokenizer, device, global_step


# ===========================================================================
# Inference for one record (using SARVLMDataset pipeline)
# ===========================================================================
@torch.no_grad()
def run_one(batch: dict, vlm, tokenizer, device) -> str:
    """Generate prediction for a single batch using the same pipeline as val.py."""
    # Move batch to device
    sar_input = batch["sar_input"].to(device, dtype=torch.float32)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Find where the answer starts (where labels are not -100)
    # The dataset masks the prompt with -100, so the first non -100 is the answer start
    labels = batch["labels"][0]
    first_answer_position = 0
    for i, label in enumerate(labels):
        if label != -100:
            first_answer_position = i
            break
    
    # Use the prompt up to the answer position
    prompt_ids = input_ids[:, :first_answer_position]
    prompt_attention_mask = attention_mask[:, :first_answer_position]
    
    # Generate
    vlm.eval()
    try:
        output_ids = vlm.generate(
            sar_input=sar_input,
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=64,
            do_sample=False
        )
        # vlm.generate() returns ONLY the generated tokens (not the full sequence)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        log(f"Generation error: {e}")
        return ""


# ===========================================================================
# Main benchmark loop (using SARVLMDataset)
# ===========================================================================
def run_benchmark(dataset, vlm, tokenizer, device, split_name: str, dataset_name: str = "val") -> list:
    results = []
    
    # Create dataloader with batch_size=1 for individual processing
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=1,
        pin_memory=False
    )
    
    n = len(dataset)
    
    for idx, batch in enumerate(val_loader):
        # Get ground truth from labels (convert -100 to actual token IDs)
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        gt_text = tokenizer.decode(labels[0], skip_special_tokens=True)
        
        # Get question from the dataset record
        record = dataset.records[idx]
        category = record.get("category", "unknown")
        question, _ = extract_question_and_gt(record)
        rec_id = record.get("id", f"idx_{idx}")
        
        t0 = time.time()
        try:
            pred    = run_one(batch, vlm, tokenizer, device)
            elapsed = time.time() - t0
            metrics = score_all(pred, gt_text)
            error   = None
        except Exception as e:
            pred    = ""
            elapsed = time.time() - t0
            metrics = {"is_single_word": is_single_word_gt(gt_text),
                       "exact_match": None,
                       "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
                       "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            error   = str(e)
            log(f"  ERROR on {rec_id}: {e}")

        row = {
            "id":         rec_id,
            "dataset":    dataset_name,
            "split":      split_name,
            "category":   category,
            "image":      record.get("image", ""),
            "question":   question,
            "ground_truth": gt_text,
            "prediction": pred,
            "elapsed_s":  round(elapsed, 2),
            "error":      error,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        }
        results.append(row)

        # Live progress line
        em_str = f" EM={metrics['exact_match']:.0f}" if metrics["exact_match"] is not None else ""
        print(
            f"\r  [{dataset_name}] {idx+1}/{n} ({(idx+1)/n*100:.0f}%)"
            f"  bleu1={metrics['bleu1']:.3f}"
            f"  rouge1={metrics['rouge1']:.3f}"
            f"{em_str}"
            f"  [{category}]          ",
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
def save_results(results: list, split_name: str) -> None:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = SCRIPT_DIR / f"results_{split_name}_{ts}"

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
    W = 64
    summary_lines = [
        "=" * W,
        f"  SAR-VLM Benchmark  --  {split_name.upper()} split",
        f"  Timestamp : {ts}",
        f"  Total samples : {len(results)}   Errors: {sum(1 for r in results if r['error'])}",
        "=" * W,
    ]

    datasets = {}
    for r in results:
        datasets.setdefault(r["dataset"], []).append(r)

    # Per-dataset, per-category breakdown
    for ds_name in sorted(datasets):
        ds_rows = datasets[ds_name]
        summary_lines += [
            "",
            "#" * W,
            f"  DATASET: {ds_name}  (n={len(ds_rows)})",
            "#" * W,
            "",
            "  DATASET-WIDE STATISTICS",
            "  " + "-" * (W - 2),
        ]
        summary_lines += _stats_block(ds_rows)

        cats = {}
        for r in ds_rows:
            cats.setdefault(r["category"], []).append(r)

        summary_lines += ["", "  PER-CATEGORY STATISTICS", "  " + "-" * (W - 2)]
        for cat in sorted(cats):
            summary_lines += [
                "",
                f"    [{cat}]  (n={len(cats[cat])})",
                "    " + "-" * (W // 2),
            ]
            summary_lines += _stats_block(cats[cat], prefix="      ")

    # Overall, across all datasets combined
    summary_lines += [
        "",
        "=" * W,
        "  OVERALL  (all datasets combined)",
        "=" * W,
        "",
        "OVERALL DATASET-WIDE STATISTICS",
        "-" * W,
    ]
    summary_lines += _stats_block(results)

    overall_cats = {}
    for r in results:
        overall_cats.setdefault(r["category"], []).append(r)

    summary_lines += ["", "OVERALL PER-CATEGORY STATISTICS (pooled across datasets)", "-" * W]
    for cat in sorted(overall_cats):
        summary_lines += [
            "",
            f"  [{cat}]  (n={len(overall_cats[cat])})",
            "-" * (W // 2),
        ]
        summary_lines += _stats_block(overall_cats[cat], prefix="    ")

    summary_lines += ["", "=" * W]

    summary_text = "\n".join(summary_lines)
    summary_path = SCRIPT_DIR / f"summary_{split_name}_{ts}.txt"
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

    vlm, tokenizer, device, global_step = load_model(args.config, args.checkpoint)

    gpu_alloc = torch.cuda.memory_allocated() / 1024**3
    gpu_res   = torch.cuda.memory_reserved()  / 1024**3
    log(f"GPU: {gpu_alloc:.2f} GB allocated / {gpu_res:.2f} GB reserved")

    # Discover every val dataset defined in the config (val_jsonl, val_jsonl_2, ...)
    val_datasets = discover_val_datasets(c_data)
    if not val_datasets:
        raise ValueError("No val_jsonl found in config['data']")
    log(f"\nFound {len(val_datasets)} val dataset(s): {[d['name'] for d in val_datasets]}")

    all_results = []
    for ds in val_datasets:
        log(f"\nLoading dataset '{ds['name']}' from {ds['val_jsonl']} ...")
        dataset = SARVLMDataset(
            ds["val_jsonl"],
            ds["data_root"],
            tokenizer,
            max_length=c_train["max_length"]
        )
        log(f"  Total samples: {len(dataset)}")

        if args.category:
            filtered_records = [r for r in dataset.records if r.get("category") == args.category]
            dataset.records = filtered_records
            log(f"  After category filter '{args.category}': {len(dataset.records)}")

        if args.shuffle:
            import random
            random.shuffle(dataset.records)

        if args.subset is not None:
            dataset.records = dataset.records[: args.subset]
            log(f"  Subset: {len(dataset.records)}")

        if not dataset.records:
            log(f"  Skipping '{ds['name']}' -- no samples after filtering")
            continue

        log(f"\nStarting evaluation on '{ds['name']}' ({len(dataset.records)} samples) ...\n")
        t0 = time.time()
        results = run_benchmark(dataset, vlm, tokenizer, device, "val", dataset_name=ds["name"])
        elapsed = time.time() - t0
        log(f"Done '{ds['name']}' in {elapsed:.1f}s  ({elapsed/len(results):.2f}s/sample)")
        all_results.extend(results)

    save_results(all_results, "val")

    log("\nAll done!")


if __name__ == "__main__":
    main()
