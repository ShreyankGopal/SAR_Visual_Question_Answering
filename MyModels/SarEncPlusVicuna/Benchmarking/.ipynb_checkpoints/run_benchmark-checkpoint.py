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
  python Benchmarking/run_benchmark.py --split test
  python Benchmarking/run_benchmark.py --split val
  python Benchmarking/run_benchmark.py --split both

Outputs (all saved to Benchmarking/)
  results_<split>_<ts>.csv   – per-sample rows with all metric scores
  results_<split>_<ts>.json  – same, JSON format
  summary_<split>_<ts>.txt   – human-readable dataset-wide + per-category report
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
DATA_DIR = Path("/home/saishruti/Research1/Shreyank_20_credit/DataGen/data")
DATA_ROOT = Path("/home/saishruti/Research1/Shreyank_20_credit/DataGen")

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
LAND_COVER_CLASSES = [
    "Bareland", "Rangeland", "Developed Space", "Road",
    "Tree", "Water", "Agriculture Land", "Building",
]

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    f"The possible land-cover classes are: {', '.join(LAND_COVER_CLASSES)}."
)

ONE_WORD_SUFFIX = " Answer in one word only."

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


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def is_single_word_gt(gt: str) -> bool:
    """True if the ground truth normalises to a single token."""
    return len(normalise(gt).split()) == 1


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

    # Find latest checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_path, global_step = get_latest_checkpoint(c_train["save_dir"])
        log(f"Using latest checkpoint: {checkpoint_path} (step {global_step})")
    else:
        global_step = int(checkpoint_path.split("_")[-1])
        log(f"Using specified checkpoint: {checkpoint_path}")

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

    # Load checkpoint weights
    log(f"Loading checkpoint weights from {checkpoint_path}...")
    vlm.hybrid_vicuna.load_adapter(checkpoint_path, adapter_name="defualt")
    
    projector_path = os.path.join(checkpoint_path, "projector.pth")
    vlm.projector.load_state_dict(torch.load(projector_path, map_location=device))

    log("Model loaded successfully")
    return vlm, tokenizer, device, global_step


# ===========================================================================
# Inference for one record
# ===========================================================================
@torch.no_grad()
def run_one(record: dict, vlm, tokenizer, device) -> str:
    """Generate prediction for a single record."""
    # Load image
    img_path = DATA_ROOT / record["image"]
    try:
        import tifffile
        img_np = tifffile.imread(img_path)
        img_tensor = torch.from_numpy(img_np).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
    except Exception as e:
        log(f"Error loading image {img_path}: {e}")
        return ""

    # Get question and ground truth
    question, gt = extract_question_and_gt(record)

    # Append one-word instruction if GT is a single word
    if is_single_word_gt(gt):
        question = question + ONE_WORD_SUFFIX

    # Format prompt with system prompt
    prompt = f"{SYSTEM_PROMPT} USER: {question} ASSISTANT:"

    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Prepare SAR input
    sar_input = img_tensor.unsqueeze(0).to(device, dtype=torch.float32)

    # Generate
    vlm.eval()
    try:
        output_ids = vlm.generate(
            sar_input=sar_input,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        log(f"Generation error: {e}")
        return ""


# ===========================================================================
# Main benchmark loop
# ===========================================================================
def run_benchmark(records: list, vlm, tokenizer, device, split_name: str) -> list:
    results = []
    n = len(records)

    for idx, record in enumerate(records):
        rec_id   = record.get("id", f"idx_{idx}")
        category = record.get("category", "unknown")
        question, gt = extract_question_and_gt(record)

        t0 = time.time()
        try:
            pred    = run_one(record, vlm, tokenizer, device)
            elapsed = time.time() - t0
            metrics = score_all(pred, gt)
            error   = None
        except Exception as e:
            pred    = ""
            elapsed = time.time() - t0
            metrics = {"is_single_word": is_single_word_gt(gt),
                       "exact_match": None,
                       "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
                       "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            error   = str(e)
            log(f"  ERROR on {rec_id}: {e}")

        row = {
            "id":         rec_id,
            "split":      split_name,
            "category":   category,
            "image":      record.get("image", ""),
            "question":   question,
            "ground_truth": gt,
            "prediction": pred,
            "elapsed_s":  round(elapsed, 2),
            "error":      error,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        }
        results.append(row)

        # Live progress line
        em_str = f" EM={metrics['exact_match']:.0f}" if metrics["exact_match"] is not None else ""
        print(
            f"\r  {idx+1}/{n} ({(idx+1)/n*100:.0f}%)"
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
    summary_path = SCRIPT_DIR / f"summary_{split_name}_{ts}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    log(f"Summary -> {summary_path}")


# ===========================================================================
# CLI + Entry point
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark SAR-VLM on LULC QA datasets")
    p.add_argument("--split",    choices=["test", "val", "both"], default="test")
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
    log(f"System prompt : {SYSTEM_PROMPT[:80]}...")
    log("=" * 60)

    # Change to repo directory for imports
    os.chdir(REPO_DIR)

    vlm, tokenizer, device, global_step = load_model(args.config, args.checkpoint)

    gpu_alloc = torch.cuda.memory_allocated() / 1024**3
    gpu_res   = torch.cuda.memory_reserved()  / 1024**3
    log(f"GPU: {gpu_alloc:.2f} GB allocated / {gpu_res:.2f} GB reserved")

    splits = ["test", "val"] if args.split == "both" else [args.split]

    for split in splits:
        jsonl_path = DATA_DIR / f"{split}.jsonl"
        if not jsonl_path.exists():
            log(f"WARNING: {jsonl_path} not found -- skipping")
            continue

        log(f"\nLoading {split}.jsonl ...")
        records = load_jsonl(jsonl_path)
        log(f"  Total records: {len(records)}")

        if args.category:
            records = [r for r in records if r.get("category") == args.category]
            log(f"  After category filter '{args.category}': {len(records)}")

        if args.shuffle:
            import random; random.shuffle(records)

        if args.subset is not None:
            records = records[: args.subset]
            log(f"  Subset: {len(records)}")

        log(f"\nStarting evaluation on {len(records)} records ...\n")
        t0 = time.time()
        results = run_benchmark(records, vlm, tokenizer, device, split)
        elapsed = time.time() - t0
        log(f"Done in {elapsed:.1f}s  ({elapsed/len(results):.2f}s/sample)")

        save_results(results, split)

    log("\nAll done!")


if __name__ == "__main__":
    main()
