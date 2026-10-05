#!/usr/bin/env python3
"""
run_benchmark.py
================
Benchmarks SAR-VLM on every dataset in the config, using val + test combined
for each dataset (both splits reported together under that dataset's name).

Datasets come from config['data']: val_jsonl / data_root, val_jsonl_2 /
data_root_2, ... (same numbering as training). For each one, the sibling
split file is picked up automatically (val.jsonl <-> test.jsonl,
val_added.jsonl <-> test_added.jsonl, ...) if it exists; explicit
test_jsonl / test_jsonl_N keys are also honoured. Every file used is logged.

The model is loaded exactly as in train.py: fp32 everywhere (LLM, projector,
GeoRoPE adapter), the GeoRoPE adapter weights are restored from the
checkpoint, and gsd_ratio is passed so GCC calibration is active.

Metrics
-------
  - Exact Match (EM)   : normalised full-answer match, on every sample
  - BLEU-1/2/3/4       : sentence BLEU via NLTK
  - ROUGE-1/2/L        : via rouge-score
  - BBox IoU           : for answers that are a [x1, y1, x2, y2] box
                         (mean IoU and accuracy at IoU >= 0.5)

Reported per dataset (overall + per category), per category pooled across
datasets, and overall.

Usage
-----
  python Benchmarking/run_benchmark.py --subset 10              # smoke-test
  python Benchmarking/run_benchmark.py --batch_size 8
  python Benchmarking/run_benchmark.py --category region_grounding
  python Benchmarking/run_benchmark.py --checkpoint checkpoints/step_1000

Outputs (saved in the experiment's main directory -- the folder containing
the checkpoints folder, e.g. GeoRoPE_CKA_Data/ -- named by checkpoint step)
  predictions_step<N>_<ts>.jsonl  - one JSON object per sample (id, dataset,
                                    split, category, question, ground_truth,
                                    prediction, metrics), written as each
                                    batch finishes so a crash keeps the
                                    predictions made so far
  results_step<N>_<ts>.csv        - same rows as CSV, at the end
  summary_step<N>_<ts>.txt        - per-dataset / per-category / overall report
"""

# ===========================================================================
# Imports
# ===========================================================================
import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from model.sar_vlm import SARVLM, build_sar_encoder
from dataset import SARVLMDataset, collate_fn


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


def extract_question_and_gt(record: dict):
    """Return (human_question, gt_answer) from a JSONL record."""
    question, gt = "", ""
    for turn in record.get("conversations", []):
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


def _swap_split(path: str, a: str, b: str):
    """Same file with split token `a` replaced by `b` in its name, if it exists."""
    p = Path(path)
    new_name = re.sub(rf"(?<![A-Za-z]){a}(?![A-Za-z])", b, p.name, count=1)
    if new_name == p.name:
        return None
    cand = p.with_name(new_name)
    return str(cand) if cand.exists() else None


def _split_label(path: str) -> str:
    name = Path(path).name
    for label in ("test", "val"):
        if re.search(rf"(?<![A-Za-z]){label}(?![A-Za-z])", name):
            return label
    return "eval"


def discover_eval_datasets(c_data: dict) -> list:
    """
    Every dataset in config['data'] with all of its val/test files.

    Keys: val_jsonl / test_jsonl / data_root (primary), then
    val_jsonl_N / test_jsonl_N / data_root_N for N = 2, 3, ... An optional
    dataset_name / dataset_name_N gives it a label; otherwise the label is
    inferred from the jsonl path. For each file given, its sibling split
    (val <-> test) is added automatically when that file exists.
    """
    datasets = []
    n = 1
    while True:
        sfx = "" if n == 1 else f"_{n}"
        given = [c_data[k] for k in (f"val_jsonl{sfx}", f"test_jsonl{sfx}") if c_data.get(k)]
        if not given:
            break
        paths = list(given)
        for p in given:
            for a, b in (("val", "test"), ("test", "val")):
                s = _swap_split(p, a, b)
                if s and s not in paths:
                    paths.append(s)
        name = c_data.get(f"dataset_name{sfx}") or Path(given[0]).parent.parent.name or Path(given[0]).stem
        datasets.append({
            "name": name,
            "data_root": c_data[f"data_root{sfx}"],
            "files": [(p, _split_label(p)) for p in paths],
        })
        n += 1
    return datasets


def list_checkpoints_desc(checkpoints_dir: str) -> list:
    """All step_* checkpoint dirs under checkpoints_dir, sorted newest-first."""
    step_dirs = []
    for dir_path in glob.glob(os.path.join(checkpoints_dir, "step_*")):
        try:
            step_num = int(os.path.basename(dir_path).replace("step_", ""))
        except ValueError:
            continue
        step_dirs.append((step_num, dir_path))
    if not step_dirs:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")
    step_dirs.sort(key=lambda x: x[0], reverse=True)
    return step_dirs


def load_checkpoint_weights(vlm, checkpoint_path: str, device) -> None:
    """Restore LoRA adapters, the projector and (if present) the GeoRoPE adapter."""
    vlm.hybrid_vicuna.load_adapter(checkpoint_path, adapter_name="default")
    vlm.hybrid_vicuna.set_adapter("default")
    vlm.projector.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "projector.pth"), map_location=device)
    )
    if vlm.georope_adapter is not None:
        georope_path = os.path.join(checkpoint_path, "georope_adapter.pth")
        if not os.path.exists(georope_path):
            raise FileNotFoundError(
                f"{georope_path} not found. If this checkpoint was trained without "
                f"GeoRoPE, set georope.enable: false in the config."
            )
        vlm.georope_adapter.load_state_dict(torch.load(georope_path, map_location=device))


def checkpoint_is_healthy(vlm, checkpoint_path: str, health_batch: dict, device) -> bool:
    """
    Load weights from checkpoint_path into vlm, then run one real forward
    pass on health_batch and check the loss/logits are finite. Returns False
    (without raising) on any load or NaN/Inf failure, so the caller can fall
    back to the next-older checkpoint.
    """
    try:
        load_checkpoint_weights(vlm, checkpoint_path, device)
    except Exception as e:
        log(f"    Failed to load weights from {checkpoint_path}: {e}")
        return False

    vlm.eval()
    try:
        gsd = health_batch.get("gsd_ratio")
        with torch.no_grad():
            output = vlm(
                sar_input=health_batch["sar_input"].to(device, dtype=torch.float32),
                input_ids=health_batch["input_ids"].to(device),
                attention_mask=health_batch["attention_mask"].to(device),
                labels=health_batch["labels"].to(device),
                gsd_ratio=gsd.to(device) if gsd is not None else None,
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


def build_health_check_batch(c_data: dict, c_train: dict, tokenizer, ref_gsd_m) -> dict:
    """One-sample batch (from the primary val set) used to sanity-check a checkpoint."""
    health_dataset = SARVLMDataset(
        c_data["val_jsonl"], c_data["data_root"], tokenizer, max_length=c_train["max_length"]
    )
    health_dataset.records = health_dataset.records[:1]
    return collate_fn([health_dataset[0]], tokenizer, ref_gsd_m=ref_gsd_m)


# ===========================================================================
# Metric functions
# ===========================================================================
_NUM = r"(-?\d+(?:\.\d+)?)"
BBOX_RE = re.compile(r"\[\s*" + r"\s*,\s*".join([_NUM] * 4) + r"\s*\]")


def parse_bbox(text: str):
    """First [x1, y1, x2, y2] box in the text, or None."""
    m = BBOX_RE.search(text)
    return [float(v) for v in m.groups()] if m else None


def bbox_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def exact_match(pred: str, gt: str) -> float:
    """Case-insensitive exact match after normalisation."""
    return 1.0 if normalise(pred) == normalise(gt) else 0.0


def compute_bleu(pred: str, gt: str) -> dict:
    """Sentence BLEU 1-4 using NLTK (smoothing method 1)."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoother = SmoothingFunction().method1
    ref = [normalise(gt).split()]
    hyp = normalise(pred).split()
    if not hyp:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    return {
        "bleu1": sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smoother),
        "bleu2": sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother),
        "bleu3": sentence_bleu(ref, hyp, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoother),
        "bleu4": sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother),
    }


_ROUGE = None


def compute_rouge(pred: str, gt: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = _ROUGE.score(normalise(gt), normalise(pred))
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def score_all(pred: str, gt: str) -> dict:
    """Compute all metrics for one (pred, gt) pair."""
    gt_box = parse_bbox(gt)
    iou = None
    if gt_box is not None:
        pred_box = parse_bbox(pred)
        iou = bbox_iou(pred_box, gt_box) if pred_box is not None else 0.0
    return {
        "exact_match": exact_match(pred, gt),
        **compute_bleu(pred, gt),
        **compute_rouge(pred, gt),
        "iou": iou,
    }


EMPTY_METRICS = {"exact_match": None, "bleu1": None, "bleu2": None, "bleu3": None, "bleu4": None,
                 "rouge1": None, "rouge2": None, "rougeL": None, "iou": None}


# ===========================================================================
# Model loading
# ===========================================================================
def load_model(config: dict, checkpoint_path: str = None):
    """Load SAR-VLM exactly as train.py builds it, then restore a checkpoint."""
    c_data = config["data"]
    c_model = config["model"]
    c_lora = config["lora"]
    c_train = config["training"]
    c_georope = config.get("georope", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Running on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(c_model["vicuna_path"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    log("Loading SAR Encoder...")
    sar_encoder = build_sar_encoder(
        checkpoint_path=c_model["encoder_checkpoint"],
        freeze=True,
        d_sar=c_model["d_sar"],
    )

    log("Loading SAR-VLM model (fp32, same as training)...")
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
        low_memory_load=True,
        use_georope_adapter=c_georope.get("enable", True),
        georope_bottleneck_dim=c_georope.get("bottleneck_dim", 256),
        georope_num_heads=c_georope.get("num_heads", 4),
        georope_gcc_alpha=c_georope.get("gcc_alpha", 0.5),
        georope_gfc_hidden_dim=c_georope.get("gfc_hidden_dim", 64),
        georope_zero_init_output=c_georope.get("zero_init_output", False),
    )
    vlm = vlm.to(device)
    vlm.eval()

    ref_gsd_m = c_georope.get("ref_gsd_m")

    if checkpoint_path is not None:
        # User explicitly chose this checkpoint -- load it as-is, no health check.
        global_step = int(Path(checkpoint_path).name.split("_")[-1])
        log(f"Loading specified checkpoint: {checkpoint_path}")
        load_checkpoint_weights(vlm, checkpoint_path, device)
    else:
        # Walk step_* checkpoints newest-first; skip any that fail to load or
        # produce a NaN/Inf loss on a real sample, until a healthy one is found.
        candidates = list_checkpoints_desc(c_train["save_dir"])
        log(f"Found {len(candidates)} checkpoint(s) in {c_train['save_dir']}")
        log("Building 1-sample health-check batch from the primary val set...")
        health_batch = build_health_check_batch(c_data, c_train, tokenizer, ref_gsd_m)

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

    log(f"Model loaded successfully (checkpoint step {global_step})")
    return vlm, tokenizer, device, ref_gsd_m, checkpoint_path, global_step


# ===========================================================================
# Batched inference
# ===========================================================================
def _keep_items(items):
    """DataLoader collate that returns the raw dataset items (picklable for workers)."""
    return items


@torch.no_grad()
def generate_batch(items: list, vlm, tokenizer, device, ref_gsd_m, max_new_tokens: int) -> list:
    """
    Generate answers for a batch of dataset items.

    Prompts are left-padded so every prompt ends right where generation
    starts; padding sits between the visual tokens and the prompt and is
    masked out, and position ids are derived from the attention mask, so
    each sample sees the same positions it would see unbatched.
    """
    batch = collate_fn(items, tokenizer, ref_gsd_m=ref_gsd_m)  # sar_input + gsd_ratio

    prompts = [it["input_ids"][it["labels"] == -100] for it in items]
    max_len = max(len(p) for p in prompts)
    prompt_ids = torch.full((len(prompts), max_len), tokenizer.pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
    for i, p in enumerate(prompts):
        prompt_ids[i, max_len - len(p):] = p
        prompt_mask[i, max_len - len(p):] = 1

    gsd = batch.get("gsd_ratio")
    output_ids = vlm.generate(
        sar_input=batch["sar_input"].to(device, dtype=torch.float32),
        input_ids=prompt_ids.to(device),
        attention_mask=prompt_mask.to(device),
        gsd_ratio=gsd.to(device) if gsd is not None else None,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # vlm.generate() returns ONLY the generated tokens (not the prompt)
    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in output_ids]


def run_benchmark(dataset, ds: dict, vlm, tokenizer, device, ref_gsd_m,
                  batch_size: int, max_new_tokens: int, pred_file) -> list:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_keep_items,
        num_workers=4,
        pin_memory=False,
    )

    results = []
    n = len(dataset)
    offset = 0
    for items in loader:
        recs = dataset.records[offset: offset + len(items)]
        offset += len(items)

        t0 = time.time()
        try:
            preds = generate_batch(items, vlm, tokenizer, device, ref_gsd_m, max_new_tokens)
            error = None
        except Exception as e:
            preds = [""] * len(items)
            error = str(e)
            log(f"  ERROR on batch starting at {recs[0][0].get('id')}: {e}")
        elapsed = (time.time() - t0) / len(items)

        for (record, _root, src_idx), pred in zip(recs, preds):
            question, gt = extract_question_and_gt(record)
            metrics = score_all(pred, gt) if error is None else dict(EMPTY_METRICS)
            row = {
                "id":           record.get("id", ""),
                "dataset":      ds["name"],
                "split":        ds["files"][src_idx][1],
                "category":     record.get("category", "unknown"),
                "image":        record.get("image", ""),
                "question":     question,
                "ground_truth": gt,
                "prediction":   pred,
                "elapsed_s":    round(elapsed, 3),
                "error":        error,
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
            }
            results.append(row)
            pred_file.write(json.dumps(row) + "\n")
        pred_file.flush()

        done = len(results)
        em = _safe_mean([r["exact_match"] for r in results])
        print(f"\r  [{ds['name']}] {done}/{n} ({done/n*100:.0f}%)  running EM={em or 0:.3f}          ",
              end="", flush=True)

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
    em = _safe_mean([r["exact_match"] for r in valid])
    lines = [
        f"{prefix}Samples          : {n}",
        f"{prefix}EXACT MATCH      : {em:.4f}  ({int(round(em * n))}/{n} correct)",
    ]
    for k in ["bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rouge2", "rougeL"]:
        lines.append(f"{prefix}{k.upper():<17}: {_safe_mean([r[k] for r in valid]):.4f}")

    ious = [r["iou"] for r in valid if r["iou"] is not None]
    if ious:
        acc = float(np.mean([v >= 0.5 for v in ious]))
        lines.append(f"{prefix}BBOX IoU (n={len(ious)}): mean {np.mean(ious):.4f},  acc@0.5 {acc:.4f}")
    return lines


# ===========================================================================
# Save + Report
# ===========================================================================
def save_results(results: list, out_dir: Path, tag: str, title: str) -> None:
    csv_path = out_dir / f"results_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log(f"CSV  -> {csv_path}")

    W = 64
    summary_lines = [
        "=" * W,
        f"  SAR-VLM Benchmark  --  {title}",
        f"  Run : {tag}",
        f"  Total samples : {len(results)}   Errors: {sum(1 for r in results if r['error'])}",
        "=" * W,
    ]

    datasets = {}
    for r in results:
        datasets.setdefault(r["dataset"], []).append(r)

    # Per-dataset, per-category breakdown
    for ds_name in sorted(datasets):
        ds_rows = datasets[ds_name]
        split_counts = {}
        for r in ds_rows:
            split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
        summary_lines += [
            "",
            "#" * W,
            f"  DATASET: {ds_name}  (n={len(ds_rows)}; " +
            ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items())) + ")",
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
    summary_path = out_dir / f"summary_{tag}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    log(f"Summary -> {summary_path}")


# ===========================================================================
# CLI + Entry point
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark SAR-VLM on val + test of every dataset")
    p.add_argument("--subset", type=int, default=None,
                   help="Evaluate only the first N samples of each dataset (smoke-test)")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle each dataset before taking --subset")
    p.add_argument("--category", type=str, default=None,
                   help="Filter to one question category")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to specific checkpoint (e.g., checkpoints/step_1000)")
    p.add_argument("--config", type=str, default="train_config.yaml",
                   help="Path to training config file")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Samples generated together per batch")
    p.add_argument("--max_new_tokens", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    import nltk
    for res in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(res.split("/")[-1], quiet=True)

    log("=" * 60)
    log("SAR-VLM Benchmark Runner (val + test)")
    log("=" * 60)

    os.chdir(REPO_DIR)
    config = load_config(args.config)
    c_data = config["data"]
    c_train = config["training"]

    vlm, tokenizer, device, ref_gsd_m, checkpoint_path, global_step = load_model(config, args.checkpoint)

    # Outputs go in the experiment's main directory: the folder that contains
    # the checkpoints folder (e.g. GeoRoPE_CKA_Data/), named by step.
    out_dir = Path(checkpoint_path).resolve().parent.parent
    tag = f"step{global_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pred_path = out_dir / f"predictions_{tag}.jsonl"
    log(f"Predictions will be written to {pred_path}")

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated / "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

    eval_datasets = discover_eval_datasets(c_data)
    if not eval_datasets:
        raise ValueError("No val_jsonl / test_jsonl found in config['data']")
    log(f"\nFound {len(eval_datasets)} dataset(s):")
    for ds in eval_datasets:
        log(f"  {ds['name']}:")
        for path, split in ds["files"]:
            log(f"    [{split}] {path}")
        if len(ds["files"]) < 2:
            log(f"    WARNING: only one split found for '{ds['name']}'")

    all_results = []
    pred_file = open(pred_path, "w")
    for ds in eval_datasets:
        dataset = SARVLMDataset(
            sources=[{"jsonl_path": p, "data_root": ds["data_root"], "sample_size": None}
                     for p, _ in ds["files"]],
            tokenizer=tokenizer,
            max_length=c_train["max_length"],
        )
        log(f"\n'{ds['name']}': {len(dataset)} samples (val + test)")

        if args.category:
            dataset.records = [r for r in dataset.records if r[0].get("category") == args.category]
            log(f"  After category filter '{args.category}': {len(dataset.records)}")

        if args.shuffle:
            random.shuffle(dataset.records)

        if args.subset is not None:
            dataset.records = dataset.records[: args.subset]
            log(f"  Subset: {len(dataset.records)}")

        if not dataset.records:
            log(f"  Skipping '{ds['name']}' -- no samples after filtering")
            continue

        t0 = time.time()
        results = run_benchmark(dataset, ds, vlm, tokenizer, device, ref_gsd_m,
                                args.batch_size, args.max_new_tokens, pred_file)
        elapsed = time.time() - t0
        log(f"Done '{ds['name']}' in {elapsed:.1f}s  ({elapsed/len(results):.2f}s/sample)")
        all_results.extend(results)
    pred_file.close()
    log(f"Predictions -> {pred_path}")

    if not all_results:
        log("No samples evaluated.")
        return
    save_results(all_results, out_dir, tag, f"VAL + TEST, checkpoint step {global_step}")
    log("\nAll done!")


if __name__ == "__main__":
    main()
