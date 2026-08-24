#!/usr/bin/env python3
"""
run_benchmark.py
================
Benchmarks GeoChat on the generated test.jsonl / val.jsonl datasets.

Metrics
-------
  - Exact Match (EM)   : case-insensitive, for single-word ground-truth answers
  - BLEU-1/2/3/4       : sentence BLEU via NLTK
  - ROUGE-1/2/L        : via rouge-score

Prompt strategy
---------------
  - A system prompt listing all 8 land-cover classes is prepended to every query.
  - If the ground truth is a single word (after normalisation), the question is
    appended with "Answer in one word only."

Usage
-----
  python BenchMarks_GeoChat/run_benchmark.py --subset 10   # smoke-test
  python BenchMarks_GeoChat/run_benchmark.py --split test
  python BenchMarks_GeoChat/run_benchmark.py --split val
  python BenchMarks_GeoChat/run_benchmark.py --split both

Outputs (all saved to BenchMarks_GeoChat/)
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
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_DIR     = SCRIPT_DIR.parent
GEOCHAT_DIR  = REPO_DIR / "GeoChat"
DATA_DIR     = Path("/home/saishruti/Research1/Shreyank_20_credit/DataGen/data")
PATCHES_ROOT = Path("/home/saishruti/Research1/Shreyank_20_credit/DataGen")

sys.path.insert(0, str(GEOCHAT_DIR))

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
LAND_COVER_CLASSES = [
    "Bareland", "Rangeland", "Developed Space", "Road",
    "Tree", "Water", "Agriculture Land", "Building",
]

SYSTEM_PROMPT = (
    "You are a remote sensing expert analysing SAR (Synthetic Aperture Radar) "
    "satellite imagery for land-cover classification. "
    "The possible land-cover classes are: "
    + ", ".join(LAND_COVER_CLASSES) + ". "
    "Answer concisely and accurately based on the image."
)

ONE_WORD_SUFFIX = " Answer in one word only."

# ===========================================================================
# Helpers
# ===========================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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
            question = turn["value"].replace("<image>\n", "").strip()
        elif turn["from"] == "gpt":
            gt = turn["value"].strip()
    return question, gt


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
def load_model():
    log("Importing GeoChat modules ...")
    from geochat.conversation import conv_templates, Chat
    from geochat.model.builder import load_pretrained_model
    from geochat.mm_utils import get_model_name_from_path

    model_path = "MBZUAI/geochat-7B"
    model_name = get_model_name_from_path(model_path)
    log(f"Detected model name: {model_name}")

    log("Loading pretrained GeoChat model (~60 s) ...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, False, False, device="cuda"
    )

    log("Reloading CLIP position embeddings ...")
    model.get_vision_tower().load_model()
    model.get_vision_tower().to(device=model.device, dtype=model.dtype)
    model = model.eval()

    log("Creating Chat object ...")
    chat = Chat(model, image_processor, tokenizer, device="cuda")
    return chat, conv_templates


# ===========================================================================
# Inference for one record
# ===========================================================================
def run_one(record: dict, chat, conv_templates) -> str:
    img_path = PATCHES_ROOT / record["image"]
    question, gt = extract_question_and_gt(record)

    # Append one-word instruction if GT is a single word
    if is_single_word_gt(gt):
        question = question + ONE_WORD_SUFFIX

    # Fresh conversation state; override system prompt with domain-aware version
    chat_state = conv_templates["llava_v1"].copy()
    chat_state.system = SYSTEM_PROMPT

    image = Image.open(img_path).convert("RGB")
    img_list = []
    chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    chat.ask(question, chat_state)

    output = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=64,    # short answers; can raise for descriptive categories
        max_length=2000,
    )
    return output.strip() if output else ""


# ===========================================================================
# Main benchmark loop
# ===========================================================================
def run_benchmark(records: list, chat, conv_templates, split_name: str) -> list:
    results = []
    n = len(records)

    for idx, record in enumerate(records):
        rec_id   = record.get("id", f"idx_{idx}")
        category = record.get("category", "unknown")
        question, gt = extract_question_and_gt(record)

        t0 = time.time()
        try:
            pred    = run_one(record, chat, conv_templates)
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
        f"  GeoChat Benchmark  --  {split_name.upper()} split",
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
    p = argparse.ArgumentParser(description="Benchmark GeoChat on LULC QA datasets")
    p.add_argument("--split",    choices=["test", "val", "both"], default="test")
    p.add_argument("--subset",   type=int, default=None,
                   help="Evaluate only the first N samples (smoke-test)")
    p.add_argument("--shuffle",  action="store_true",
                   help="Shuffle before taking --subset")
    p.add_argument("--category", type=str, default=None,
                   help="Filter to one question category")
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
    log("GeoChat Benchmark Runner")
    log(f"System prompt : {SYSTEM_PROMPT[:80]}...")
    log("=" * 60)

    chat, conv_templates = load_model()

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
        results = run_benchmark(records, chat, conv_templates, split)
        elapsed = time.time() - t0
        log(f"Done in {elapsed:.1f}s  ({elapsed/len(records):.2f}s/sample)")

        save_results(results, split)

    log("\nAll done!")


if __name__ == "__main__":
    main()
