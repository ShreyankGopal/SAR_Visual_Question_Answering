#!/usr/bin/env python3
"""
run_benchmark.py
================
Benchmarks GeoChat (and ablation variants) on the generated test.jsonl / val.jsonl datasets.

Ablation modes (configured via benchmarks_config.yaml):
  baseline      : GeoChat + CLIP ViT  (original)
  sar_encoder   : GeoChat + SAR encoder (SwinV2-Base), no extra projector  [Bench 1]
  sar_projector : GeoChat + SAR encoder + trained MLP projector             [Bench 2]

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
  python BenchMarks_GeoChat/run_benchmark.py                          # uses benchmarks_config.yaml
  python BenchMarks_GeoChat/run_benchmark.py --config path/to/cfg.yaml
  python BenchMarks_GeoChat/run_benchmark.py --subset 10             # smoke-test
  python BenchMarks_GeoChat/run_benchmark.py --split val
  python BenchMarks_GeoChat/run_benchmark.py --split both

Outputs (all saved to the configured save_dir)
  results_<mode>_<split>_<ts>.csv   – per-sample rows with all metric scores
  results_<mode>_<split>_<ts>.json  – same, JSON format
  summary_<mode>_<split>_<ts>.txt   – dataset-wide + per-category report
"""

# ===========================================================================
# Imports
# ===========================================================================
import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

# Tell the CUDA allocator to release cached blocks more aggressively.
# This reduces fragmentation that can cause OOM even when total usage is
# nominally within budget.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Resolved at start-up from config
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_DIR    = SCRIPT_DIR.parent
GEOCHAT_DIR = REPO_DIR / "GeoChat"

sys.path.insert(0, str(GEOCHAT_DIR))


# ===========================================================================
# Config loading
# ===========================================================================

def load_config(cfg_path: Path) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_ablation_mode(cfg: dict) -> str:
    """
    Return the active ablation mode string ('baseline', 'sar_encoder',
    'sar_projector') from the ablation section.  Exactly one must be true.
    """
    ablation = cfg.get("ablation", {})
    active = [k for k, v in ablation.items() if v is True]
    if len(active) != 1:
        raise ValueError(
            f"Exactly one ablation mode must be set to true in config. "
            f"Found: {active}"
        )
    return active[0]


def build_system_prompt(cfg: dict) -> str:
    classes = cfg["prompts"]["land_cover_classes"]
    return (
        "You are a remote sensing expert analysing SAR (Synthetic Aperture Radar) "
        "satellite imagery for land-cover classification. "
        "The possible land-cover classes are: "
        + ", ".join(classes) + ". "
        "Answer concisely and accurately based on the image."
    )


# ===========================================================================
# Logging
# ===========================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Data helpers
# ===========================================================================

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
            question = turn["value"].replace("<image>\n", "").strip()
        elif turn["from"] == "gpt":
            gt = turn["value"].strip()
    return question, gt


# ===========================================================================
# Text normalisation
# ===========================================================================

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def is_single_word_gt(gt: str) -> bool:
    """True if the ground truth normalises to a single token."""
    return len(normalise(gt).split()) == 1


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
    ref = [normalise(gt).split()]
    hyp = normalise(pred).split()
    if not hyp:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    return {
        "bleu1": sentence_bleu(ref, hyp, weights=(1, 0, 0, 0),                    smoothing_function=smoother),
        "bleu2": sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0),               smoothing_function=smoother),
        "bleu3": sentence_bleu(ref, hyp, weights=(1/3, 1/3, 1/3, 0),             smoothing_function=smoother),
        "bleu4": sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25),       smoothing_function=smoother),
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
# Model loading — one function per ablation mode
# ===========================================================================

def _load_geochat_base(cfg: dict):
    """Load base GeoChat tokenizer + model + image_processor from HuggingFace."""
    from geochat.conversation import conv_templates, Chat
    from geochat.model.builder import load_pretrained_model
    from geochat.mm_utils import get_model_name_from_path

    model_path = cfg["model"]["geochat_path"]
    model_name = get_model_name_from_path(model_path)
    log(f"Model name: {model_name}")

    log("Loading pretrained GeoChat LLM (~60 s) ...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, False, False,
        device=cfg["model"]["device"]
    )
    return tokenizer, model, image_processor, conv_templates


def load_model_baseline(cfg: dict):
    """Baseline: GeoChat + CLIP ViT (original model, no changes)."""
    log("── Ablation: BASELINE (CLIP ViT) ──")
    tokenizer, model, image_processor, conv_templates = _load_geochat_base(cfg)

    log("Reloading CLIP position embeddings ...")
    model.get_vision_tower().load_model()
    model.get_vision_tower().to(device=model.device, dtype=model.dtype)
    model = model.eval()
    torch.cuda.empty_cache()

    from geochat.conversation import Chat
    chat = Chat(model, image_processor, tokenizer, device=cfg["model"]["device"])
    return chat, conv_templates


def load_model_sar_encoder(cfg: dict):
    """SAR encoder only: replace CLIP ViT with SwinV2-Base SAR encoder."""
    log("── Ablation: SAR ENCODER only (Bench 1) ──")
    tokenizer, model, image_processor, conv_templates = _load_geochat_base(cfg)

    sar_enc_path = (SCRIPT_DIR / cfg["model"]["sar_encoder_weights"]).resolve()
    if not sar_enc_path.exists():
        raise FileNotFoundError(f"SAR encoder weights not found: {sar_enc_path}")

    # ── Explicitly evict old CLIP tower from GPU before loading SAR encoder ──
    # Without this, both towers occupy VRAM simultaneously during the swap.
    log("Evicting old CLIP vision tower from GPU ...")
    old_tower = model.model.vision_tower
    if hasattr(old_tower, 'to'):
        old_tower.to('cpu')       # move weights off GPU first
    model.model.vision_tower = None
    del old_tower
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  GPU after eviction: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    log(f"Loading SAR encoder from: {sar_enc_path}")
    from geochat.model.multimodal_encoder.sar_encoder import SARVisionTower
    model.model.vision_tower = SARVisionTower(str(sar_enc_path))
    model.model.vision_tower.load_model()
    model.model.vision_tower.to(device=model.device, dtype=torch.float16)
    log("SAR encoder loaded successfully.")

    model = model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  GPU after SAR encoder: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    from geochat.conversation import Chat
    chat = Chat(model, image_processor, tokenizer, device=cfg["model"]["device"])
    return chat, conv_templates


def load_model_sar_projector(cfg: dict):
    """SAR encoder + trained MLP projector: replace both CLIP ViT and mm_projector."""
    log("── Ablation: SAR ENCODER + MLP PROJECTOR (Bench 2) ──")
    tokenizer, model, image_processor, conv_templates = _load_geochat_base(cfg)

    # ── 1. Evict old CLIP tower before loading SAR encoder ─────────────────
    log("Evicting old CLIP vision tower from GPU ...")
    old_tower = model.model.vision_tower
    if hasattr(old_tower, 'to'):
        old_tower.to('cpu')
    model.model.vision_tower = None
    del old_tower
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  GPU after eviction: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    # ── 2. Load SAR encoder ─────────────────────────────────────────────────
    sar_enc_path = (SCRIPT_DIR / cfg["model"]["sar_encoder_weights"]).resolve()
    if not sar_enc_path.exists():
        raise FileNotFoundError(f"SAR encoder weights not found: {sar_enc_path}")

    log(f"Loading SAR encoder from: {sar_enc_path}")
    from geochat.model.multimodal_encoder.sar_encoder import SARVisionTower
    model.model.vision_tower = SARVisionTower(str(sar_enc_path))
    model.model.vision_tower.load_model()
    model.model.vision_tower.to(device=model.device, dtype=torch.float16)
    log("SAR encoder loaded successfully.")
    log(f"  GPU after SAR encoder: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    # ── 3. Swap mm_projector for trained SARProjector ───────────────────────
    proj_path = (SCRIPT_DIR / cfg["model"]["sar_projector_weights"]).resolve()
    if not proj_path.exists():
        raise FileNotFoundError(f"Projector weights not found: {proj_path}")

    log(f"Loading trained SAR projector from: {proj_path}")
    from geochat.model.multimodal_projector.sar_projector import SARProjector

    # Load projector on CPU first to avoid double VRAM spike
    sar_projector = SARProjector(d_sar=1024, llm_hidden_size=4096, hidden_dim=4096)
    proj_state = torch.load(proj_path, map_location="cpu", weights_only=True)
    for unwrap_key in ("model", "state_dict", "projector", "projector_state_dict"):
        if isinstance(proj_state, dict) and unwrap_key in proj_state:
            proj_state = proj_state[unwrap_key]
            break
    missing, unexpected = sar_projector.load_state_dict(proj_state, strict=True)
    if missing:
        log(f"  WARNING — missing projector keys: {missing}")
    if unexpected:
        log(f"  WARNING — unexpected projector keys: {unexpected}")
    del proj_state   # free CPU RAM immediately

    # Move old CLIP projector off GPU, then replace
    old_proj = model.model.mm_projector
    if hasattr(old_proj, 'to'):
        old_proj.to('cpu')
    model.model.mm_projector = None
    del old_proj
    gc.collect()
    torch.cuda.empty_cache()

    sar_projector = sar_projector.to(device=model.device, dtype=torch.float16)
    sar_projector.eval()
    model.model.mm_projector = sar_projector
    log("SAR projector loaded and attached successfully.")
    log(f"  GPU after projector swap: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    model = model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    from geochat.conversation import Chat
    chat = Chat(model, image_processor, tokenizer, device=cfg["model"]["device"])
    return chat, conv_templates


def load_model(cfg: dict, mode: str):
    """Dispatch to the correct model loader based on ablation mode."""
    loaders = {
        "baseline":      load_model_baseline,
        "sar_encoder":   load_model_sar_encoder,
        "sar_projector": load_model_sar_projector,
    }
    return loaders[mode](cfg)


# ===========================================================================
# Image loading — adapts to ablation mode
# ===========================================================================

def load_image_for_mode(img_path: Path, mode: str):
    """
    Returns the image in the format expected by the active ablation mode.
    - baseline      : PIL RGB image (CLIP ViT expects RGB)
    - sar_encoder   : TIFF float tensor [1, 1, 512, 512] (grayscale)
    - sar_projector : TIFF float tensor [1, 1, 512, 512] (grayscale)
    """
    if mode == "baseline":
        return Image.open(img_path).convert("RGB")
    else:
        # SAR modes: load single-channel TIFF as float tensor
        import tifffile
        img_np = tifffile.imread(img_path)
        img_t  = torch.from_numpy(img_np).float()
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)   # [1, 512, 512]
        img_t = img_t.unsqueeze(0)       # [1, 1, 512, 512]
        return img_t


# ===========================================================================
# Single-record inference
# ===========================================================================

@torch.inference_mode()   # disables grad tracking + autograd engine overhead
def run_one(record: dict, chat, conv_templates, cfg: dict, mode: str) -> str:
    patches_root = Path(cfg["data"]["patches_root"])
    img_path     = patches_root / record["image"]
    question, gt = extract_question_and_gt(record)

    # Append one-word hint when GT is a single word
    one_word_suffix = cfg["prompts"]["one_word_suffix"]
    if is_single_word_gt(gt):
        question = question + one_word_suffix

    # Fresh conversation state with domain-aware system prompt
    conv_tpl   = cfg["inference"]["conv_template"]
    chat_state = conv_templates[conv_tpl].copy()
    chat_state.system = build_system_prompt(cfg)

    image    = load_image_for_mode(img_path, mode)
    img_list = []
    chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    chat.ask(question, chat_state)

    if cfg.get("debug", {}).get("verbose"):
        log(f"  [DBG] id={record.get('id')}  img={img_path}")
        log(f"  [DBG] Q: {question}")
        log(f"  [DBG] GT: {gt}")

    output = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=cfg["inference"]["max_new_tokens"],
        max_length=cfg["inference"]["max_length"],
    )

    if cfg.get("debug", {}).get("verbose"):
        log(f"  [DBG] Pred: {output}")

    # Free activation memory immediately after each forward pass
    del img_list, image, chat_state
    torch.cuda.empty_cache()
    return output.strip() if output else ""


# ===========================================================================
# Main benchmark loop
# ===========================================================================

def run_benchmark(records: list, chat, conv_templates, cfg: dict,
                  mode: str, split_name: str) -> list:
    results = []
    n = len(records)

    for idx, record in enumerate(records):
        rec_id   = record.get("id", f"idx_{idx}")
        category = record.get("category", "unknown")
        question, gt = extract_question_and_gt(record)

        t0 = time.time()
        try:
            pred    = run_one(record, chat, conv_templates, cfg, mode)
            elapsed = time.time() - t0
            metrics = score_all(pred, gt)
            error   = None
        except Exception as e:
            pred    = ""
            elapsed = time.time() - t0
            metrics = {
                "is_single_word": is_single_word_gt(gt),
                "exact_match": None,
                "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
                "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
            }
            error = str(e)
            log(f"  ERROR on {rec_id}: {e}")

        # Periodic Python GC to prevent reference-count leaks from accumulating
        if (idx + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        row = {
            "id":           rec_id,
            "split":        split_name,
            "ablation_mode": mode,
            "category":     category,
            "image":        record.get("image", ""),
            "question":     question,
            "ground_truth": gt,
            "prediction":   pred,
            "elapsed_s":    round(elapsed, 2),
            "error":        error,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        }
        results.append(row)

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

    n   = len(valid)
    sw  = [r for r in valid if r["is_single_word"]]
    mw  = [r for r in valid if not r["is_single_word"]]

    lines = []
    lines.append(f"{prefix}Samples          : {n}  (single-word GT: {len(sw)}, multi-word: {len(mw)})")

    # BLEU (all samples)
    for k in ["bleu1", "bleu2", "bleu3", "bleu4"]:
        v = _safe_mean([r[k] for r in valid])
        lines.append(f"{prefix}{k.upper():<10}: {v:.4f}")

    # ROUGE (all samples)
    for k in ["rouge1", "rouge2", "rougeL"]:
        v = _safe_mean([r[k] for r in valid])
        lines.append(f"{prefix}{k.upper():<10}: {v:.4f}")

    # Exact match (single-word only)
    if sw:
        em = _safe_mean([r["exact_match"] for r in sw])
        lines.append(
            f"{prefix}EXACT MATCH (single-word GT only): "
            f"{em:.4f}  ({int(em*len(sw))}/{len(sw)} correct)"
        )

    return lines


# ===========================================================================
# Save + Report
# ===========================================================================

def save_results(results: list, split_name: str, mode: str, cfg: dict) -> None:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(cfg["output"].get("save_dir") or SCRIPT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"results_{mode}_{split_name}_{ts}"
    base      = save_dir / base_name

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

    # ── Summary ──────────────────────────────────────────────────────────────
    W = 68
    summary_lines = [
        "=" * W,
        f"  GeoChat Ablation Benchmark  --  mode={mode.upper()}  split={split_name.upper()}",
        f"  Timestamp : {ts}",
        f"  Total samples : {len(results)}   Errors: {sum(1 for r in results if r['error'])}",
        "=" * W,
        "",
        "DATASET-WIDE STATISTICS",
        "-" * W,
    ]
    summary_lines += _stats_block(results)

    # Per-category breakdown (covers all 9 categories including the 2 new ones)
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
    summary_path = save_dir / f"summary_{mode}_{split_name}_{ts}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    log(f"Summary -> {summary_path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark GeoChat (ablation variants) on LULC QA datasets"
    )
    p.add_argument(
        "--config", type=Path,
        default=SCRIPT_DIR / "benchmarks_config.yaml",
        help="Path to benchmarks_config.yaml",
    )
    p.add_argument("--split",    choices=["test", "val", "both"], default="test")
    p.add_argument("--subset",   type=int, default=None,
                   help="Evaluate only the first N samples (smoke-test)")
    p.add_argument("--shuffle",  action="store_true",
                   help="Shuffle before taking --subset")
    p.add_argument("--category", type=str, default=None,
                   help="Filter to one question category")
    return p.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg_path = args.config.resolve()
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg  = load_config(cfg_path)
    mode = resolve_ablation_mode(cfg)

    # ── Ensure NLTK data ─────────────────────────────────────────────────────
    import nltk
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)

    log("=" * 60)
    log(f"GeoChat Ablation Benchmark — mode: {mode.upper()}")
    log(f"Config : {cfg_path}")
    log("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────────
    chat, conv_templates = load_model(cfg, mode)

    gpu_alloc = torch.cuda.memory_allocated() / 1024**3
    gpu_res   = torch.cuda.memory_reserved()  / 1024**3
    log(f"GPU: {gpu_alloc:.2f} GB allocated / {gpu_res:.2f} GB reserved")

    # ── Run splits ───────────────────────────────────────────────────────────
    data_dir = Path(cfg["data"]["data_dir"])
    splits   = ["test", "val"] if args.split == "both" else [args.split]

    for split in splits:
        jsonl_path = data_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            log(f"WARNING: {jsonl_path} not found — skipping")
            continue

        log(f"\nLoading {split}.jsonl ...")
        records = load_jsonl(jsonl_path)
        log(f"  Total records: {len(records)}")

        if args.category:
            records = [r for r in records if r.get("category") == args.category]
            log(f"  After category filter '{args.category}': {len(records)}")

        if args.shuffle:
            import random
            random.shuffle(records)

        if args.subset is not None:
            records = records[:args.subset]
            log(f"  Subset: {len(records)}")

        if not records:
            log("  No records to evaluate — skipping.")
            continue

        log(f"\nStarting evaluation on {len(records)} records ...\n")
        t0      = time.time()
        results = run_benchmark(records, chat, conv_templates, cfg, mode, split)
        elapsed = time.time() - t0
        log(f"Done in {elapsed:.1f}s  ({elapsed/len(results):.2f}s/sample)")

        save_results(results, split, mode, cfg)

    log("\nAll done!")


if __name__ == "__main__":
    main()
