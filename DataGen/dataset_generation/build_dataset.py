"""
build_dataset.py — Orchestrator for the full dataset generation pipeline.

Usage:
    python dataset_generation/build_dataset.py --config dataset_generation/config.yaml

Runs Phases 1–10 in sequence for a sample of train tiles.
"""

import os
import sys
import json
import random
import argparse
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── suppress rasterio georef warnings ─────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_config, read_sar, read_mask, stretch_sar, mask_to_rgb,
    list_tif_files, ensure_dir, save_json, append_jsonl,
    get_class_map, get_class_colors, proportion_to_bin,
    split_into_patches, save_patch_tif,
)
from inspect_dataset        import inspect as run_inspect
from extract_global_facts   import extract_global_facts
from extract_regional_facts import extract_regional_facts, find_region_with_most
from extract_sar_facts      import compute_global_threshold, extract_sar_facts
from generate_questions     import generate_qa, qa_to_conversation
from validate_dataset       import validate


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_output_dirs(config: dict) -> dict:
    base = config["output"]["base_dir"]
    patches_root = os.path.join(base, config["output"].get("patches_dir", "patches"))
    dirs = {
        "reports":      os.path.join(base, config["output"]["reports_dir"]),
        "data":         os.path.join(base, config["output"]["data_dir"]),
        "vis":          os.path.join(base, config["output"]["vis_dir"]),
        "facts":        os.path.join(base, config["output"]["facts_dir"]),
        "patches_sar":  os.path.join(patches_root, "sar"),
        "patches_lbl":  os.path.join(patches_root, "labels"),
        "patches_root": patches_root,
    }
    for d in dirs.values():
        ensure_dir(d)
    return dirs


def split_tiles(tiles: list, config: dict) -> dict:
    """Split sampled tile list into train / val / test using fixed seed.
    Since val/labels doesn't exist in this dataset, val and test are both
    carved from the sampled train tiles.
    """
    seed       = config["pipeline"]["random_seed"]
    test_frac  = config["pipeline"]["test_fraction"]
    val_frac   = config["pipeline"].get("val_fraction", 0.15)

    rng = random.Random(seed)
    shuffled = tiles[:]
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_test - n_val

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def build_record(patch_id: str, sar_rel: str, label_rel: str,
                 tile: str, patch_suffix: str, patch_position: str,
                 split: str,
                 global_facts: dict, regional_facts: dict,
                 sar_class_facts: dict,
                 qa_list: list, config: dict, sample_idx: int) -> list:
    """Convert facts + QA list into JSONL-ready records (one per QA pair)."""
    records = []
    for idx, qa in enumerate(qa_list):
        rec_id = f"lulc_{sample_idx:05d}_{idx:03d}"

        gt_facts = {
            "image_id":       patch_id,
            "original_tile":  tile,
            "patch_suffix":   patch_suffix,
            "patch_position": patch_position,
            "dominant_class": global_facts["dominant_class"],
            "classes_present": global_facts["classes_present"],
            "n_classes_present": global_facts["n_classes_present"],
            "class_proportions": {
                c: v["proportion"]
                for c, v in global_facts["global"].items()
                if v["present"]
            },
            # SAR-derived per-class facts (relative amplitude stats, not calibrated)
            "sar_class_facts": {
                cls: {
                    "mean_sar_response":    v["mean_sar_response"],
                    "bright_pixel_fraction": v["bright_pixel_fraction"],
                }
                for cls, v in sar_class_facts.items()
            },
        }
        if "region" in qa:
            rn = qa["region"]
            gt_facts["region"] = rn
            gt_facts["region_dominant"] = regional_facts[rn].get("dominant_class")
            gt_facts["region_proportions"] = regional_facts[rn].get("class_proportions", {})
        if "bbox" in qa:
            gt_facts["bbox"] = qa["bbox"]

        records.append({
            "id":      rec_id,
            "image":   sar_rel,
            "mask":    label_rel,
            "category":   qa["category"],
            "ground_truth_facts": gt_facts,
            "conversations": [
                {"from": "human", "value": qa["question"]},
                {"from": "gpt",   "value": qa["answer"]},
            ],
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG = "#1a1a2e"
TEXT_CLR = "#e0e0e0"
ACCENT   = "#4fc3f7"


def visualise_tile(sar: np.ndarray, mask: np.ndarray,
                   qa_list: list, patch_id: str,
                   config: dict, out_path: str):
    """Generate a panel for a single 512×512 patch: SAR | Label RGB | QA card."""
    class_colors = get_class_colors(config)
    class_map    = get_class_map(config)

    sar_disp  = stretch_sar(sar)
    label_rgb = mask_to_rgb(mask, class_colors)

    fig = plt.figure(figsize=(20, 9), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.02, right=0.98,
                            top=0.88, bottom=0.05,
                            wspace=0.05)

    # ── Left: SAR ────────────────────────────────────────────────────────────
    ax_sar = fig.add_subplot(gs[0])
    ax_sar.imshow(sar_disp, cmap="gray")
    ax_sar.set_title("SAR Image", color=ACCENT, fontsize=13,
                     fontweight="bold", pad=6)
    ax_sar.axis("off")

    # ── Center: Label ─────────────────────────────────────────────────────────
    ax_lbl = fig.add_subplot(gs[1])
    ax_lbl.imshow(label_rgb)
    ax_lbl.set_title("Label (RGB)", color=ACCENT, fontsize=13,
                     fontweight="bold", pad=6)
    ax_lbl.axis("off")

    # Mini legend below label
    present_ids = [i for i in class_map if np.any(mask == i)]
    handles = [
        mpatches.Patch(
            color=[c / 255 for c in class_colors[i]],
            label=f"{i}: {class_map[i]}"
        )
        for i in sorted(present_ids)
    ]
    ax_lbl.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2, frameon=True,
        facecolor=DARK_BG, edgecolor="#555",
        fontsize=7.5, labelcolor=TEXT_CLR,
    )

    # ── Right: QA card ───────────────────────────────────────────────────────
    ax_qa = fig.add_subplot(gs[2])
    ax_qa.set_facecolor("#0d1117")
    ax_qa.axis("off")
    ax_qa.set_title("QA Pairs", color=ACCENT, fontsize=13,
                    fontweight="bold", pad=6)

    # Draw QA pairs as text blocks
    CATEGORY_COLORS = {
        "global_classification": "#80cbc4",
        "global_quantitative":   "#fff176",
        "regional_vqa":          "#ef9a9a",
        "region_grounding":      "#ce93d8",
        "comparative_spatial":   "#90caf9",
        "sar_observation":       "#ffcc80",   # amber — SAR amplitude observation
        "sar_comparative":       "#f48fb1",   # pink  — SAR comparative
    }

    y = 0.97
    for qa in qa_list[:10]:
        cat   = qa.get("category", "")
        q_txt = qa["question"].replace("<image>\n", "")
        a_txt = qa["answer"]
        cat_color = CATEGORY_COLORS.get(cat, "#aaa")

        # Category label
        ax_qa.text(0.02, y, f"[{cat.replace('_',' ').upper()}]",
                   transform=ax_qa.transAxes, fontsize=6,
                   color=cat_color, fontweight="bold", va="top")
        y -= 0.032

        # Question
        q_wrapped = _wrap(q_txt, 55)
        ax_qa.text(0.02, y, f"Q: {q_wrapped}",
                   transform=ax_qa.transAxes, fontsize=7.5,
                   color=TEXT_CLR, va="top", wrap=False)
        n_lines = q_wrapped.count("\n") + 1
        y -= 0.028 * n_lines

        # Answer
        ax_qa.text(0.02, y, f"A: {a_txt}",
                   transform=ax_qa.transAxes, fontsize=7.5,
                   color="#a5d6a7", va="top", fontstyle="italic")
        y -= 0.045

        if y < 0.05:
            break

    fig.suptitle(f"OpenEarthMap — {patch_id}",
                 color="white", fontsize=14, fontweight="bold", y=0.96)

    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)


def _wrap(text: str, width: int) -> str:
    """Simple word-wrap."""
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = line + " " + w if line else w
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Distribution reports
# ─────────────────────────────────────────────────────────────────────────────

def save_distribution_reports(all_records: list, config: dict, dirs: dict):
    cat_dist   = defaultdict(int)
    cls_dist   = defaultdict(int)
    region_dist = defaultdict(int)

    for rec in all_records:
        cat_dist[rec["category"]] += 1
        for cls in rec["ground_truth_facts"].get("classes_present", []):
            cls_dist[cls] += 1
        if "region" in rec["ground_truth_facts"]:
            region_dist[rec["ground_truth_facts"]["region"]] += 1

    save_json(dict(cat_dist),    os.path.join(dirs["reports"], "question_distribution.json"))
    save_json(dict(cls_dist),    os.path.join(dirs["reports"], "class_distribution.json"))
    save_json(dict(region_dist), os.path.join(dirs["reports"], "region_distribution.json"))
    print("✓ Distribution reports saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: dict, do_vis: bool = True):
    dirs     = make_output_dirs(config)
    cfg_ds   = config["dataset"]
    cfg_pip  = config["pipeline"]
    seed     = cfg_pip["random_seed"]
    n_sample = cfg_pip["sample_count"]
    # Visualise flag: CLI --no-vis overrides config; config default is True
    do_vis = do_vis and cfg_pip.get("visualise", True)

    random.seed(seed)
    np.random.seed(seed)

    # ── Phase 1: Inspect ─────────────────────────────────────────────────────
    print("\n══ Phase 1: Inspecting dataset ══")
    run_inspect(config)

    # ── Sample tiles ─────────────────────────────────────────────────────────
    sar_dir   = cfg_ds["train_sar_dir"]
    label_dir = cfg_ds["train_label_dir"]
    all_tiles = list_tif_files(sar_dir)
    if n_sample is None:
        sampled = all_tiles
        print(f"\nUsing all {len(sampled)} tiles from train (sample_count: null).")
    else:
        sampled = random.sample(all_tiles, min(n_sample, len(all_tiles)))
        print(f"\nSampled {len(sampled)} tiles from train.")

    # ── Phase 2b: Compute global SAR brightness threshold T ──────────────────
    # T is the dataset-wide percentile threshold used to define "bright" pixels.
    # Computed ONCE from all sampled SAR images; applied consistently to all
    # images and classes. NOT image-specific or class-specific.
    sar_cfg        = config.get("sar", {})
    bright_pctile  = float(sar_cfg.get("bright_pixel_percentile", 90))
    sampled_sar_paths = [os.path.join(sar_dir, t) for t in sampled]
    print(f"\nComputing global SAR brightness threshold "
          f"(p{bright_pctile:.0f} of {len(sampled)} tiles) …")
    global_sar_threshold = compute_global_threshold(sampled_sar_paths, bright_pctile)
    print(f"  Global SAR threshold T = {global_sar_threshold:.2f} "
          f"(relative 8-bit amplitude; NOT calibrated sigma^0)")

    # ── Phase 8: Split tiles into train / val / test ──────────────────────────
    splits = split_tiles(sampled, config)
    print(f"  Train: {len(splits['train'])} tiles | "
          f"Val: {len(splits['val'])} tiles | "
          f"Test: {len(splits['test'])} tiles")
    patch_size = int(cfg_pip.get("patch_size", 512))
    print(f"  ↳ ×4 patches each ({patch_size}×{patch_size}) → "
          f"~{len(splits['train'])*4} train / "
          f"{len(splits['val'])*4} val / "
          f"{len(splits['test'])*4} test patches")

    # ── Clear old JSONL files ─────────────────────────────────────────────────
    for sp in ("train", "val", "test"):
        p = os.path.join(dirs["data"], f"{sp}.jsonl")
        if os.path.exists(p):
            os.remove(p)

    all_records = []
    global_idx  = 0

    for split_name, tile_list in splits.items():
        if not tile_list:
            continue

        print(f"\n══ Processing split: {split_name} "
              f"({len(tile_list)} tiles × 4 patches) ══")

        for tile in tile_list:
            tile_stem = os.path.splitext(tile)[0]
            sar_path  = os.path.join(sar_dir,   tile)
            lbl_path  = os.path.join(label_dir, tile)

            # Read the full 1024×1024 tile once
            sar_full  = read_sar(sar_path)    # float32 (1024, 1024)
            mask_full = read_mask(lbl_path)   # uint8   (1024, 1024)

            # Split both into 512×512 patches (4 patches per tile)
            sar_patches  = split_into_patches(sar_full,  patch_size)
            mask_patches = split_into_patches(mask_full, patch_size)

            for sar_info, mask_info in zip(sar_patches, mask_patches):
                suffix    = sar_info["suffix"]         # e.g. "p00"
                position  = sar_info["position"]       # e.g. "top_left"
                patch_id  = f"{tile_stem}_{suffix}"    # e.g. "TrainArea_001_p00"
                sar_patch  = sar_info["patch"]         # float32 (512, 512)
                mask_patch = mask_info["patch"]        # uint8   (512, 512)

                # ── Save patch TIFs ───────────────────────────────────────
                sar_tif_abs = os.path.join(dirs["patches_sar"], f"{patch_id}.tif")
                lbl_tif_abs = os.path.join(dirs["patches_lbl"], f"{patch_id}.tif")
                
                if not (os.path.exists(sar_tif_abs) and os.path.exists(lbl_tif_abs)):
                    save_patch_tif(sar_patch.astype(np.uint8), sar_tif_abs)
                    save_patch_tif(mask_patch,                 lbl_tif_abs)

                # Relative paths stored in JSONL (relative to base_dir)
                patches_subdir = config["output"].get("patches_dir", "patches")
                sar_rel   = f"{patches_subdir}/sar/{patch_id}.tif"
                label_rel = f"{patches_subdir}/labels/{patch_id}.tif"

                # ── Phase 2: Global facts (on 512×512 patch mask) ────────────
                global_facts = extract_global_facts(mask_patch, config, patch_id)

                # ── Phase 2b: SAR per-class facts ─────────────────────────────
                # mean_sar_response and bright_pixel_fraction per present class.
                # Both are relative SAR amplitude statistics — NOT calibrated.
                class_map_dict  = get_class_map(config)
                sar_class_facts = extract_sar_facts(
                    sar_patch, mask_patch, class_map_dict,
                    global_threshold=global_sar_threshold,
                    present_min_proportion=config["thresholds"]["present_min_proportion"]
                )

                # ── Phase 3: Regional facts (3×3 grid on 512×512 patch) ───────
                regional_facts = extract_regional_facts(mask_patch, config)

                # ── Phases 4+5+6+7: QA ────────────────────────────────────────
                # mask_patch is passed so gen_region_grounding can compute
                # class stats on random free-form bboxes directly.
                qa_list = generate_qa(global_facts, regional_facts,
                                      sar_class_facts, config,
                                      mask=mask_patch, sar=sar_patch)

                # ── Phase 9: Build JSONL records ──────────────────────────────
                records = build_record(
                    patch_id=patch_id,
                    sar_rel=sar_rel, label_rel=label_rel,
                    tile=tile, patch_suffix=suffix, patch_position=position,
                    split=split_name,
                    global_facts=global_facts, regional_facts=regional_facts,
                    sar_class_facts=sar_class_facts,
                    qa_list=qa_list, config=config, sample_idx=global_idx
                )
                global_idx += 1

                jsonl_path = os.path.join(dirs["data"], f"{split_name}.jsonl")
                for rec in records:
                    append_jsonl(rec, jsonl_path)
                all_records.extend(records)

                # ── Visualisation (one panel per patch) ────────────────────
                if do_vis:
                    vis_path = os.path.join(dirs["vis"], f"{patch_id}.png")
                    visualise_tile(sar_patch, mask_patch, qa_list,
                                   patch_id, config, vis_path)
                print(f"    ✓ {patch_id} [{position}]: "
                      f"{len(qa_list)} QA | patches saved"
                      + (f" + vis" if do_vis else ""))


    # ── Phase 7: Distribution reports ────────────────────────────────────────
    print("\n══ Phase 7: Saving distribution reports ══")
    save_distribution_reports(all_records, config, dirs)

    # ── Phase 10: Validate ───────────────────────────────────────────────────
    print("\n══ Phase 10: Validating dataset ══")
    validate(config)

    print(f"\n══ Pipeline complete ══")
    print(f"  Total QA records: {len(all_records)}")
    print(f"  JSONL files:      {dirs['data']}/")
    print(f"  Visualisations:   {dirs['vis']}/")
    print(f"  Reports:          {dirs['reports']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenEarthMap → MLLM dataset generation pipeline"
    )
    parser.add_argument("--config", required=True,
                        help="Path to config.yaml")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip QA visualisation panels (faster for full-dataset runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, do_vis=not args.no_vis)
