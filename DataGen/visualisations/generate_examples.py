"""
DataGen/visualisations/generate_examples.py
---------------------------------------------
For every dataset (SIVED, HRSID, OGSOD, SAR-AIRcraft, OEM) and every question
category that dataset generates, picks one example record from train.jsonl,
draws its image (percentile-stretched for display) with its bbox/region/point
overlay (if any), captions it with the question + answer, and saves it to:

    DataGen/visualisations/<dataset>/<category>.png

Requires each dataset's build stage to have already been run (reads straight
from the pipelines' own train.jsonl output -- doesn't regenerate anything).

Run from anywhere:
    python DataGen/visualisations/generate_examples.py
"""
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATAGEN = os.path.dirname(HERE)
SAR_VQA_DIR = os.path.join(DATAGEN, "sar_vqa_data")
OEM_DIR = os.path.join(DATAGEN, "dataset_generation")

sys.path.insert(0, SAR_VQA_DIR)
sys.path.insert(0, OEM_DIR)

from core.io import load_config, out_path, read_jsonl  # noqa: E402
from utils import stretch_sar  # noqa: E402  (dataset_generation/utils.py)

SEED = 42


def stretch_plain(arr, p_low=2.0, p_high=98.0):
    """Percentile contrast-stretch an arbitrary (non-georeferenced) array to [0, 1]."""
    lo, hi = np.percentile(arr, (p_low, p_high))
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def draw_rect(ax, box, W, H, color):
    x1, y1, x2, y2 = box
    ax.add_patch(plt.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False, ec=color, lw=2))


def save_example(img, gt, q, a, grid_regions, title, out_file):
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.imshow(img, cmap="gray")
    if gt.get("bbox"):
        draw_rect(ax, gt["bbox"], W, H, "cyan")
    for rg in ([gt["region"]] if gt.get("region") else []) + (gt.get("regions") or []):
        if grid_regions and rg in grid_regions:
            draw_rect(ax, grid_regions[rg], W, H, "yellow")
    for i, p in enumerate(gt.get("points") or ([gt["point"]] if gt.get("point") else [])):
        ax.plot(p[0] * W, p[1] * H, "o", ms=10, mfc="none", mec=["red", "lime"][i % 2], mew=2)
    ax.axis("off")
    ax.set_title(f"{title}\nQ: {q}\nA: {a}", fontsize=9, wrap=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=110, bbox_inches="tight")
    plt.close(fig)


def one_per_category(records, rng):
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    return {cat: rng.choice(recs) for cat, recs in by_cat.items()}


def do_sar_vqa_dataset(name, cfg_path, out_root):
    cfg = load_config(cfg_path)
    jsonl_path = out_path(cfg, "data_dir", "train.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[skip] {name}: no train.jsonl at {jsonl_path} -- run the build stage first")
        return
    picked = one_per_category(read_jsonl(jsonl_path), random.Random(SEED))
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    grid_regions = cfg.get("grid_regions")
    for cat, r in picked.items():
        img_path = os.path.join(cfg["data"]["root"], r["image"])
        if not os.path.exists(img_path):
            print(f"[skip] {name}/{cat}: image not found: {img_path}")
            continue
        arr = stretch_plain(np.array(Image.open(img_path).convert("L")))
        q, a = (c["value"] for c in r["conversations"])
        save_example(arr, r["ground_truth_facts"], q, a, grid_regions,
                     f"{name} | {cat}", os.path.join(out_dir, f"{cat}.png"))
    print(f"[done] {name}: {len(picked)} category examples -> {out_dir}")


def do_oem(out_root):
    cfg = load_config(os.path.join(OEM_DIR, "config.yaml"))
    jsonl_path = os.path.join(cfg["output"]["base_dir"], cfg["output"]["data_dir"], "train.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[skip] oem: no train.jsonl at {jsonl_path} -- run build_dataset.py first")
        return
    picked = one_per_category(read_jsonl(jsonl_path), random.Random(SEED))
    out_dir = os.path.join(out_root, "oem")
    os.makedirs(out_dir, exist_ok=True)
    grid_regions = cfg.get("grid_regions")
    for cat, r in picked.items():
        img_path = os.path.join(cfg["output"]["base_dir"], r["image"])
        if not os.path.exists(img_path):
            print(f"[skip] oem/{cat}: image not found: {img_path}")
            continue
        import rasterio
        with rasterio.open(img_path) as src:
            arr = stretch_sar(src.read(1).astype(np.float32))
        q, a = (c["value"] for c in r["conversations"])
        save_example(arr, r["ground_truth_facts"], q, a, grid_regions,
                     f"oem | {cat}", os.path.join(out_dir, f"{cat}.png"))
    print(f"[done] oem: {len(picked)} category examples -> {out_dir}")


def main():
    for name, cfg_file in [
        ("sived", "sived.yaml"),
        ("hrsid", "hrsid.yaml"),
        ("ogsod", "ogsod.yaml"),
        ("sar_aircraft", "sar_aircraft.yaml"),
    ]:
        do_sar_vqa_dataset(name, os.path.join(SAR_VQA_DIR, "configs", cfg_file), HERE)
    do_oem(HERE)


if __name__ == "__main__":
    main()
