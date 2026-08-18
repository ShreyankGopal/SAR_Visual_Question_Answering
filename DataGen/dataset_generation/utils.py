"""
utils.py — Shared utilities for the dataset generation pipeline.
"""

import os
import json
import yaml
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_sar(path: str) -> np.ndarray:
    """Read a single-band SAR GeoTIFF and return float32 array (H, W)."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def read_mask(path: str) -> np.ndarray:
    """Read a single-band label GeoTIFF and return uint8 array (H, W)."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8)


def stretch_sar(sar: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Percentile contrast-stretch to [0, 1] for display."""
    lo, hi = np.percentile(sar, (p_low, p_high))
    return np.clip((sar - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def mask_to_rgb(mask: np.ndarray, class_colors: dict) -> np.ndarray:
    """Convert single-band label array to RGB using class_colors dict {id: [R,G,B]}."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in class_colors.items():
        rgb[mask == int(cls_id)] = color
    return rgb


def list_tif_files(directory: str) -> list:
    """Return sorted list of .tif filenames in a directory (ignores checkpoints)."""
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".tif") and not f.startswith(".")
    )


def ensure_dir(*paths):
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(record: dict, path: str):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Percentage binning
# ─────────────────────────────────────────────────────────────────────────────

def proportion_to_bin(proportion: float, bins: list) -> str:
    """
    Convert a proportion (0.0-1.0) to a human-readable percentage bin string.
    bins: list of [lo_pct, hi_pct, label] from config.
    """
    pct = proportion * 100.0
    for lo, hi, label in bins:
        if lo <= pct < hi:
            return label
    return "more than 90%"  # fallback for pct==100


# ─────────────────────────────────────────────────────────────────────────────
# Class helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_class_map(config: dict) -> dict:
    """Return {int_id: str_name} class map."""
    return {int(k): v for k, v in config["class_map"].items()}


def get_class_colors(config: dict) -> dict:
    """Return {int_id: [R,G,B]} colour map."""
    return {int(k): v for k, v in config["class_colors"].items()}


def get_grid_regions(config: dict) -> dict:
    """Return {region_name: [x1, y1, x2, y2]} with float coords."""
    return {k: [float(v) for v in vs] for k, vs in config["grid_regions"].items()}


# ─────────────────────────────────────────────────────────────────────────────
# Region extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_region(mask: np.ndarray, bbox_norm: list) -> np.ndarray:
    """
    Extract a sub-array from mask using normalized [x1, y1, x2, y2] bbox.
    x = column (width), y = row (height).
    """
    H, W = mask.shape
    x1, y1, x2, y2 = bbox_norm
    r0 = int(round(y1 * H))
    r1 = int(round(y2 * H))
    c0 = int(round(x1 * W))
    c1 = int(round(x2 * W))
    return mask[r0:r1, c0:c1]


# ─────────────────────────────────────────────────────────────────────────────
# Class statistics on a mask patch
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_stats(patch: np.ndarray, class_map: dict,
                        present_threshold: float) -> dict:
    """
    Compute per-class pixel_count, proportion, and presence for a mask patch.
    Returns:
        stats: {class_name: {pixel_count, proportion, present}}
        total_pixels: int
    """
    total = patch.size
    stats = {}
    for cls_id, cls_name in class_map.items():
        px = int(np.sum(patch == cls_id))
        prop = px / total if total > 0 else 0.0
        stats[cls_name] = {
            "pixel_count": px,
            "proportion": round(prop, 6),
            "present": prop >= present_threshold,
        }
    return stats, total


def rank_classes(stats: dict) -> list:
    """Return class names sorted by pixel_count descending."""
    return sorted(stats.keys(), key=lambda c: stats[c]["pixel_count"], reverse=True)


def dominant_class(stats: dict, margin: float) -> str | None:
    """
    Return the dominant class name if it leads the second class by at least `margin`.
    Returns None if no clear dominant.
    """
    ranked = rank_classes(stats)
    present = [c for c in ranked if stats[c]["present"]]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    top_prop  = stats[present[0]]["proportion"]
    sec_prop  = stats[present[1]]["proportion"]
    if (top_prop - sec_prop) >= margin:
        return present[0]
    return present[0]   # still return top even without clear margin


def classes_present(stats: dict) -> list:
    return [c for c, v in stats.items() if v["present"]]


# ─────────────────────────────────────────────────────────────────────────────
# Patch splitting
# ─────────────────────────────────────────────────────────────────────────────

# Canonical patch position names keyed by (row_idx, col_idx)
PATCH_POSITION_NAMES = {
    (0, 0): "top_left",
    (0, 1): "top_right",
    (1, 0): "bottom_left",
    (1, 1): "bottom_right",
}


def split_into_patches(arr: np.ndarray, patch_size: int = 512) -> list:
    """
    Split a 2-D array (H, W) into non-overlapping patch_size×patch_size tiles.

    For a 1024×1024 input with patch_size=512 this yields 4 patches:
        (row=0, col=0) top_left      → arr[0:512,   0:512  ]
        (row=0, col=1) top_right     → arr[0:512,   512:1024]
        (row=1, col=0) bottom_left   → arr[512:1024, 0:512  ]
        (row=1, col=1) bottom_right  → arr[512:1024, 512:1024]

    Returns list of dicts:
        {
          "patch":    np.ndarray (patch_size, patch_size),
          "row":      int,
          "col":      int,
          "position": str,   e.g. "top_left"
          "suffix":   str,   e.g. "p00"
        }

    Partial patches at the image boundary are silently dropped to guarantee
    every returned patch is exactly patch_size×patch_size.
    """
    H, W = arr.shape
    patches = []
    for r in range(H // patch_size):
        for c in range(W // patch_size):
            r0, r1 = r * patch_size, (r + 1) * patch_size
            c0, c1 = c * patch_size, (c + 1) * patch_size
            position = PATCH_POSITION_NAMES.get((r, c), f"r{r}c{c}")
            suffix   = f"p{r}{c}"
            patches.append({
                "patch":    arr[r0:r1, c0:c1],
                "row":      r,
                "col":      c,
                "position": position,
                "suffix":   suffix,
            })
    return patches


def save_patch_tif(arr: np.ndarray, path: str):
    """
    Save a 2-D numpy array as a single-band GeoTIFF using rasterio.
    Works for both SAR patches (float32/uint8, 0-255) and label patches (uint8, 1-8).
    No CRS or geotransform is written — these are cropped dataset patches.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
    ) as dst:
        dst.write(arr, 1)
