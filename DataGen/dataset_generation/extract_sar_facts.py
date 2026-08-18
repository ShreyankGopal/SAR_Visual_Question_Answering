"""
extract_sar_facts.py — SAR-derived features per LULC class (Phase 2b).

Computes two RELATIVE SAR amplitude statistics per class:

  mean_sar_response(c)
      = arithmetic mean of 8-bit SAR pixel values for all pixels of class c.
      IMPORTANT: This is a relative SAR amplitude/response statistic derived
      from the raw 8-bit OpenEarthMap-SAR TIFF. It is NOT a calibrated sigma^0
      or any other physical backscatter coefficient.

  bright_pixel_fraction(c)
      = fraction of class-c pixels whose SAR value is >= global threshold T.
      T is computed from the training dataset's pixel distribution
      (e.g., the 90th percentile across all sampled SAR images).
      IMPORTANT: "bright" here means "above the dataset-wide high-amplitude
      percentile threshold" — it is NOT a measure of dielectric constant,
      surface roughness, or moisture.

Only classes actually present in the segmentation mask are included.
Classes with zero pixels are skipped safely.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_sar, get_class_map


def compute_global_threshold(sar_paths: list, percentile: float = 90.0,
                              max_pixels: int = 5_000_000) -> float:
    """
    Compute a global brightness threshold T from a list of SAR image paths.

    Samples up to max_pixels from the provided images uniformly at random,
    then returns the given percentile of those values.

    T is used consistently across the entire dataset to define "bright" pixels.
    It is NOT image-specific or class-specific.

    Args:
        sar_paths:   list of absolute paths to SAR TIFFs
        percentile:  e.g. 90.0 for the 90th percentile
        max_pixels:  cap on total sampled pixels to keep memory reasonable

    Returns:
        threshold T (float, in 8-bit range 0–255)
    """
    all_vals = []
    budget   = max_pixels

    np.random.seed(0)  # reproducible sampling
    for path in sar_paths:
        if budget <= 0:
            break
        sar = read_sar(path)          # float32, 0–255 (raw 8-bit)
        flat = sar.ravel()
        n = min(len(flat), budget)
        idx = np.random.choice(len(flat), n, replace=False)
        all_vals.append(flat[idx])
        budget -= n

    combined = np.concatenate(all_vals)
    T = float(np.percentile(combined, percentile))
    return T


def extract_sar_facts(sar: np.ndarray, mask: np.ndarray,
                      class_map: dict, global_threshold: float,
                      present_min_proportion: float = 0.005) -> dict:
    """
    Compute mean_sar_response and bright_pixel_fraction for every
    LULC class present in the segmentation mask.

    Args:
        sar:                    float32 array (H, W), raw 8-bit SAR values
        mask:                   uint8 array (H, W), class labels (1-indexed)
        class_map:              {int_id: str_name}
        global_threshold:       T — dataset-wide "bright" threshold
        present_min_proportion: minimum proportion to consider a class present

    Returns:
        sar_class_facts: {class_name: {mean_sar_response, bright_pixel_fraction}}
        Only includes classes with enough pixels (>= present_min_proportion).
    """
    total_pixels = mask.size
    sar_class_facts = {}

    for cls_id, cls_name in class_map.items():
        # Select pixels belonging to this class
        class_mask = (mask == cls_id)
        n_pixels   = int(np.sum(class_mask))

        # Skip absent or near-absent classes
        if n_pixels == 0:
            continue
        if n_pixels / total_pixels < present_min_proportion:
            continue

        class_sar_pixels = sar[class_mask]  # 1-D array of SAR values for this class

        # ── Feature A: mean_sar_response ──────────────────────────────────
        # Arithmetic mean of 8-bit SAR amplitude for class pixels.
        # Relative statistic only — NOT calibrated sigma^0.
        mean_val = float(np.mean(class_sar_pixels))

        # ── Feature B: bright_pixel_fraction ──────────────────────────────
        # Fraction of class pixels above the GLOBAL threshold T.
        # "Bright" = unusually high SAR amplitude relative to the dataset.
        # NOT a measure of surface roughness, moisture, or dielectric constant.
        bright_frac = float(np.sum(class_sar_pixels >= global_threshold) / n_pixels)

        sar_class_facts[cls_name] = {
            "mean_sar_response":    round(mean_val, 2),
            "bright_pixel_fraction": round(bright_frac, 4),
            "n_pixels":             n_pixels,  # kept for auditability
        }

    return sar_class_facts


def compute_sar_bbox_variance(sar: np.ndarray, mask: np.ndarray, bbox_norm: list, 
                              class_map: dict, present_th: float) -> dict:
    """
    For a given bounding box, find the dominant class, compute overall SAR variance,
    and per-class SAR variance.
    
    Args:
        sar: float32 array (H, W), raw 8-bit SAR values
        mask: uint8 array (H, W), class labels (1-indexed)
        bbox_norm: [x1, y1, x2, y2] normalized coordinates
        class_map: {int_id: str_name}
        present_th: minimum proportion to consider a class present
        
    Returns:
        dict with dominant_class, overall_variance, and per_class_variance
    """
    from utils import extract_region, compute_class_stats, dominant_class
    
    # Extract regions from both SAR and mask
    sar_patch = extract_region(sar, bbox_norm)
    mask_patch = extract_region(mask, bbox_norm)
    
    if sar_patch.size == 0 or mask_patch.size == 0:
        return {}
        
    # Determine dominant class
    stats, _ = compute_class_stats(mask_patch, class_map, present_th)
    dom = dominant_class(stats, margin=0.0) # We just want the most frequent class
    
    if not dom:
        return {}
        
    # Overall variance
    overall_var = float(np.var(sar_patch))
    
    # Per-class variance
    per_class_var = {}
    for cls_id, cls_name in class_map.items():
        class_mask = (mask_patch == cls_id)
        if np.any(class_mask):
            class_sar = sar_patch[class_mask]
            # Need at least 2 pixels for meaningful variance, otherwise 0
            if len(class_sar) > 1:
                per_class_var[cls_name] = float(np.var(class_sar))
            else:
                per_class_var[cls_name] = 0.0
                
    return {
        "dominant_class": dom,
        "overall_variance": round(overall_var, 2),
        "per_class_variance": {k: round(v, 2) for k, v in per_class_var.items()}
    }
