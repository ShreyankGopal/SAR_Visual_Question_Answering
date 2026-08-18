"""
extract_regional_facts.py — Phase 3: 3×3 spatial grid statistics.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    get_class_map, get_grid_regions,
    extract_region, compute_class_stats,
    rank_classes, dominant_class, classes_present,
)


def extract_regional_facts(mask: np.ndarray, config: dict) -> dict:
    """
    Divide the mask into a 3×3 grid and compute per-region class statistics.

    Returns:
        {region_name: {bbox, dominant_class, second_dominant_class,
                       class_proportions, classes_present, n_classes_present}}
    """
    class_map    = get_class_map(config)
    grid_regions = get_grid_regions(config)
    thresholds   = config["thresholds"]
    present_th   = thresholds["present_min_proportion"]
    dom_margin   = thresholds["dominant_margin"]

    regional = {}

    for region_name, bbox in grid_regions.items():
        patch = extract_region(mask, bbox)

        if patch.size == 0:
            regional[region_name] = {"bbox": bbox, "error": "empty patch"}
            continue

        stats, total = compute_class_stats(patch, class_map, present_th)
        ranked  = rank_classes(stats)
        present = classes_present(stats)
        dom     = dominant_class(stats, dom_margin)

        present_ranked = [c for c in ranked if stats[c]["present"]]
        second = present_ranked[1] if len(present_ranked) >= 2 else None

        # Compact proportions dict (present classes only)
        proportions = {
            c: round(stats[c]["proportion"], 6)
            for c in present
        }

        regional[region_name] = {
            "bbox": bbox,
            "dominant_class": dom,
            "second_dominant_class": second,
            "class_proportions": proportions,
            "classes_present": present,
            "n_classes_present": len(present),
        }

    return regional


def find_region_with_most(regional: dict, class_name: str) -> str | None:
    """Return region_name where class_name has the highest proportion."""
    best_region = None
    best_prop   = -1.0
    for region_name, info in regional.items():
        prop = info.get("class_proportions", {}).get(class_name, 0.0)
        if prop > best_prop:
            best_prop   = prop
            best_region = region_name
    return best_region if best_prop > 0 else None
