"""
extract_global_facts.py — Phase 2: Per-image global class statistics.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    read_mask, get_class_map,
    compute_class_stats, rank_classes,
    dominant_class, classes_present,
)


def extract_global_facts(mask: np.ndarray, config: dict, image_id: str) -> dict:
    """
    Compute global class statistics for a single mask.

    Returns a dict with:
        image_id, global (per-class stats), dominant_class,
        second_dominant_class, smallest_nonzero_class,
        classes_present, n_classes_present
    """
    class_map  = get_class_map(config)
    thresholds = config["thresholds"]
    present_th = thresholds["present_min_proportion"]
    dom_margin = thresholds["dominant_margin"]

    stats, total = compute_class_stats(mask, class_map, present_th)
    ranked  = rank_classes(stats)
    present = classes_present(stats)

    # Dominant
    dom = dominant_class(stats, dom_margin)

    # Second dominant
    present_ranked = [c for c in ranked if stats[c]["present"]]
    second = present_ranked[1] if len(present_ranked) >= 2 else None

    # Smallest nonzero (present) class
    smallest = min(present, key=lambda c: stats[c]["pixel_count"]) if present else None

    # Add rank to each class
    rank_lookup = {c: i + 1 for i, c in enumerate(ranked)}
    for cls_name in stats:
        stats[cls_name]["rank"] = rank_lookup[cls_name]

    return {
        "image_id": image_id,
        "total_pixels": total,
        "global": stats,
        "dominant_class": dom,
        "second_dominant_class": second,
        "smallest_nonzero_class": smallest,
        "classes_present": present,
        "n_classes_present": len(present),
    }
