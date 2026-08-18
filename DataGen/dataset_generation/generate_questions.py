"""
generate_questions.py — Phases 4 & 5: Deterministic QA generation from facts.

All questions are grounded strictly in facts derived from the segmentation mask.
No object-level information, no bounding boxes, no instance counts.
"""

import os
import sys
import random
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    proportion_to_bin, get_grid_regions, get_class_map,
    extract_region, compute_class_stats,
    rank_classes, dominant_class, classes_present,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_region(name: str) -> str:
    return name.replace("_", " ")


def _fmt_bbox(bbox: list) -> str:
    return "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*bbox)


def _bin(prop: float, bins: list) -> str:
    return proportion_to_bin(prop, bins)


# ─────────────────────────────────────────────────────────────────────────────
# Category 1 — Global Classification
# ─────────────────────────────────────────────────────────────────────────────

def gen_global_classification(global_facts: dict, config: dict) -> list:
    qa = []
    gf = global_facts

    dom   = gf["dominant_class"]
    pres  = gf["classes_present"]

    if not pres:
        return qa

    # Q1: dominant class
    if dom:
        qa.append({
            "category": "global_classification",
            "question": "<image>\nWhat is the dominant land-cover type in this image?",
            "answer": f"{dom}.",
        })

    # Q2: largest area
    if dom:
        qa.append({
            "category": "global_classification",
            "question": "<image>\nWhich land-cover class occupies the largest area in this image?",
            "answer": f"{dom}.",
        })

    # Q3: classes present
    pres_str = ", ".join(pres[:-1]) + f", and {pres[-1]}" if len(pres) > 1 else pres[0]
    qa.append({
        "category": "global_classification",
        "question": "<image>\nWhich land-cover classes are visible in this image?",
        "answer": f"{pres_str}.",
    })

    # Q4: number of classes
    qa.append({
        "category": "global_classification",
        "question": "<image>\nHow many distinct land-cover classes are present in this image?",
        "answer": f"{gf['n_classes_present']}.",
    })

    # Q5: yes/no presence questions (pick up to 2 random classes)
    all_classes = list(gf["global"].keys())
    random.shuffle(all_classes)
    for cls in all_classes[:2]:
        is_pres = gf["global"][cls]["present"]
        qa.append({
            "category": "global_classification",
            "question": f"<image>\nIs {cls.lower()} present in this image?",
            "answer": "Yes." if is_pres else "No.",
        })

    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Category 2 — Global Quantitative
# ─────────────────────────────────────────────────────────────────────────────

def gen_global_quantitative(global_facts: dict, config: dict) -> list:
    qa     = []
    gf     = global_facts
    bins   = config["percentage_bins"]
    pres   = gf["classes_present"]
    cmp_th = config["thresholds"]["comparison_min_diff"]

    if not pres:
        return qa

    # Q1: percentage of dominant class
    dom = gf["dominant_class"]
    if dom:
        prop = gf["global"][dom]["proportion"]
        qa.append({
            "category": "global_quantitative",
            "question": f"<image>\nApproximately what percentage of this image is covered by {dom.lower()}?",
            "answer": f"{_bin(prop, bins)}.",
        })

    # Q2: percentage of a randomly chosen present class (not dominant)
    non_dom = [c for c in pres if c != dom]
    if non_dom:
        cls = random.choice(non_dom)
        prop = gf["global"][cls]["proportion"]
        qa.append({
            "category": "global_quantitative",
            "question": f"<image>\nApproximately what percentage of this image is covered by {cls.lower()}?",
            "answer": f"{_bin(prop, bins)}.",
        })

    # Q3: comparison between two present classes
    if len(pres) >= 2:
        pairs = list(combinations(pres, 2))
        random.shuffle(pairs)
        for cls_a, cls_b in pairs[:2]:
            prop_a = gf["global"][cls_a]["proportion"]
            prop_b = gf["global"][cls_b]["proportion"]
            if abs(prop_a - prop_b) >= cmp_th:
                winner = cls_a if prop_a > prop_b else cls_b
                qa.append({
                    "category": "global_quantitative",
                    "question": (
                        f"<image>\nWhich occupies more area in this image, "
                        f"{cls_a.lower()} or {cls_b.lower()}?"
                    ),
                    "answer": f"{winner}.",
                })

    # Q4: top-2 classes
    ranked_pres = sorted(pres, key=lambda c: gf["global"][c]["proportion"], reverse=True)
    if len(ranked_pres) >= 2:
        qa.append({
            "category": "global_quantitative",
            "question": "<image>\nWhat are the two most prevalent land-cover classes in this image?",
            "answer": f"{ranked_pres[0]} and {ranked_pres[1]}.",
        })

    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Category 3 — Regional VQA (delta-aware cell merging)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical merge rules: frozenset of cell names → human-readable merged label.
# Checked in the order listed; first match wins.
_MERGE_RULES = [
    (frozenset(["top_left", "top_center", "top_right"]),                          "top region"),
    (frozenset(["bottom_left", "bottom_center", "bottom_right"]),                 "bottom region"),
    (frozenset(["top_left", "middle_left", "bottom_left"]),                       "left region"),
    (frozenset(["top_right", "middle_right", "bottom_right"]),                    "right region"),
    # Centered patterns — checked before the simple row/col rules
    (frozenset(["top_center", "center", "bottom_center",
                "middle_left", "middle_right"]),                                  "centered in a cross shape"),
    (frozenset(["top_center", "center", "bottom_center"]),                        "vertically centered region"),
    (frozenset(["middle_left", "center", "middle_right"]),                        "horizontally centered region"),
]


def _merge_cells(shortlisted: set) -> str:
    """
    Given a set of shortlisted cell names, try to merge them into a
    human-readable label using _MERGE_RULES.
    Falls back to a comma-separated list of human-readable cell names.
    """
    for required_cells, label in _MERGE_RULES:
        if required_cells.issubset(shortlisted):
            return label
    # No merge matched — format individual cell names nicely
    names = sorted(_fmt_region(c) for c in shortlisted)
    if len(names) == 1:
        return f"the {names[0]} region"
    return "the " + ", ".join(names[:-1]) + f", and {names[-1]} regions"


def gen_regional_vqa(regional_facts: dict, global_facts: dict,
                     config: dict, mask: np.ndarray = None) -> list:
    """
    Generate Regional VQA questions using delta-aware cell selection and
    semantic merging of shortlisted cells into human-readable region labels.
    """
    qa         = []
    bins       = config["percentage_bins"]
    pres_th    = config["thresholds"]["present_min_proportion"]

    # ── Part A: dominant-class question per grid cell ─────────────────────────
    for region_name, rinfo in regional_facts.items():
        rname_fmt = _fmt_region(region_name)
        dom       = rinfo.get("dominant_class")
        pres      = rinfo.get("classes_present", [])

        if not pres or not dom:
            continue

        # Q: dominant in region
        qa.append({
            "category": "regional_vqa",
            "question": f"<image>\nWhat is the dominant land-cover type in the {rname_fmt} region?",
            "answer": f"{dom}.",
            "region": region_name,
        })

        # Q: classes present in region
        if pres:
            pres_str = ", ".join(pres[:-1]) + f", and {pres[-1]}" if len(pres) > 1 else pres[0]
            qa.append({
                "category": "regional_vqa",
                "question": f"<image>\nWhich land-cover classes are present in the {rname_fmt} region?",
                "answer": f"{pres_str}.",
                "region": region_name,
            })

    # ── Part B: "which region has most of class X?" with delta+merge ──────────
    all_classes = global_facts.get("classes_present", [])
    random.shuffle(all_classes)

    # delta_fraction is in proportion space (0.0–1.0).
    # A cell is shortlisted if: best_prop - cell_prop < delta_fraction
    # e.g. delta=0.10 means include any cell within 10 percentage points of best.
    delta_fraction = config.get("regional_vqa", {}).get("delta_fraction", 0.1)

    region_names = list(regional_facts.keys())

    for cls in all_classes[:3]:
        # Collect (cell_name, pixel_count_of_cls) for all regions
        cell_counts = []
        for rn, ri in regional_facts.items():
            prop = ri.get("class_proportions", {}).get(cls, 0.0)
            # We need pixel counts. Reconstruct: proportion × total_region_pixels.
            # Since proportions are stored (not raw counts), and each region is
            # 1/9 of 512×512 = ~29127 pixels for a 3×3 grid on 512px patch.
            # Compute actual region pixel count from bbox.
            cell_counts.append((rn, prop))

        if not cell_counts:
            continue

        # Sort descending by proportion
        cell_counts.sort(key=lambda x: x[1], reverse=True)
        best_name, best_prop = cell_counts[0]

        if best_prop < pres_th:
            continue

        # Delta: cells with proportion >= best_prop - delta are included
        delta_prop = delta_fraction  # delta in proportion space (same units)
        shortlisted = set()
        for rn, prop in cell_counts:
            if best_prop - prop < delta_prop:
                shortlisted.add(rn)

        merged_label = _merge_cells(shortlisted)

        qa.append({
            "category": "regional_vqa",
            "question": f"<image>\nWhich region of the image contains the highest proportion of {cls.lower()}?",
            "answer": f"The {merged_label}.",
            "region": best_name,   # primary region for ground-truth reference
            "shortlisted_regions": list(shortlisted),
        })

    return qa


def _region_pixel_count(regional_facts: dict) -> int:
    """Estimate per-region pixel count from any available region (unused; kept for clarity)."""
    return 512 * 512 // 9  # approximate for a 3×3 grid on 512px patch


# ─────────────────────────────────────────────────────────────────────────────
# Category 4 — Region Grounding (random free-form bounding boxes)
# ─────────────────────────────────────────────────────────────────────────────

def gen_region_grounding(mask: np.ndarray, config: dict) -> list:
    """
    Sample random free-form bounding boxes within [0,1] and compute
    land-cover statistics directly from the mask for each box.

    Config keys used (under region_grounding):
      min_size   — minimum normalized height/width (default 0.08)
      max_size   — maximum normalized height/width (default 0.30)
      num_regions — number of random regions to try (default 3)
    """
    qa = []
    rg_cfg     = config.get("region_grounding", {}) # region grounding config variable to get the min max size 
    min_size   = float(rg_cfg.get("min_size",    0.08))
    max_size   = float(rg_cfg.get("max_size",    0.30))
    num_regions = int(rg_cfg.get("num_regions",  3))
    pres_th    = config["thresholds"]["present_min_proportion"]
    dom_margin = config["thresholds"]["dominant_margin"]
    class_map  = get_class_map(config)

    generated = 0
    attempts  = 0
    max_attempts = num_regions * 10  # avoid infinite loop

    while generated < num_regions and attempts < max_attempts:
        attempts += 1

        # Sample random center
        cx = random.uniform(0.0, 1.0)
        cy = random.uniform(0.0, 1.0)

        # Sample random width and height independently
        w = random.uniform(min_size, max_size)
        h = random.uniform(min_size, max_size)

        # Compute corners and clamp to [0, 1]
        x1 = max(0.0, cx - w / 2)
        x2 = min(1.0, cx + w / 2)
        y1 = max(0.0, cy - h / 2)
        y2 = min(1.0, cy + h / 2)

        # Ensure the clamped box is still large enough
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue

        bbox = [x1, y1, x2, y2]
        patch = extract_region(mask, bbox)

        if patch.size == 0:
            continue

        stats, _ = compute_class_stats(patch, class_map, pres_th) # extract class statistics(number of pixels and so on)
        pres  = classes_present(stats)# find what classes are actually present
        dom   = dominant_class(stats, dom_margin) # calculate teh dominant classes based on pixel sizes

        if not pres or not dom:
            continue

        bbox_str  = _fmt_bbox(bbox)
        pres_str  = ", ".join(pres)

        # Q1: dominant class in this random bbox
        qa.append({
            "category": "region_grounding",
            "question": (
                f"<image>\nWhat is the dominant land-cover type in the region "
                f"{bbox_str} (normalized [x1, y1, x2, y2])?"
            ),
            "answer": f"{dom}.",
            "bbox": bbox,
        })

        # Q2: describe land cover in this random bbox
        qa.append({
            "category": "region_grounding",
            "question": (
                f"<image>\nDescribe the land cover in the region "
                f"{bbox_str} (normalized [x1, y1, x2, y2])."
            ),
            "answer": (
                f"The region is predominantly {dom.lower()}. "
                f"Other classes present include: {pres_str}."
            ),
            "bbox": bbox,
        })

        generated += 1

    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Category 5 — Comparative Spatial Reasoning
# ─────────────────────────────────────────────────────────────────────────────

def gen_comparative_spatial(regional_facts: dict, global_facts: dict, config: dict) -> list:
    qa     = []
    cmp_th = config["thresholds"]["comparison_min_diff"]
    pres   = global_facts.get("classes_present", [])

    region_names = list(regional_facts.keys())

    # Q1: compare two regions for a class
    cls_sample = random.sample(pres, min(3, len(pres)))
    for cls in cls_sample:
        pairs = list(combinations(region_names, 2))
        random.shuffle(pairs)
        for rn_a, rn_b in pairs[:2]:
            prop_a = regional_facts[rn_a].get("class_proportions", {}).get(cls, 0.0)
            prop_b = regional_facts[rn_b].get("class_proportions", {}).get(cls, 0.0)
            if abs(prop_a - prop_b) >= cmp_th:
                more_region = rn_a if prop_a > prop_b else rn_b
                qa.append({
                    "category": "comparative_spatial",
                    "question": (
                        f"<image>\nIs there more {cls.lower()} in the "
                        f"{_fmt_region(rn_a)} or {_fmt_region(rn_b)} region?"
                    ),
                    "answer": f"The {_fmt_region(more_region)} region.",
                })
                break  # one Q per class is enough

    # Q2: is class more prevalent in region A than B?
    if len(pres) >= 1 and len(region_names) >= 2:
        cls = random.choice(pres)
        rn_a, rn_b = random.sample(region_names, 2)
        prop_a = regional_facts[rn_a].get("class_proportions", {}).get(cls, 0.0)
        prop_b = regional_facts[rn_b].get("class_proportions", {}).get(cls, 0.0)
        if abs(prop_a - prop_b) >= cmp_th:
            answer = "Yes." if prop_a > prop_b else "No."
            qa.append({
                "category": "comparative_spatial",
                "question": (
                    f"<image>\nIs {cls.lower()} more prevalent in the "
                    f"{_fmt_region(rn_a)} region than in the {_fmt_region(rn_b)} region?"
                ),
                "answer": answer,
            })

    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Category 6 — SAR Observation
# Questions derived strictly from mean_sar_response and bright_pixel_fraction.
#
# IMPORTANT: All answers are OBSERVATIONAL — they compare relative SAR amplitude
# statistics across classes. They do NOT make claims about physical properties
# such as roughness, moisture, dielectric constant, or double-bounce.
# ─────────────────────────────────────────────────────────────────────────────

def gen_sar_observation(sar_class_facts: dict, config: dict) -> list:
    """
    Generate SAR observation questions about which class has the highest/lowest
    mean SAR response or bright-pixel fraction.
    Answers are derived purely from computed SAR statistics.
    """
    qa  = []
    cfg = config.get("sar", {})

    if len(sar_class_facts) < 2:
        return qa

    classes = list(sar_class_facts.keys())

    # Sort by mean_sar_response
    ranked_by_mean = sorted(classes,
                            key=lambda c: sar_class_facts[c]["mean_sar_response"],
                            reverse=True)
    top_mean  = ranked_by_mean[0]
    low_mean  = ranked_by_mean[-1]

    # Sort by bright_pixel_fraction
    ranked_by_bright = sorted(classes,
                              key=lambda c: sar_class_facts[c]["bright_pixel_fraction"],
                              reverse=True)
    top_bright  = ranked_by_bright[0]
    low_bright  = ranked_by_bright[-1]

    # Q1: class with strongest average SAR response
    qa.append({
        "category": "sar_observation",
        "question": "<image>\nWhich land-cover class has the strongest average SAR response in this image?",
        "answer": f"{top_mean}.",
    })

    # Q2: class with weakest average SAR response
    qa.append({
        "category": "sar_observation",
        "question": "<image>\nWhich land-cover class has the weakest average SAR response in this image?",
        "answer": f"{low_mean}.",
    })

    # Q3: class with highest bright-pixel fraction
    qa.append({
        "category": "sar_observation",
        "question": "<image>\nWhich land-cover class contains the highest fraction of very bright SAR pixels?",
        "answer": f"{top_bright}.",
    })

    # Q4: class with lowest bright-pixel fraction
    qa.append({
        "category": "sar_observation",
        "question": "<image>\nWhich land-cover class contains the lowest fraction of bright SAR pixels?",
        "answer": f"{low_bright}.",
    })

    return qa


def gen_sar_class_specific(sar_class_facts: dict, config: dict) -> list:
    """
    Generate class-specific SAR observation questions giving approximate
    numerical values of mean_sar_response and bright_pixel_fraction.
    Answers report relative SAR amplitude statistics only.
    """
    qa  = []
    cfg = config.get("sar", {})
    r_digits  = int(cfg.get("mean_response_round_digits", 0))
    pf_digits = int(cfg.get("bright_fraction_round_digits", 0))

    classes = list(sar_class_facts.keys())
    random.shuffle(classes)

    for cls in classes[:3]:   # limit to 3 class-specific questions
        mean_val   = sar_class_facts[cls]["mean_sar_response"]
        bright_frac = sar_class_facts[cls]["bright_pixel_fraction"]

        mean_str  = str(round(mean_val,  r_digits)).rstrip("0").rstrip(".")
        pct_str   = str(round(bright_frac * 100, pf_digits)).rstrip("0").rstrip(".")

        # Q: mean response for a specific class
        qa.append({
            "category": "sar_observation",
            "question": f"<image>\nWhat is the average SAR response of the {cls} class in this image?",
            "answer": f"The average SAR response of {cls} is approximately {mean_str}.",
        })

        # Q: bright-pixel fraction for a specific class
        qa.append({
            "category": "sar_observation",
            "question": (
                f"<image>\nWhat fraction of pixels belonging to {cls} have "
                f"a very bright SAR response in this image?"
            ),
            "answer": f"Approximately {pct_str}% of {cls} pixels have a bright SAR response.",
        })

    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Category 7 — SAR Comparative
# Pairwise comparisons of SAR statistics between two present classes.
# Strictly observational — no physical interpretation.
# ─────────────────────────────────────────────────────────────────────────────

def gen_sar_comparative(sar_class_facts: dict, config: dict) -> list:
    """
    Generate pairwise comparative questions about mean_sar_response and
    bright_pixel_fraction between two randomly chosen present classes.
    Only generates a question when the difference exceeds the configured
    minimum margin (to avoid trivially ambiguous comparisons).
    """
    qa  = []
    cfg = config.get("sar", {})
    mean_min_diff   = float(cfg.get("sar_mean_comparison_min_diff",   5.0))
    bright_min_diff = float(cfg.get("sar_bright_fraction_min_diff",   0.05))

    classes = list(sar_class_facts.keys())
    if len(classes) < 2:
        return qa

    pairs = list(combinations(classes, 2))
    random.shuffle(pairs)

    mean_q_added   = 0
    bright_q_added = 0

    for cls_a, cls_b in pairs:
        mean_a   = sar_class_facts[cls_a]["mean_sar_response"]
        mean_b   = sar_class_facts[cls_b]["mean_sar_response"]
        bright_a = sar_class_facts[cls_a]["bright_pixel_fraction"]
        bright_b = sar_class_facts[cls_b]["bright_pixel_fraction"]

        # Mean SAR response comparison
        if mean_q_added < 2 and abs(mean_a - mean_b) >= mean_min_diff:
            stronger = cls_a if mean_a > mean_b else cls_b
            qa.append({
                "category": "sar_comparative",
                "question": (
                    f"<image>\nDoes {cls_a} or {cls_b} have the stronger "
                    f"average SAR response in this image?"
                ),
                "answer": f"{stronger}.",
            })
            mean_q_added += 1

        # Bright-pixel fraction comparison
        if bright_q_added < 2 and abs(bright_a - bright_b) >= bright_min_diff:
            more_bright = cls_a if bright_a > bright_b else cls_b
            qa.append({
                "category": "sar_comparative",
                "question": (
                    f"<image>\nWhich has a larger fraction of bright SAR pixels, "
                    f"{cls_a} or {cls_b}?"
                ),
                "answer": f"{more_bright}.",
            })
            bright_q_added += 1

        if mean_q_added >= 2 and bright_q_added >= 2:
            break

    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Category 8 — SAR BBox Variance
# ─────────────────────────────────────────────────────────────────────────────

def gen_sar_bbox_variance(sar: np.ndarray, mask: np.ndarray, config: dict) -> list:
    qa = []
    if sar is None or mask is None:
        return qa
        
    cfg = config.get("sar_bbox_variance", {})
    min_size = float(cfg.get("min_size", 0.10))
    max_size = float(cfg.get("max_size", 0.35))
    num_regions = int(cfg.get("num_regions", 2))
    pres_th = config["thresholds"]["present_min_proportion"]
    class_map = get_class_map(config)
    
    from extract_sar_facts import compute_sar_bbox_variance
    
    generated = 0
    attempts = 0
    max_attempts = num_regions * 10
    
    while generated < num_regions and attempts < max_attempts:
        attempts += 1
        
        # Sample random center
        cx = random.uniform(0.0, 1.0)
        cy = random.uniform(0.0, 1.0)

        # Sample random width and height independently
        w = random.uniform(min_size, max_size)
        h = random.uniform(min_size, max_size)

        # Compute corners and clamp to [0, 1]
        x1 = max(0.0, cx - w / 2)
        x2 = min(1.0, cx + w / 2)
        y1 = max(0.0, cy - h / 2)
        y2 = min(1.0, cy + h / 2)

        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
            
        bbox = [x1, y1, x2, y2]
        
        res = compute_sar_bbox_variance(sar, mask, bbox, class_map, pres_th)
        if not res:
            continue
            
        dom = res["dominant_class"]
        var = res["overall_variance"]
        
        # Qualitative label
        if var < 50:
            het_label = "low heterogeneity"
        elif var < 150:
            het_label = "moderate heterogeneity"
        else:
            het_label = "high heterogeneity"
            
        bbox_str = _fmt_bbox(bbox)
        
        qa.append({
            "category": "sar_bbox_variance",
            "question": (
                f"<image>\nGiven the region {bbox_str} (normalized [x1, y1, x2, y2]), "
                f"what is the dominant land-cover class and how heterogeneous is the SAR response?"
            ),
            "answer": (
                f"The dominant class is {dom}. The SAR response variance in this region is {var}, "
                f"indicating {het_label}."
            ),
            "bbox": bbox,
        })
        generated += 1
        
    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Category 9 — Surrounding Classes
# ─────────────────────────────────────────────────────────────────────────────

def gen_surrounding_classes(regional_facts: dict, mask: np.ndarray, config: dict) -> list:
    qa = []
    if mask is None:
        return qa
        
    cfg = config.get("surrounding_classes", {})
    max_q = int(cfg.get("max_questions", 2))
    class_map = get_class_map(config)
    grid_regions = get_grid_regions(config)
    
    from extract_adjacency_facts import compute_adjacency_for_region
    
    region_names = list(regional_facts.keys())
    random.shuffle(region_names)
    
    generated = 0
    
    for rn in region_names:
        if generated >= max_q:
            break
            
        dom_name = regional_facts[rn].get("dominant_class")
        if not dom_name:
            continue
            
        # Find the ID for the dominant class
        dom_id = None
        for cid, cname in class_map.items():
            if cname == dom_name:
                dom_id = cid
                break
                
        if dom_id is None:
            continue
            
        bbox = grid_regions[rn]
        patch = extract_region(mask, bbox)
        
        res = compute_adjacency_for_region(patch, class_map, dom_id)
        touching = res["touching_classes"]
        props = res["class_proportions"]
        
        if not touching:
            continue
            
        # Format the touching classes string
        parts = []
        for c in touching:
            pct = props[c] * 100
            parts.append(f"{c} ({pct:.1f}% of boundary)")
            
        if len(parts) == 1:
            touch_str = parts[0]
        else:
            touch_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"
            
        rname_fmt = _fmt_region(rn)
        
        qa.append({
            "category": "surrounding_classes",
            "question": f"<image>\nWhat land-cover classes surround the dominant class in the {rname_fmt} region?",
            "answer": (
                f"The dominant class in the {rname_fmt} region is {dom_name}. "
                f"The classes directly adjacent to it are: {touch_str}."
            ),
            "region": rn,
        })
        generated += 1
        
    return qa


# ─────────────────────────────────────────────────────────────────────────────
# Main entry: generate all QA for one image
# ─────────────────────────────────────────────────────────────────────────────

def generate_qa(global_facts: dict, regional_facts: dict,
                sar_class_facts: dict, config: dict,
                mask: np.ndarray = None, sar: np.ndarray = None) -> list:
    """
    Generate all question-answer pairs for a single image.

    Args:
        global_facts:    Mask-derived global class statistics.
        regional_facts:  Mask-derived per-grid-cell statistics.
        sar_class_facts: SAR-derived per-class amplitude statistics.
        config:          Pipeline configuration dict.
        mask:            Raw 2-D mask array (uint8). Required for random
                         free-form region grounding bboxes.
        sar:             Raw 2-D sar array (float32). Required for SAR
                         bbox variance.

    Returns list of QA dicts (not yet formatted as conversations).
    """
    qa_all = []
    qa_all += gen_global_classification(global_facts, config)
    qa_all += gen_global_quantitative(global_facts, config)
    qa_all += gen_regional_vqa(regional_facts, global_facts, config, mask)

    # Region grounding uses random free-form bboxes computed directly on mask
    if mask is not None:
        qa_all += gen_region_grounding(mask, config)

    qa_all += gen_comparative_spatial(regional_facts, global_facts, config)

    # SAR-derived observation questions
    qa_all += gen_sar_observation(sar_class_facts, config)
    qa_all += gen_sar_class_specific(sar_class_facts, config)
    qa_all += gen_sar_comparative(sar_class_facts, config)

    # New categories
    qa_all += gen_sar_bbox_variance(sar, mask, config)
    qa_all += gen_surrounding_classes(regional_facts, mask, config)

    # Cap and shuffle
    max_q = config["pipeline"]["max_qa_per_image"]
    random.shuffle(qa_all)
    return qa_all[:max_q]


def qa_to_conversation(qa: dict) -> dict:
    """Convert a single QA dict to the conversations format."""
    return {
        "from_human": qa["question"],
        "from_gpt":   qa["answer"],
    }
