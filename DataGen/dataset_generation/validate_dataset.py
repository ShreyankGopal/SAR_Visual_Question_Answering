"""
validate_dataset.py — Phase 10: Automated JSONL dataset validator.

Usage:
    python dataset_generation/validate_dataset.py --config dataset_generation/config.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, read_mask, get_class_map, save_json, ensure_dir

VALID_CATEGORIES = {
    "global_classification",
    "global_quantitative",
    "regional_vqa",
    "region_grounding",
    "comparative_spatial",
    "sar_observation",    # relative SAR amplitude observation questions
    "sar_comparative",    # pairwise SAR response comparisons
    "sar_bbox_variance",  # new SAR variance questions
    "surrounding_classes", # new spatial adjacency questions
}

VALID_REGIONS = {
    "top_left", "top", "top_right",
    "left", "center", "right",
    "bottom_left", "bottom", "bottom_right",
}


def validate_split(jsonl_path: str, config: dict, split_name: str,
                   all_image_ids: dict, base_dir: str) -> dict:
    """Validate a single JSONL split file. Returns error report."""
    class_map   = get_class_map(config)
    valid_names = set(class_map.values())
    errors      = []
    warnings    = []
    seen_ids    = set()
    seen_qa     = set()

    if not os.path.exists(jsonl_path):
        return {"error": f"File not found: {jsonl_path}"}

    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    for i, rec in enumerate(records):
        rid = rec.get("id", f"line_{i}")
        prefix = f"[{split_name}:{rid}]"

        # 1. Check required fields
        for field in ("id", "image", "mask", "category", "conversations", "ground_truth_facts"):
            if field not in rec:
                errors.append(f"{prefix} Missing field: '{field}'")

        # 2. Image exists — path is relative to output base_dir
        img_rel = rec.get("image", "")
        img_abs = os.path.join(base_dir, img_rel)
        if not os.path.exists(img_abs):
            errors.append(f"{prefix} Image not found: {img_abs}")

        # 3. Mask exists — path is relative to output base_dir
        msk_rel = rec.get("mask", "")
        msk_abs = os.path.join(base_dir, msk_rel)
        if not os.path.exists(msk_abs):
            errors.append(f"{prefix} Mask not found: {msk_abs}")

        # 4. Category valid
        cat = rec.get("category", "")
        if cat not in VALID_CATEGORIES:
            errors.append(f"{prefix} Invalid category: '{cat}'")

        # 5. Conversations structure
        convs = rec.get("conversations", [])
        if not convs:
            errors.append(f"{prefix} Empty conversations list")
        for c in convs:
            if not isinstance(c, dict) or "from" not in c or "value" not in c:
                errors.append(f"{prefix} Malformed conversation entry: {c}")

        # 6. No duplicate IDs within split
        if rid in seen_ids:
            warnings.append(f"{prefix} Duplicate ID in split")
        seen_ids.add(rid)

        # 7. No duplicate QA (question text)
        for c in convs:
            if c.get("from") == "human":
                q_key = (rid[:15], c["value"][:80])
                if q_key in seen_qa:
                    warnings.append(f"{prefix} Duplicate question detected")
                seen_qa.add(q_key)

        # 8. No unsupported class names in answers
        for c in convs:
            if c.get("from") == "gpt":
                ans = c["value"]
                for word in ans.split():
                    w = word.strip(".,;:?!").title()
                    # Only flag if it looks like a class name that's wrong
                    # (basic heuristic: known bad names)
                    bad = {"Object", "Instance", "Bounding", "Box", "Count", "Number"}
                    if w in bad:
                        warnings.append(f"{prefix} Potentially hallucinated object term: '{w}'")

        # 9. Region coordinates valid (if present)
        bbox = rec.get("ground_truth_facts", {}).get("bbox")
        if bbox:
            if not (len(bbox) == 4 and all(0.0 <= v <= 1.0 for v in bbox)):
                errors.append(f"{prefix} Invalid bbox coordinates: {bbox}")

        # 10. Track image IDs across splits for leakage check
        img_id = rec.get("ground_truth_facts", {}).get("image_id", rec.get("id", ""))
        if img_id in all_image_ids and all_image_ids[img_id] != split_name:
            errors.append(
                f"{prefix} DATA LEAKAGE: {img_id} appears in both "
                f"'{all_image_ids[img_id]}' and '{split_name}'"
            )
        all_image_ids[img_id] = split_name

    return {
        "split": split_name,
        "total_records": len(records),
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
    }


def validate(config: dict):
    cfg_out   = config["output"]
    base_dir  = cfg_out["base_dir"]
    data_dir  = os.path.join(base_dir, cfg_out["data_dir"])
    rep_dir   = os.path.join(base_dir, cfg_out["reports_dir"])
    ensure_dir(rep_dir)

    all_image_ids = {}  # used across splits for leakage detection
    report = {"splits": {}, "overall_passed": True}

    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}.jsonl")
        result = validate_split(path, config, split, all_image_ids, base_dir)
        report["splits"][split] = result
        if not result.get("passed", False):
            report["overall_passed"] = False

    out_path = os.path.join(rep_dir, "validation_report.json")
    save_json(report, out_path)

    # Print summary
    print("\n── Validation Report ────────────────────────────────")
    for split, res in report["splits"].items():
        status = "✓ PASS" if res.get("passed") else "✗ FAIL"
        nerrs  = len(res.get("errors", []))
        nwarn  = len(res.get("warnings", []))
        print(f"  {split:6s}: {status}  |  {res.get('total_records',0)} records  "
              f"|  {nerrs} errors  |  {nwarn} warnings")
        for e in res.get("errors", [])[:5]:
            print(f"    ERROR: {e}")
        for w in res.get("warnings", [])[:3]:
            print(f"    WARN : {w}")

    print(f"\nSaved: {out_path}")
    if report["overall_passed"]:
        print("✓ All splits passed validation.")
    else:
        print("✗ Validation failed — see report for details.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate(cfg)
