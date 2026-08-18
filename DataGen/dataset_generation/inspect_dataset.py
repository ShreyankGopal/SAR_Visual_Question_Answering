"""
inspect_dataset.py — Phase 1: Dataset inspection and statistics report.

Usage:
    python dataset_generation/inspect_dataset.py --config dataset_generation/config.yaml
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_config, read_mask, list_tif_files,
    get_class_map, ensure_dir, save_json
)


def inspect(config: dict) -> dict:
    cfg_ds   = config["dataset"]
    cfg_out  = config["output"]
    class_map = get_class_map(config)

    base_dir    = cfg_out["base_dir"]
    reports_dir = os.path.join(base_dir, cfg_out["reports_dir"])
    ensure_dir(reports_dir)

    report = {
        "source": "OpenEarthMap SAR+LULC",
        "actual_mask_encoding": "1-indexed (1–8), NOT 0-indexed",
        "class_map_detected": {str(k): v for k, v in class_map.items()},
        "splits": {},
        "image_properties": {},
        "mask_properties": {},
        "class_pixel_distribution": {},
        "missing_corrupt_pairs": [],
    }

    # ── Inspect each split ────────────────────────────────────────────────────
    for split_name, sar_key, lbl_key in [
        ("train", "train_sar_dir", "train_label_dir"),
    ]:
        sar_dir = cfg_ds[sar_key]
        lbl_dir = cfg_ds[lbl_key]

        if not os.path.exists(sar_dir):
            report["splits"][split_name] = {"error": f"{sar_dir} not found"}
            continue

        sar_tiles = set(list_tif_files(sar_dir))
        lbl_tiles = set(list_tif_files(lbl_dir))

        matched   = sar_tiles & lbl_tiles
        sar_only  = sar_tiles - lbl_tiles
        lbl_only  = lbl_tiles - sar_tiles

        report["splits"][split_name] = {
            "sar_count": len(sar_tiles),
            "label_count": len(lbl_tiles),
            "matched_pairs": len(matched),
            "sar_missing_label": sorted(sar_only),
            "label_missing_sar": sorted(lbl_only),
        }
        for f in sorted(sar_only):
            report["missing_corrupt_pairs"].append(
                {"split": split_name, "type": "missing_label", "file": f}
            )
        for f in sorted(lbl_only):
            report["missing_corrupt_pairs"].append(
                {"split": split_name, "type": "missing_sar", "file": f}
            )

    # ── Sample image & mask properties (from first train tile) ───────────────
    train_sar_dir = cfg_ds["train_sar_dir"]
    train_lbl_dir = cfg_ds["train_label_dir"]
    sample_tiles  = list_tif_files(train_sar_dir)[:5]

    all_unique_vals = set()
    dim_set         = set()
    mask_dim_set    = set()

    for tile in sample_tiles:
        import rasterio
        sar_path = os.path.join(train_sar_dir, tile)
        lbl_path = os.path.join(train_lbl_dir, tile)
        with rasterio.open(sar_path) as s:
            dim_set.add((s.count, s.height, s.width, str(s.dtypes[0])))
        mask = read_mask(lbl_path)
        mask_dim_set.add((1, mask.shape[0], mask.shape[1], str(mask.dtype)))
        all_unique_vals.update(np.unique(mask).tolist())

    dims = list(dim_set)
    report["image_properties"] = {
        "bands": dims[0][0] if dims else None,
        "height": dims[0][1] if dims else None,
        "width": dims[0][2] if dims else None,
        "dtype": dims[0][3] if dims else None,
        "consistent_across_samples": len(dims) == 1,
    }
    mdims = list(mask_dim_set)
    report["mask_properties"] = {
        "bands": mdims[0][0] if mdims else None,
        "height": mdims[0][1] if mdims else None,
        "width": mdims[0][2] if mdims else None,
        "dtype": mdims[0][3] if mdims else None,
        "consistent_across_samples": len(mdims) == 1,
        "unique_values_in_sample": sorted(all_unique_vals),
    }

    # ── Class pixel distribution across all train tiles ────────────────────
    print("Computing class distribution across all train tiles …")
    all_tiles  = list_tif_files(train_lbl_dir)
    class_totals = defaultdict(int)
    grand_total  = 0

    for tile in all_tiles:
        mask = read_mask(os.path.join(train_lbl_dir, tile))
        for cls_id in class_map:
            class_totals[cls_id] += int(np.sum(mask == cls_id))
        grand_total += mask.size

    dist = {}
    for cls_id, cls_name in class_map.items():
        px   = class_totals[cls_id]
        prop = px / grand_total if grand_total > 0 else 0.0
        dist[cls_name] = {
            "class_id": cls_id,
            "total_pixels": px,
            "proportion": round(prop, 6),
        }
    report["class_pixel_distribution"] = dist
    report["total_train_pixels"] = grand_total

    out_path = os.path.join(reports_dir, "dataset_statistics.json")
    save_json(report, out_path)
    print(f"✓ Saved: {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = inspect(cfg)

    print("\n── Summary ──────────────────────────────────")
    for split, info in report["splits"].items():
        print(f"  {split}: {info.get('matched_pairs', '?')} matched pairs")
    print(f"  Unique mask values: {report['mask_properties']['unique_values_in_sample']}")
    print(f"  Class distribution (proportion):")
    for name, v in report["class_pixel_distribution"].items():
        print(f"    [{v['class_id']}] {name:<20} {v['proportion']:.4f}")
