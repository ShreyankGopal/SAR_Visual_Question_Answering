"""validate.py: checks the train/val/test JSONL of any dataset.

Checks: required fields, image exists, category allowed for the source, conversation shape,
coordinates in [0, 1], GSD fields, duplicate ids / questions, split leakage, and count /
presence answers recomputed from images.csv.
"""
import json
import os

import pandas as pd

from .io import out_path, slug, read_jsonl

REQUIRED = ("id", "image", "category", "conversations", "ground_truth_facts")


def coords_ok(v):
    flat = [x for p in v for x in (p if isinstance(p, list) else [p])]
    return all(0.0 <= x <= 1.0 for x in flat)


def check_record(r, cfg, img, p, errors):
    gt, conv = r["ground_truth_facts"], r["conversations"]
    stem, src, cat = gt["image_id"], gt["source"], r["category"]
    if not os.path.exists(os.path.join(cfg["data"]["root"], r["image"])):
        errors.append(f"{p} image not found")
    if cat not in cfg["categories_per_source"].get(src, []):
        errors.append(f"{p} category {cat} not allowed for {src}")
    if len(conv) != 2 or [c.get("from") for c in conv] != ["human", "gpt"]:
        errors.append(f"{p} malformed conversations")
    for key in ("gsd_m", "native_gsd_m", "resample_factor"):
        if not gt.get(key):
            errors.append(f"{p} missing {key}")
    for key in ("bbox", "point", "points"):
        if gt.get(key) is not None and not coords_ok(gt[key]):
            errors.append(f"{p} {key} outside [0, 1]: {gt[key]}")

    ans, row = conv[1]["value"], img.loc[stem]
    s = slug(gt["cls"]) if gt.get("cls") else None
    if cat == "object_counting" and gt.get("subtype") == "image" and ans != f"{int(row[f'n__{s}'])}.":
        errors.append(f"{p} image count {ans} != {row[f'n__{s}']}")
    if cat == "object_counting" and gt.get("subtype") == "cell" and ans != f"{int(row[f'count__{s}__' + gt['region']])}.":
        errors.append(f"{p} cell count {ans} != {row[f'count__{s}__' + gt['region']]}")
    if cat == "regional_vqa" and s:
        expect = "Yes." if row[f"count__{s}__" + gt["region"]] > 0 else "No."
        if ans != expect:
            errors.append(f"{p} presence {ans} != {expect}")
    if cat == "global_classification" and s:
        expect = "Yes." if row[f"n__{s}"] > 0 else "No."
        if ans != expect:
            errors.append(f"{p} image presence {ans} != {expect}")


def run(cfg):
    img = pd.read_csv(out_path(cfg, "facts_dir", "images.csv")).set_index("stem")
    errors, warnings, seen_split, counts = [], [], {}, {}
    for split in ("train", "val", "test"):
        path = out_path(cfg, "data_dir", f"{split}.jsonl")
        recs = read_jsonl(path) if os.path.exists(path) else []
        ids, qs = set(), set()
        for r in recs:
            p = f"[{split}:{r.get('id')}]"
            miss = [k for k in REQUIRED if k not in r]
            if miss:
                errors.append(f"{p} missing {miss}")
                continue
            check_record(r, cfg, img, p, errors)
            if r["id"] in ids:
                errors.append(f"{p} duplicate id")
            ids.add(r["id"])
            stem = r["ground_truth_facts"]["image_id"]
            qkey = (stem, r["conversations"][0]["value"])
            if qkey in qs:
                warnings.append(f"{p} duplicate question for the same image")
            qs.add(qkey)
            if seen_split.setdefault(stem, split) != split:
                errors.append(f"{p} LEAKAGE: {stem} in {seen_split[stem]} and {split}")
        counts[split] = len(recs)

    print("records:", counts)
    print(f"errors: {len(errors)}  warnings: {len(warnings)}")
    for e in errors[:10]:
        print("  ERROR", e)
    for w in warnings[:5]:
        print("  WARN ", w)
    rep = out_path(cfg, "reports_dir", "validation_report.json")
    json.dump(dict(records=counts, errors=errors, warnings=warnings), open(rep, "w"), indent=1)
    print(f"Saved {rep}")
    return not errors
