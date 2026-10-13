"""validate_sived.py: checks the SIVED VQA splits.

Checks: required fields, image exists, category valid and allowed for the sensor,
conversation shape, coordinates in [0, 1], duplicate ids / questions,
split leakage, and count / presence answers recomputed from images.csv.

Usage:
  python validate_sived.py --config config_sived.yaml
"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml

REQUIRED = ("id", "image", "category", "conversations", "ground_truth_facts")


def coords_ok(v):
    flat = [x for p in v for x in (p if isinstance(p, list) else [p])]
    return all(0.0 <= x <= 1.0 for x in flat)


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    base = cfg["output"]["base_dir"]
    data_dir = os.path.join(base, cfg["output"]["data_dir"])
    img = pd.read_csv(os.path.join(base, cfg["output"]["facts_dir"], "images.csv")).set_index("stem")
    allowed = cfg["categories_per_source"]

    errors, warnings, seen_split, report = [], [], {}, {}
    for split in ("train", "val", "test"):
        recs = [json.loads(l) for l in open(os.path.join(data_dir, f"{split}.jsonl")) if l.strip()]
        ids, qs = set(), set()
        for r in recs:
            p = f"[{split}:{r.get('id')}]"
            miss = [k for k in REQUIRED if k not in r]
            if miss:
                errors.append(f"{p} missing {miss}")
                continue
            gt, conv = r["ground_truth_facts"], r["conversations"]
            stem, src = gt["image_id"], gt["source"]
            if not os.path.exists(os.path.join(cfg["data"]["root"], r["image"])):
                errors.append(f"{p} image not found")
            if r["category"] not in allowed.get(src, []):
                errors.append(f"{p} category {r['category']} not allowed for {src}")
            if len(conv) != 2 or [c.get("from") for c in conv] != ["human", "gpt"]:
                errors.append(f"{p} malformed conversations")
            for key in ("bbox", "point", "points"):
                if gt.get(key) is not None and not coords_ok(gt[key]):
                    errors.append(f"{p} {key} outside [0, 1]: {gt[key]}")
            if r["id"] in ids:
                errors.append(f"{p} duplicate id")
            ids.add(r["id"])
            qkey = (stem, conv[0]["value"])
            if qkey in qs:
                warnings.append(f"{p} duplicate question for the same image")
            qs.add(qkey)
            if seen_split.setdefault(stem, split) != split:
                errors.append(f"{p} LEAKAGE: {stem} in {seen_split[stem]} and {split}")

            # recompute answers that come straight from images.csv
            ans, row = conv[1]["value"], img.loc[stem]
            if r["category"] == "object_counting" and gt.get("subtype") == "image" and ans != f"{int(row.n_vehicles)}.":
                errors.append(f"{p} image count {ans} != {row.n_vehicles}")
            if r["category"] == "object_counting" and gt.get("subtype") == "cell" and ans != f"{int(row['count_' + gt['region']])}.":
                errors.append(f"{p} cell count {ans} != {row['count_' + gt['region']]}")
            if r["category"] == "regional_vqa":
                expect = "Yes." if row["count_" + gt["region"]] > 0 else "No."
                if ans != expect:
                    errors.append(f"{p} presence {ans} != {expect}")
        report[split] = len(recs)

    print("records:", report)
    print(f"errors: {len(errors)}  warnings: {len(warnings)}")
    for e in errors[:10]:
        print("  ERROR", e)
    for w in warnings[:5]:
        print("  WARN ", w)
    rep = os.path.join(base, cfg["output"]["reports_dir"], "validation_report.json")
    json.dump(dict(records=report, errors=errors, warnings=warnings), open(rep, "w"), indent=1)
    print(f"Saved {rep}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)