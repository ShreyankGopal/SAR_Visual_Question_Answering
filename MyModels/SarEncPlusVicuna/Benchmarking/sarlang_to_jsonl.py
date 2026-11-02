#!/usr/bin/env python3
"""
sarlang_to_jsonl.py -- SARLANG-1M VQA test JSON -> our benchmark JSONL.

SARLANG items look like
  {"messages": [{"role": "user", "content": "<image>question"}, {"role": "assistant", "content": "answer"}, ...],
   "images": ["/your_path/SARimages/<name>.png"]}
The image path is a placeholder, so images are matched by file NAME only:
  part 1 (SARDet-100K) : <data>/SARDET_100k/SARDet_100K/JPEGImages/{test,val,train}/<name>.{jpg,png,bmp}
  part 2               : OEM  <data>/SARLANG-1M/SARimages_original/OpenEarthMap-SAR/<name>.tif (8-bit, as is)
                         SpaceNet6 / DFC2023 PNGs from sarlang_images.py --convert
Prompt = SARLANG's: "Question: {q}. Answer the question concisely."
Each record carries ground_truth_facts.source and .leaked (image is in OUR training data,
from sardet_test_leakage.csv / oem_test_in_our_train.txt), so scores can be split later.
Images that can't be found are skipped and counted.

  python Benchmarking/sarlang_to_jsonl.py --vqa-json <SARVQA1 test json> --out sarlang_sardet_test.jsonl
  python Benchmarking/sarlang_to_jsonl.py --vqa-json <SARVQA2 test json> --out sarlang_other_test.jsonl
"""
import argparse
import collections
import csv
import glob
import json
import os

PROMPT = "Question: {q}. Answer the question concisely."


def index_images(data: str) -> dict:
    """file stem -> (absolute path, source). SARDet test wins over its val/train."""
    idx = {}
    sl = os.path.join(data, "SARLANG-1M")
    groups = [
        ("SpaceNet6", os.path.join(sl, "SARimages_converted", "SpaceNet6", "*.png")),
        ("DFC2023", os.path.join(sl, "SARimages_converted", "DFC2023", "*.png")),
        ("OEM", os.path.join(sl, "SARimages_original", "OpenEarthMap-SAR", "*.tif")),
    ] + [("SARDet-100K", os.path.join(data, "SARDET_100k", "SARDet_100K", "JPEGImages", s, "*"))
         for s in ("train", "val", "test")]
    for source, pat in groups:              # later groups overwrite earlier -> SARDet test last
        for p in glob.glob(pat):
            idx[os.path.splitext(os.path.basename(p))[0]] = (p, source)
    return idx


def leaked_names(data: str) -> set:
    sl = os.path.join(data, "SARLANG-1M")
    out = set()
    for csv_path in glob.glob(os.path.join(sl, "sardet_*_leakage.csv")):   # one per SARDet folder checked
        out |= {os.path.splitext(r["sardet_file"])[0] for r in csv.DictReader(open(csv_path))
                if r["found_in"] == "train"}
    txt = os.path.join(sl, "oem_test_in_our_train.txt")
    if os.path.exists(txt):
        out |= {l.strip() for l in open(txt) if l.strip()}
    print(f"leaked image names loaded: {len(out)}")
    return out


def convert(items: list, idx: dict, leaked: set, tag: str):
    """Yield our records; count what was skipped."""
    miss = collections.Counter()
    for i, it in enumerate(items):
        name = os.path.splitext(os.path.basename(it["images"][0]))[0]
        hit = idx.get(name)
        if hit is None:
            miss["image not found"] += 1
            continue
        path, source = hit
        msgs = it["messages"]
        for t in range(0, len(msgs) - 1, 2):          # user/assistant pairs (usually just one)
            q = msgs[t]["content"].replace("<image>", "").strip()
            a = msgs[t + 1]["content"].strip()
            yield {
                "id": f"{tag}_{i:06d}_{t // 2}",
                "dataset": f"SARLANG-{source}",
                "image": path,                            # absolute; data_root is ignored by os.path.join
                "category": source,
                "ground_truth_facts": {"source": source, "leaked": name in leaked, "image_name": name},
                "conversations": [{"from": "human", "value": PROMPT.format(q=q)},
                                  {"from": "gpt", "value": a}],
            }
    for k, v in miss.items():
        print(f"  skipped ({k}): {v}")


def main():
    ap = argparse.ArgumentParser(description="SARLANG-1M VQA test JSON -> benchmark JSONL")
    ap.add_argument("--vqa-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/SAR_VQA/data"))
    ap.add_argument("--tag", default=None, help="id prefix (default: json file name)")
    args = ap.parse_args()

    items = json.load(open(args.vqa_json))
    print(f"{args.vqa_json}: {len(items)} items")
    idx = index_images(args.data)
    print(f"images indexed: {len(idx)}")
    tag = args.tag or os.path.splitext(os.path.basename(args.vqa_json))[0]
    n, by = 0, collections.Counter()
    with open(args.out, "w") as f:
        for r in convert(items, idx, leaked_names(args.data), tag):
            f.write(json.dumps(r) + "\n")
            n += 1
            by[(r["category"], r["ground_truth_facts"]["leaked"])] += 1
    print(f"wrote {n} questions -> {args.out}")
    for (src, lk), c in sorted(by.items()):
        print(f"  {src:12s} leaked={lk!s:5s} {c}")


if __name__ == "__main__":
    main()
