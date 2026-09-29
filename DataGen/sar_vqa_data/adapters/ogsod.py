"""adapters/ogsod.py: OGSOD-1.0 SAR images + YOLO labels -> clean JSONL.

  - SAR only (sar/images, sar/labels); the optical rgb/ folders are ignored
  - YOLO 'class cx cy w h' (normalized) -> pixel hbox, clipped to the image
  - boxes under min_box_px in width or height are dropped
  - val carved from train, stratified by the set of classes in each image
"""
import glob
import os

import numpy as np

from core.io import write_jsonl


def run(cfg):
    d = cfg["data"]
    root, sub, size = d["root"], d["sar_subdir"], d["image_size"]
    names = {int(k): v for k, v in d["class_ids"].items()}
    min_px = d["min_box_px"]

    recs, dropped, clipped = [], 0, 0
    for split in ("train", "test"):
        for f in sorted(glob.glob(f"{root}/{sub}/labels/{split}/*.txt")):
            stem = os.path.splitext(os.path.basename(f))[0]
            objs = []
            for line in open(f):
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * size
                cb = box.clip(0, size)
                clipped += int((cb != box).any())
                if cb[2] - cb[0] < min_px or cb[3] - cb[1] < min_px:
                    dropped += 1
                    continue
                objs.append(dict(category=names[c], hbox=[round(float(v), 2) for v in cb], rbox=None))
            if not objs:
                continue
            recs.append(dict(dataset="OGSOD", image=f"{sub}/images/{split}/{stem}.png", split=split,
                             source="OGSOD", scenario=None, band=None, polarization=None,
                             native_gsd_m=cfg["native_gsd_m"]["OGSOD"], width=size, height=size, objects=objs))

    # carve val from train, stratified by the classes present in each image
    rng = np.random.default_rng(cfg["pipeline"]["random_seed"])
    train = [r for r in recs if r["split"] == "train"]
    strata = {}
    for r in train:
        strata.setdefault(tuple(sorted({o["category"] for o in r["objects"]})), []).append(r)
    for key in sorted(strata):
        group = strata[key]
        for i in rng.permutation(len(group))[:round(d["val_fraction"] * len(group))]:
            group[i]["split"] = "val"

    write_jsonl(recs, d["clean_jsonl"])
    n = {s: sum(r["split"] == s for r in recs) for s in ("train", "val", "test")}
    print(f"Wrote {len(recs)} images {n}, {sum(len(r['objects']) for r in recs)} objects "
          f"({dropped} degenerate boxes dropped, {clipped} clipped) -> {d['clean_jsonl']}")
