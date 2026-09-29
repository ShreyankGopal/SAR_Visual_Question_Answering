"""adapters/sar_aircraft.py: SAR-AIRcraft-1.0 (VOC XML, 800-1500 px) -> 512 px tiles + clean JSONL.

  - official train / val / test lists kept; tiles are cut within each image, so no tile crosses splits
  - tiles at native resolution, evenly spaced with overlap (800: 2x2, 1000: 2x2, 1200: 3x3, 1500: 3x3)
  - each tile saved as PNG with a 1st-99.5th percentile contrast stretch (aircraft are dim in the raw data)
  - one class 'Aircraft'; the fine-grained type is kept on each object as metadata only
  - aircraft cut by the tile edge are clipped and flagged truncated; slivers (<25% of the aircraft visible) dropped
  - tiles without aircraft kept only up to empty_share of the tiles in each split
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from core.io import write_jsonl


def offsets(n, t):
    return [round(x) for x in np.linspace(0, n - t, math.ceil(n / t))]


def stretch(a, lo, hi):
    p1, p2 = np.percentile(a, [lo, hi])
    return (np.clip((a - p1) / max(p2 - p1, 1e-6), 0, 1) * 255).astype(np.uint8)


def run(cfg):
    d = cfg["data"]
    raw, T = d["raw_root"], d["tile_size"]
    lo, hi = d["stretch_percentiles"] or (0, 100)
    rng = np.random.default_rng(cfg["pipeline"]["random_seed"])

    split_of = {}
    for s in ("train", "val", "test"):
        for i in open(f"{raw}/ImageSets/Main/{s}.txt").read().split():
            split_of[i] = s

    tiles = []
    for stem in sorted(split_of):
        r = ET.parse(f"{raw}/Annotations/{stem}.xml").getroot()
        boxes = []
        for o in r.findall("object"):
            b = o.find("bndbox")
            boxes.append((o.findtext("name"), *(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
        im = np.asarray(Image.open(f"{raw}/JPEGImages/{stem}.jpg").convert("L"), dtype=np.float32)
        H, W = im.shape
        for oy in offsets(H, T):
            for ox in offsets(W, T):
                objs = []
                for name, x1, y1, x2, y2 in boxes:
                    cx1, cy1 = max(x1, ox) - ox, max(y1, oy) - oy
                    cx2, cy2 = min(x2, ox + T) - ox, min(y2, oy + T) - oy
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    visible = (cx2 - cx1) * (cy2 - cy1) / ((x2 - x1) * (y2 - y1))
                    if visible < d["min_visible_fraction"]:
                        continue
                    objs.append(dict(category="Aircraft", type=name, rbox=None,
                                     hbox=[round(v, 2) for v in (cx1, cy1, cx2, cy2)], truncated=bool(visible < 1)))
                rel = f"{split_of[stem]}/{stem}_{oy}_{ox}.png"      # saved now; unused empty tiles removed below
                path = os.path.join(d["root"], rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tile = im[oy:oy + T, ox:ox + T]
                Image.fromarray(tile.astype(np.uint8) if d["stretch_percentiles"] is None else stretch(tile, lo, hi)).save(path)
                tiles.append(dict(stem=stem, objs=objs, split=split_of[stem], image=rel))

    # cap tiles without aircraft per split
    keep = []
    for s in ("train", "val", "test"):
        full = [t for t in tiles if t["split"] == s and t["objs"]]
        empty = [t for t in tiles if t["split"] == s and not t["objs"]]
        n_empty = min(len(empty), round(d["empty_share"] / (1 - d["empty_share"]) * len(full)))
        pick = set(rng.choice(len(empty), n_empty, replace=False).tolist()) if n_empty else set()
        chosen = [t for i, t in enumerate(empty) if i in pick]
        for i, t in enumerate(empty):
            if i not in pick:
                os.remove(os.path.join(d["root"], t["image"]))
        keep += full + chosen

    recs = [dict(dataset="SAR-AIRcraft", image=t["image"], split=t["split"], source="SAR-AIRcraft",
                 scenario=None, band=None, polarization=None, native_gsd_m=cfg["native_gsd_m"]["SAR-AIRcraft"],
                 resolution_m=d["stated_resolution_m"], parent=t["stem"], width=T, height=T, objects=t["objs"])
            for t in keep]
    write_jsonl(recs, d["clean_jsonl"])
    n = {s: sum(r["split"] == s for r in recs) for s in ("train", "val", "test")}
    n_obj = sum(len(r["objects"]) for r in recs)
    n_tr = sum(o["truncated"] for r in recs for o in r["objects"])
    print(f"Wrote {len(recs)} tiles {n} from {len(split_of)} images ({sum(not r['objects'] for r in recs)} without aircraft), "
          f"{n_obj} aircraft ({n_tr} truncated) -> {d['clean_jsonl']}")