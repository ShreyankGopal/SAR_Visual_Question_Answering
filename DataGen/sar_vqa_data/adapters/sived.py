"""adapters/sived.py: SIVED raw files -> clean JSONL.

Reproduces the notebook cleaning:
  - rotated boxes from Annotations/*.xml (long-edge, angle in [-90, 90), + = clockwise)
  - boxes whose XML angle disagrees with the DOTA polygon by > 2 px are refit from the polygon
    (481 boxes; confirmed visually that the DOTA polygon is correct)
"""
import glob
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from core.geometry import rbox_pts
from core.io import write_jsonl


def load_dota(f):
    out = []
    for line in open(f):
        p = line.split()
        if len(p) < 9:
            continue
        try:
            out.append(np.array(list(map(float, p[:8]))).reshape(4, 2))
        except ValueError:
            continue
    return out


def corner_err(a, b):
    return min(np.abs(a - np.roll(q, k, 0)).max() for q in (b, b[::-1]) for k in range(4))


def quad_to_rbox(p):
    """Min-area rectangle of a quad -> (cx, cy, w, h, angle), long-edge convention."""
    best = None
    for i in range(4):
        e = p[(i + 1) % 4] - p[i]
        t = math.atan2(e[1], e[0])
        R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        q = p @ R
        mn, mx = q.min(0), q.max(0)
        area = np.prod(mx - mn)
        if best is None or area < best[0]:
            best = (area, t, R, mn, mx)
    _, t, R, mn, mx = best
    cx, cy = ((mn + mx) / 2) @ R.T
    w, h = mx - mn
    if w < h:
        w, h, t = h, w, t + math.pi / 2
    return cx, cy, w, h, (math.degrees(t) + 90) % 180 - 90


def ftext(e, tag):
    x = e.find(tag)
    return x.text.strip() if x is not None and x.text else None


def run(cfg):
    root = cfg["data"]["root"]
    size = cfg["data"]["image_size"]
    splits = {}
    for split in os.listdir(f"{root}/ImageSets/images"):
        for f in glob.glob(f"{root}/ImageSets/images/{split}/*.jpg"):
            splits[os.path.splitext(os.path.basename(f))[0]] = split
    dota = {os.path.splitext(os.path.basename(f))[0]: load_dota(f)
            for f in glob.glob(f"{root}/ImageSets/labelTxt/**/*.txt", recursive=True)}

    recs, n_refit = [], 0
    for f in sorted(glob.glob(f"{root}/Annotations/*.xml")):
        stem = os.path.splitext(os.path.basename(f))[0]
        r = ET.parse(f).getroot()
        source = ftext(r, "source")
        objs = []
        xml_objs = list(r.iter("object"))
        polys = dota.get(stem, [])
        paired = len(polys) == len(xml_objs)
        for i, o in enumerate(xml_objs):
            rb = o.find("rbndbox")
            box = [float(ftext(rb, k)) for k in ("cx", "cy", "w", "h", "angle")]
            src = "xml"
            if paired and corner_err(rbox_pts(*box), polys[i]) > 2:
                box, src = list(quad_to_rbox(polys[i])), "txt_refit"
                n_refit += 1
            c = rbox_pts(*box)
            objs.append(dict(category="Vehicle", rbox=[round(v, 2) for v in box],
                             hbox=[round(v, 2) for v in (*c.min(0).clip(0, size), *c.max(0).clip(0, size))],
                             rbox_src=src))
        split = splits[stem]
        recs.append(dict(dataset="SIVED", image=f"ImageSets/images/{split}/{stem}.jpg",
                         split="val" if split == "valid" else split, source=source,
                         scenario="MSTAR" if source.upper().startswith("MSTAR") else "Urban",
                         band=ftext(r, "band"), polarization=ftext(r, "polarization"),
                         resolution_m=float(ftext(r, "resolution")), native_gsd_m=cfg["native_gsd_m"][source],
                         width=size, height=size, objects=objs))
    write_jsonl(recs, cfg["data"]["clean_jsonl"])
    print(f"Wrote {len(recs)} images, {sum(len(r['objects']) for r in recs)} objects, {n_refit} boxes refit "
          f"-> {cfg['data']['clean_jsonl']}")
