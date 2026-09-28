"""adapters/hrsid.py: HRSID (COCO JSON with ship polygons) -> clean JSONL.

  - metadata (sensor, mode, polarization, native GSD) from the panorama prefix, per HRSID Table 2 (config)
  - prefixes not in the table (P0137) are dropped
  - each polygon -> min-area rotated box (long-edge, angle in [-90, 90), + = clockwise)
  - ships whose polygon touches the tile border are flagged truncated
  - re-split by panorama: official train/test share panoramas and 200 px tile overlaps
"""
import json
import math
import os

import numpy as np
from scipy.spatial import ConvexHull

from core.geometry import rbox_pts
from core.io import write_jsonl


def poly_to_rbox(xy):
    """Min-area rectangle of a polygon -> (cx, cy, w, h, angle), long-edge convention."""
    p = xy[ConvexHull(xy).vertices] if len(xy) >= 3 else xy
    best = None
    for i in range(len(p)):
        e = p[(i + 1) % len(p)] - p[i]
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


def expand(spec):
    """'1-40, 61-81, 123' -> ['P0001', ..., 'P0123']"""
    out = []
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        out += [f"P{i:04d}" for i in range(int(a), int(b or a) + 1)]
    return out


def split_panoramas(ships_per_pano, cfg):
    """Manual assignments first; the rest greedily by ship count towards the target shares."""
    d = cfg["data"]["split"]
    split = {p: s for s, ids in d["manual"].items() for p in expand(ids)}
    rest = [p for p in ships_per_pano if p not in split]
    rng = np.random.default_rng(cfg["pipeline"]["random_seed"])
    rest = [rest[i] for i in rng.permutation(len(rest))]
    rest.sort(key=lambda p: -ships_per_pano[p])                    # big panoramas first, ties in random order
    target = d["shares"]
    got = {s: 0 for s in target}
    for p in rest:
        total = sum(got.values()) + ships_per_pano[p]
        s = max(target, key=lambda k: target[k] * total - got[k])   # split furthest below its share
        split[p] = s
        got[s] += ships_per_pano[p]
    return split


def run(cfg):
    d = cfg["data"]
    meta = {p: m for m in d["panoramas"] for p in expand(m["ids"])}
    coco = json.load(open(os.path.join(d["root"], d["annotations"])))
    name_of = {im["id"]: im["file_name"] for im in coco["images"]}
    size = {im["file_name"]: (im["width"], im["height"]) for im in coco["images"]}

    objs, dropped_poly = {}, 0
    for a in coco["annotations"]:
        seg = a["segmentation"]
        if not isinstance(seg, list) or not seg:
            dropped_poly += 1
            continue
        xy = np.array(max(seg, key=len), dtype=float).reshape(-1, 2)
        try:
            box = poly_to_rbox(xy)
        except Exception:
            dropped_poly += 1
            continue
        name = name_of[a["image_id"]]
        W, H = size[name]
        c = rbox_pts(*box)
        tol = d["border_tol_px"]
        objs.setdefault(name, []).append(dict(
            category="Ship", rbox=[round(float(v), 2) for v in box],
            hbox=[round(float(v), 2) for v in (*c.min(0).clip(0, [W, H]), *c.max(0).clip(0, [W, H]))],
            truncated=bool(xy[:, 0].min() <= tol or xy[:, 1].min() <= tol or
                           xy[:, 0].max() >= W - 1 - tol or xy[:, 1].max() >= H - 1 - tol)))

    names = sorted(n for n in objs if n[:5] in meta)
    skipped = sorted({n[:5] for n in objs if n[:5] not in meta})
    ships = {}
    for n in names:
        ships[n[:5]] = ships.get(n[:5], 0) + len(objs[n])
    split = split_panoramas(ships, cfg)

    recs = []
    for n in names:
        m = meta[n[:5]]
        W, H = size[n]
        recs.append(dict(dataset="HRSID", image=f"{d['image_dir']}/{n}", split=split[n[:5]],
                         source=m["sensor"], scenario=m["mode"], band=None, polarization=m["polarization"],
                         native_gsd_m=float(m["gsd"]), width=W, height=H, objects=objs[n]))
    write_jsonl(recs, d["clean_jsonl"])

    n_img = {s: sum(r["split"] == s for r in recs) for s in ("train", "val", "test")}
    n_obj = {s: sum(len(r["objects"]) for r in recs if r["split"] == s) for s in ("train", "val", "test")}
    trunc = sum(o["truncated"] for r in recs for o in r["objects"])
    print(f"Wrote {len(recs)} images {n_img}, ships {n_obj} ({trunc} truncated, {dropped_poly} unusable polygons, "
          f"prefixes dropped: {skipped}) -> {d['clean_jsonl']}")
    for g in sorted({r['native_gsd_m'] for r in recs}):
        sp = {s: sorted({r['image'].split('/')[-1][:5] for r in recs if r['native_gsd_m'] == g and r['split'] == s})
              for s in ("train", "val", "test")}
        print(f"  {g} m panoramas:", {s: (len(v) if len(v) > 8 else v) for s, v in sp.items()})
