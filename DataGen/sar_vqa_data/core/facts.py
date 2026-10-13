"""facts.py: per-object and per-image facts from a clean JSONL (any dataset).

Writes <facts_dir>/objects.csv, images.csv and meta.json.
Rotated boxes are used when present (SIVED); otherwise the horizontal box (OGSOD).
"""
import json

import numpy as np
import pandas as pd

from .geometry import rbox_pts, hbox_pts, poly_mask, cell_of, orientation_bin, direction_name, axial_alignment
from .io import read_jsonl, out_path, load_gray, stem_of, slug, native_gsd


def bright_threshold(cfg, recs):
    """Dataset-wide 'bright' threshold: percentile of pixels from up to 500 train images."""
    train = sorted(r["image"] for r in recs if r["split"] == "train")
    pick = np.random.default_rng(0).choice(len(train), min(500, len(train)), replace=False)
    px = np.concatenate([load_gray(cfg, train[i]).ravel()[::16] for i in sorted(pick)])
    return float(np.percentile(px, cfg.get("sar", {}).get("bright_percentile", 90)))


OBJ_COLS = ["stem", "split", "source", "obj_id", "category", "rbox_src", "truncated", "cx", "cy", "angle",
            "length_px", "width_px", "length_m", "aspect", "px", "py", "hx1", "hy1", "hx2", "hy2",
            "box_w_px", "box_h_px", "cell", "orient", "orient_margin", "mean_int", "bright"]


def image_facts(rec, cfg, T):
    W, H = rec["width"], rec["height"]
    grid = cfg["grid_regions"]
    classes = list(cfg["classes"])
    gsd = native_gsd(cfg, rec)
    img = load_gray(cfg, rec["image"])
    stem = stem_of(rec["image"])

    union = np.zeros((H, W), bool)
    per_cls = {c: np.zeros((H, W), bool) for c in classes}
    objs = []
    for i, o in enumerate(rec["objects"]):
        x1, y1, x2, y2 = o["hbox"]
        if o.get("rbox") is not None:
            cx, cy, w, h, a = o["rbox"]
            pts = rbox_pts(cx, cy, w, h, a)
            x1, y1 = pts.min(0).clip(0, [W, H])
            x2, y2 = pts.max(0).clip(0, [W, H])
            orient, orient_margin = orientation_bin(a)
        else:
            cx, cy, w, h, a = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1), min(x2 - x1, y2 - y1), np.nan
            pts = hbox_pts(x1, y1, x2, y2)
            orient, orient_margin = None, np.nan
        m = poly_mask(pts, W, H)
        union |= m
        per_cls[o["category"]] |= m
        objs.append(dict(
            stem=stem, split=rec["split"], source=rec["source"], obj_id=i, category=o["category"],
            rbox_src=o.get("rbox_src"), truncated=bool(o.get("truncated", False)),
            cx=cx, cy=cy, angle=a, length_px=w, width_px=h, length_m=w * gsd, aspect=w / h if h else np.nan,
            px=cx / W, py=cy / H,
            hx1=x1 / W, hy1=y1 / H, hx2=x2 / W, hy2=y2 / H,
            box_w_px=x2 - x1, box_h_px=y2 - y1,
            cell=cell_of(cx / W, cy / H, grid),
            orient=orient, orient_margin=orient_margin,
            mean_int=float(img[m].mean()) if m.any() else np.nan,
            bright=float((img[m] >= T).mean()) if m.any() else np.nan,
        ))
    obj = pd.DataFrame(objs, columns=OBJ_COLS)       # columns fixed so tiles with no objects still work

    # nearest neighbour (any class) and reference ambiguity
    C = obj[["cx", "cy"]].to_numpy(dtype=float).reshape(-1, 2)
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    if len(obj) > 1:
        nn = D.argmin(1)
        obj["nn_id"] = nn
        obj["nn_cls"] = obj.category.to_numpy()[nn]
        obj["nn_dist_px"] = D[np.arange(len(obj)), nn]
        obj["nn2_dist_px"] = np.sort(D, 1)[:, 1] if len(obj) > 2 else np.inf
        dirs = [direction_name(*(C[j] - C[i])) for i, j in enumerate(nn)]
        obj["nn_dir"] = [d for d, _ in dirs]
        obj["nn_dir_margin"] = [mg for _, mg in dirs]
    else:
        obj["nn_id"], obj["nn_cls"], obj["nn_dist_px"], obj["nn2_dist_px"] = -1, None, np.inf, np.inf
        obj["nn_dir"], obj["nn_dir_margin"] = None, np.nan
    hb = obj[["hx1", "hy1", "hx2", "hy2"]].to_numpy(dtype=float).reshape(-1, 4) * [W, H, W, H]
    inside = (C[None, :, 0] >= hb[:, None, 0]) & (C[None, :, 0] <= hb[:, None, 2]) & \
             (C[None, :, 1] >= hb[:, None, 1]) & (C[None, :, 1] <= hb[:, None, 3])
    np.fill_diagonal(inside, False)
    obj["n_centres_in_hbox"] = inside.sum(1)

    # image level: totals, then per class
    row = dict(stem=stem, split=rec["split"], source=rec["source"], scenario=rec.get("scenario"),
               image=rec["image"], width=W, height=H, native_gsd_m=gsd,
               n_objects=len(obj), coverage=float(union.mean()),
               alignment=axial_alignment(obj.angle.dropna()) if len(obj) and obj.angle.notna().any() else np.nan)
    counts = obj.cell.value_counts()
    for name, (x1, y1, x2, y2) in grid.items():
        row[f"count_{name}"] = int(counts.get(name, 0))
        row[f"cov_{name}"] = float(union[round(y1 * H):round(y2 * H), round(x1 * W):round(x2 * W)].mean())
    for c in classes:
        s, oc = slug(c), obj[obj.category == c]
        row[f"n__{s}"] = len(oc)
        row[f"coverage__{s}"] = float(per_cls[c].mean())
        cc = oc.cell.value_counts()
        for name, (x1, y1, x2, y2) in grid.items():
            row[f"count__{s}__{name}"] = int(cc.get(name, 0))
            row[f"cov__{s}__{name}"] = float(per_cls[c][round(y1 * H):round(y2 * H), round(x1 * W):round(x2 * W)].mean())
    return obj, row


def run(cfg):
    recs = read_jsonl(cfg["data"]["clean_jsonl"])
    T = bright_threshold(cfg, recs)
    obj_tables, img_rows = [], []
    for k, rec in enumerate(recs):
        obj, row = image_facts(rec, cfg, T)
        obj_tables.append(obj)
        img_rows.append(row)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(recs)} images")
    objects = pd.concat(obj_tables, ignore_index=True)
    images = pd.DataFrame(img_rows)
    objects.to_csv(out_path(cfg, "facts_dir", "objects.csv"), index=False)
    images.to_csv(out_path(cfg, "facts_dir", "images.csv"), index=False)
    json.dump(dict(bright_threshold=T), open(out_path(cfg, "facts_dir", "meta.json"), "w"))
    print(f"Saved {len(images)} images, {len(objects)} objects (bright threshold {T:.1f})")