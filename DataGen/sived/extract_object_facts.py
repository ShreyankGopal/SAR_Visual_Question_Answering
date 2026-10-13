"""extract_object_facts.py: Phase 1, per-vehicle and per-image facts for SIVED.

Reads sived_clean.jsonl and writes two tables to <base_dir>/<facts_dir>/:
  objects.csv  one row per vehicle
  images.csv   one row per image (with per-grid-cell counts and coverage)

Usage:
  python extract_object_facts.py --config config_sived.yaml
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from geometry import rbox_pts, poly_mask, cell_of, orientation_bin, direction_name, axial_alignment


def image_facts(rec, cfg):
    W, H = rec["width"], rec["height"]
    grid = cfg["grid_regions"]
    gsd = cfg["gsd_m"][rec["source"]]
    img = np.asarray(Image.open(os.path.join(cfg["data"]["root"], rec["image"])).convert("L"), dtype=np.float32)
    stem = os.path.splitext(os.path.basename(rec["image"]))[0]

    union = np.zeros((H, W), bool)
    objs = []
    for i, o in enumerate(rec["objects"]):
        cx, cy, w, h, a = o["rbox"]
        pts = rbox_pts(cx, cy, w, h, a)
        m = poly_mask(pts, W, H)
        union |= m
        x1, y1 = pts.min(0).clip(0, [W, H])
        x2, y2 = pts.max(0).clip(0, [W, H])
        orient, orient_margin = orientation_bin(a)
        objs.append(dict(
            stem=stem, split=rec["split"], source=rec["source"], obj_id=i, rbox_src=o["rbox_src"],
            cx=cx, cy=cy, angle=a, length_px=w, width_px=h, length_m=w * gsd, aspect=w / h,
            px=cx / W, py=cy / H,                                  # point reference
            hx1=x1 / W, hy1=y1 / H, hx2=x2 / W, hy2=y2 / H,        # box reference
            cell=cell_of(cx / W, cy / H, grid),
            orient=orient, orient_margin=orient_margin,
            mean_int=float(img[m].mean()) if m.any() else np.nan,
        ))
    obj = pd.DataFrame(objs)

    # nearest neighbours and reference ambiguity
    C = obj[["cx", "cy"]].to_numpy()
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    if len(obj) > 1:
        nn = D.argmin(1)
        obj["nn_id"] = nn
        obj["nn_dist_px"] = D[np.arange(len(obj)), nn]
        obj["nn2_dist_px"] = np.sort(D, 1)[:, 1] if len(obj) > 2 else np.inf
        dirs = [direction_name(*(C[j] - C[i])) for i, j in enumerate(nn)]
        obj["nn_dir"] = [d for d, _ in dirs]
        obj["nn_dir_margin"] = [mg for _, mg in dirs]
    else:
        obj["nn_id"], obj["nn_dist_px"], obj["nn2_dist_px"] = -1, np.inf, np.inf
        obj["nn_dir"], obj["nn_dir_margin"] = None, np.nan
    hb = obj[["hx1", "hy1", "hx2", "hy2"]].to_numpy() * [W, H, W, H]
    inside = (C[None, :, 0] >= hb[:, None, 0]) & (C[None, :, 0] <= hb[:, None, 2]) & \
             (C[None, :, 1] >= hb[:, None, 1]) & (C[None, :, 1] <= hb[:, None, 3])
    np.fill_diagonal(inside, False)
    obj["n_centres_in_hbox"] = inside.sum(1)

    # image level
    row = dict(stem=stem, split=rec["split"], source=rec["source"], scenario=rec["scenario"],
               image=rec["image"], n_vehicles=len(obj), coverage=float(union.mean()),
               alignment=axial_alignment(obj.angle))
    counts = obj.cell.value_counts()
    for name, (x1, y1, x2, y2) in grid.items():
        row[f"count_{name}"] = int(counts.get(name, 0))
        row[f"cov_{name}"] = float(union[round(y1 * H):round(y2 * H), round(x1 * W):round(x2 * W)].mean())
    return obj, row


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    out_dir = os.path.join(cfg["output"]["base_dir"], cfg["output"]["facts_dir"])
    os.makedirs(out_dir, exist_ok=True)

    recs = [json.loads(l) for l in open(cfg["data"]["clean_jsonl"])]
    obj_tables, img_rows = [], []
    for k, rec in enumerate(recs):
        obj, row = image_facts(rec, cfg)
        obj_tables.append(obj)
        img_rows.append(row)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(recs)} images")

    objects = pd.concat(obj_tables, ignore_index=True)
    images = pd.DataFrame(img_rows)
    objects.to_csv(os.path.join(out_dir, "objects.csv"), index=False)
    images.to_csv(os.path.join(out_dir, "images.csv"), index=False)
    print(f"Saved {len(images)} images, {len(objects)} vehicles to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
