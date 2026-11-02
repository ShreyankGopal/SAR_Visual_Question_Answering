#!/usr/bin/env python3
"""
sardet_leakage.py -- which SARDet-100K test images are also in OUR training data?

SARDet-100K (the image source of SARLANG-1M's SARVQA1) collects HRSID, OGSOD,
SAR-AIRcraft, SIVED, ... under new numeric names, so file names can't be matched.
Images are matched by content instead:
  exact : identical pixels (md5 of the grey image)
  thumb : same picture re-saved/re-encoded -- 32x32 grey thumbnails, cosine >= --thr

  python sardet_leakage.py \
      --sardet "~/SAR_VQA/data/SARDET_100k/SARDet_100K/JPEGImages/test/*" \
      --train ~/SAR_VQA/data/*/train_added.jsonl --val ~/SAR_VQA/data/*/val_added.jsonl \
      --test ~/SAR_VQA/data/*/test_added.jsonl --out sardet_test_leakage.csv

Output CSV: one row per SARDet test image -> found_in (train/val/test/none), method,
similarity, our_path. Crops are NOT detected (a 512 crop of an 800 image looks different);
the size table printed at the end shows whether that could matter.
"""
import argparse
import collections
import csv
import glob
import hashlib
import os
import re

import numpy as np
from PIL import Image

IMG_RE = re.compile(r'[^"\s]+\.(?:jpg|jpeg|png|bmp|tif|tiff)', re.I)
T = 32


def load_gray(p: str) -> np.ndarray:
    if p.lower().endswith((".tif", ".tiff")):
        import tifffile
        a = tifffile.imread(p)
    else:
        a = np.asarray(Image.open(p))
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] <= 4 else a[0]
    return np.ascontiguousarray(a)


def thumb(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    if a.max() > 255:                         # 16-bit / float: bring to 0..255 for a fair thumbnail
        a = 255 * (a - a.min()) / max(a.max() - a.min(), 1e-6)
    t = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((T, T), Image.BILINEAR), np.float32).ravel()
    t -= t.mean()
    return t / max(np.linalg.norm(t), 1e-6)


def describe(paths):
    """md5 dict, thumbnail matrix, sizes for a list of image files."""
    md5, thumbs, sizes, ok = {}, [], collections.Counter(), []
    for i, p in enumerate(paths, 1):
        try:
            a = load_gray(p)
        except Exception as e:                # unreadable file: report, keep going
            print(f"  skip {p}: {e}")
            continue
        md5.setdefault(hashlib.md5(a.tobytes()).hexdigest(), p)
        thumbs.append(thumb(a))
        sizes[a.shape[:2]] += 1
        ok.append(p)
        if i % 5000 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return md5, (np.stack(thumbs) if thumbs else np.zeros((0, T * T), np.float32)), sizes, ok


def images_in(jsonls):
    out = set()
    for j in jsonls:
        base = os.path.dirname(j)
        for line in open(j):
            for p in IMG_RE.findall(line):
                p = os.path.expanduser(p)
                p = p if os.path.isabs(p) else os.path.join(base, p)
                if os.path.exists(p):
                    out.add(p)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="SARDet-100K test images that are in our splits")
    ap.add_argument("--sardet", required=True, help="glob of SARDet-100K test images")
    ap.add_argument("--train", nargs="+", required=True, help="our train JSONL files")
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--thr", type=float, default=0.98, help="thumbnail cosine for 'same image'")
    ap.add_argument("--out", default="sardet_test_leakage.csv")
    args = ap.parse_args()

    ours = {}
    for split in ("train", "val", "test"):
        files = images_in(getattr(args, split))
        print(f"our {split}: {len(files)} images")
        ours[split] = describe(files)
    sd_files = sorted(glob.glob(os.path.expanduser(args.sardet)))
    print(f"SARDet test: {len(sd_files)} images")
    sd_md5, sd_th, sd_sizes, sd_ok = describe(sd_files)

    rows = {p: ["none", "", 0.0, ""] for p in sd_ok}
    md5_of = {p: h for h, p in sd_md5.items()}
    for split in ("test", "val", "train"):        # train last so it wins if an image is in several
        md5, th, _, files = ours[split]
        if len(files) == 0:
            continue
        for i in range(0, len(sd_ok), 2000):      # chunked cosine nearest neighbour
            sim = sd_th[i:i + 2000] @ th.T
            best = sim.argmax(1)
            for k, p in enumerate(sd_ok[i:i + 2000]):
                s = float(sim[k, best[k]])
                h = md5_of.get(p)
                if h in md5:
                    rows[p] = [split, "exact", 1.0, md5[h]]
                elif s >= args.thr:
                    rows[p] = [split, "thumb", round(s, 4), files[best[k]]]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sardet_file", "found_in", "method", "similarity", "our_path"])
        for p, r in rows.items():
            w.writerow([os.path.basename(p)] + r)

    c = collections.Counter((r[0], r[1]) for r in rows.values())
    print("\nSARDet test images found in our:", dict(c))
    for split in ("train", "val", "test"):
        print(f"most common sizes, our {split}:", ours[split][2].most_common(4))
    print("most common sizes, SARDet test:", sd_sizes.most_common(6))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
