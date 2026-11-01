#!/usr/bin/env python3
"""
sarlang_images.py -- bring SARLANG-1M's float SAR images into the 8-bit (0-255)
range our model was trained on (it gets raw pixel values, no normalisation).

  SARDet-100K, OpenEarthMap-SAR : already uint8 -> used as they are (not touched here)
  SpaceNet6                     : 4 polarisation bands, float (dB-like) -> band 0 (HH), stretched
  DFC2023                       : 1 band, float linear intensity -> 10*log10 (dB), stretched

Stretch = per image, 2nd-98th percentile of valid (non-zero) pixels -> 0..255; zeros
(no-data borders) stay 0.

  # 1) look first: a grid of random images per source, converted ones next to the 8-bit ones
  python sarlang_images.py --preview --out sarlang_preview.png
  # 2) then convert everything (written as PNG under the name SARLANG's JSON uses)
  python sarlang_images.py --convert --out-dir SARLANG-1M/SARimages_converted
"""
import argparse
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw

SPACENET_BAND = 0          # SpaceNet6 band order HH, HV, VH, VV -> 0 = HH


def stretch(x: np.ndarray, lo_pct=2, hi_pct=98) -> np.ndarray:
    """float image -> uint8, percentiles taken over valid (non-zero, finite) pixels."""
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x) & (x != 0)
    if not valid.any():
        return np.zeros(x.shape, np.uint8)
    lo, hi = np.percentile(x[valid], [lo_pct, hi_pct])
    out = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1) * 255
    out[~valid] = 0
    return out.round().astype(np.uint8)


def spacenet6_to_uint8(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3 and a.shape[0] <= 8 and a.shape[0] < a.shape[-1]:   # channels-first -> last
        a = np.moveaxis(a, 0, -1)
    return stretch(a[..., SPACENET_BAND] if a.ndim == 3 else a)       # values already dB-like


def dfc2023_to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    db = np.where(a > 0, 10 * np.log10(np.maximum(a, 1e-12)), 0)       # 0 stays no-data
    return stretch(db)


def sarlang_name(tif_name: str) -> str:
    """Original file name -> the name SARLANG's JSON refers to (SpaceNet6 drops '_SAR-Intensity')."""
    return os.path.splitext(tif_name.replace("_SAR-Intensity", ""))[0] + ".png"


def read(path: str) -> np.ndarray:
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        return tifffile.imread(path)
    return np.asarray(Image.open(path))


def as_uint8_gray(a: np.ndarray) -> np.ndarray:
    """8-bit sources as the model sees them: first channel."""
    return a[..., 0] if a.ndim == 3 else a


SOURCES = {   # name -> (glob relative to --data, converter or None for 8-bit sources)
    "SARDet-100K": ("SARDET_100k/SARDet_100K/JPEGImages/test/*", None),
    "OpenEarthMap-SAR": ("SARLANG-1M/SARimages_original/OpenEarthMap-SAR/*.tif", None),
    "SpaceNet6 (HH)": ("SARLANG-1M/SARimages_original/SpaceNet6/*.tif", spacenet6_to_uint8),
    "DFC2023": ("SARLANG-1M/SARimages_original/DFC2023/*.tif", dfc2023_to_uint8),
}


def preview(data: str, out: str, n: int, seed: int):
    rng, tile = random.Random(seed), 256
    rows = []
    for name, (pat, conv) in SOURCES.items():
        files = sorted(glob.glob(os.path.join(data, pat)))
        if not files:
            print(f"  {name}: no files for {pat}")
            continue
        imgs = []
        for p in rng.sample(files, min(n, len(files))):
            a = read(p)
            u8 = conv(a) if conv else as_uint8_gray(a)
            imgs.append(u8)
            print(f"  {name:18s} {os.path.basename(p)[:60]:60s} mean {u8.mean():6.1f}  std {u8.std():5.1f}  "
                  f"zeros {np.mean(u8 == 0):.0%}")
        rows.append((name, imgs))
    if not rows:
        raise SystemExit(f"No images found under {data} -- check --data")
    label_w = 150
    grid = Image.new("L", (label_w + n * tile, len(rows) * tile), 255)
    draw = ImageDraw.Draw(grid)
    for r, (name, imgs) in enumerate(rows):
        draw.text((5, r * tile + tile // 2), name, fill=0)
        for c, u8 in enumerate(imgs):
            grid.paste(Image.fromarray(u8).resize((tile, tile)), (label_w + c * tile, r * tile))
    grid.save(out)
    print(f"\nWrote {out}  (rows: {', '.join(n for n, _ in rows)})")


def convert(data: str, out_dir: str):
    for name, (pat, conv) in SOURCES.items():
        if conv is None:
            continue
        files = sorted(glob.glob(os.path.join(data, pat)))
        dst = os.path.join(out_dir, name.split()[0])
        os.makedirs(dst, exist_ok=True)
        for i, p in enumerate(files, 1):
            Image.fromarray(conv(read(p))).save(os.path.join(dst, sarlang_name(os.path.basename(p))))
            if i % 500 == 0 or i == len(files):
                print(f"  {name}: {i}/{len(files)}", flush=True)
    print(f"Done -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Convert SARLANG-1M float SAR images to 8-bit")
    ap.add_argument("--data", default=os.path.expanduser("~/SAR_VQA/data"), help="folder holding SARLANG-1M and SARDET_100k")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--convert", action="store_true")
    ap.add_argument("--out", default="sarlang_preview.png", help="preview image")
    ap.add_argument("--out-dir", default=None, help="converted images (default <data>/SARLANG-1M/SARimages_converted)")
    ap.add_argument("-n", type=int, default=6, help="images per source in the preview")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.preview:
        preview(args.data, args.out, args.n, args.seed)
    else:
        convert(args.data, args.out_dir or os.path.join(args.data, "SARLANG-1M", "SARimages_converted"))


if __name__ == "__main__":
    main()
