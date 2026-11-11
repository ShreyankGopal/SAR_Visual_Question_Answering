#!/usr/bin/env python3
"""
linear_probe.py -- can the SAR encoder alone tell ship / aircraft / car / tank / bridge / harbour apart?

For each SARDet-100K object (COCO boxes), the image goes through the encoder exactly as in
training (raw values, bilinear resize to 512 -> 16x16 tokens), and the tokens under the box
are averaged: that is what the LLM receives for the object. A logistic-regression probe is
trained on SARDet train objects and tested on SARDet val objects.

Shortcut control: the same probe on a BACKGROUND token (a token outside every box of that
image). If background also predicts the class well, the encoder is recognising the source
dataset / scene, not the object.

Runs plain MaRS and SE-MaRS (plain + the config's `mars:` stage-A LoRA) in one go.

  python Benchmarking/linear_probe.py --config train_config_marsscaleaware.yaml \
      --sardet ~/SAR_VQA/data/SARDET_100k/SARDet_100K --out pilot_studies/linear_probe
"""
import argparse
import collections
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
GRID = 16


def pick_objects(coco: dict, per_class: int, per_image: int, rng: random.Random) -> list:
    """Up to per_class annotations per category, at most per_image per image."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_cat = collections.defaultdict(list)
    for a in coco["annotations"]:
        if a.get("bbox") and a["bbox"][2] > 0 and a["bbox"][3] > 0:
            by_cat[cats[a["category_id"]]].append(a)
    out = []
    for name, anns in sorted(by_cat.items()):
        rng.shuffle(anns)
        used = collections.Counter()
        n = 0
        for a in anns:
            if n == per_class:
                break
            if used[a["image_id"]] < per_image:
                used[a["image_id"]] += 1
                out.append({**a, "cls": name})
                n += 1
    return out


def box_tokens(bbox, w, h) -> list:
    """Token indices (row-major 16x16) covered by a COCO [x, y, w, h] box; at least the centre token."""
    x, y, bw, bh = bbox
    c0, c1 = int(x / w * GRID), math.ceil((x + bw) / w * GRID)
    r0, r1 = int(y / h * GRID), math.ceil((y + bh) / h * GRID)
    c0, r0 = min(max(c0, 0), GRID - 1), min(max(r0, 0), GRID - 1)
    c1, r1 = max(min(c1, GRID), c0 + 1), max(min(r1, GRID), r0 + 1)
    return [r * GRID + c for r in range(r0, r1) for c in range(c0, c1)]


def background_token(all_boxes, w, h, rng: random.Random):
    """A random token not touched by any box of the image (None if the boxes cover everything)."""
    covered = set()
    for b in all_boxes:
        covered.update(box_tokens(b, w, h))
    free = [t for t in range(GRID * GRID) if t not in covered]
    return rng.choice(free) if free else None


def extract(encoder, coco, objs, img_dir, device, rng, batch=16):
    """[(obj_feature, bg_feature or None, cls)] for the given objects."""
    import torch
    import torch.nn.functional as F
    from dataset import load_sar_array

    imgs = {im["id"]: im for im in coco["images"]}
    boxes = collections.defaultdict(list)
    for a in coco["annotations"]:
        boxes[a["image_id"]].append(a["bbox"])
    by_img = collections.defaultdict(list)
    for o in objs:
        by_img[o["image_id"]].append(o)
    ids, out = list(by_img), []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        x = []
        for iid in chunk:
            a = torch.from_numpy(load_sar_array(os.path.join(img_dir, imgs[iid]["file_name"])).copy()).float()
            a = a.unsqueeze(0) if a.ndim == 2 else a[:1]
            x.append(F.interpolate(a[None], size=(512, 512), mode="bilinear", align_corners=False)[0])
        with torch.no_grad():
            feats = encoder(torch.stack(x).to(device)).float().cpu().numpy()     # [B, 256, 1024]
        for f, iid in zip(feats, chunk):
            w, h = imgs[iid]["width"], imgs[iid]["height"]
            bg = background_token(boxes[iid], w, h, rng)
            for o in by_img[iid]:
                out.append((f[box_tokens(o["bbox"], w, h)].mean(0), None if bg is None else f[bg], o["cls"]))
        print(f"\r  {min(i + batch, len(ids))}/{len(ids)} images", end="", flush=True)
    print()
    return out


def probe(train, test, which: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = [(t[which], t[2]) for t in train if t[which] is not None]
    te = [(t[which], t[2]) for t in test if t[which] is not None]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    clf.fit(np.stack([f for f, _ in tr]), [c for _, c in tr])
    y, p = [c for _, c in te], clf.predict(np.stack([f for f, _ in te]))
    labels = sorted(set(y))
    cm = confusion_matrix(y, p, labels=labels)
    return {"n_train": len(tr), "n_test": len(te), "acc": float(np.mean(np.array(y) == p)),
            "per_class": {l: float(cm[i, i] / max(cm[i].sum(), 1)) for i, l in enumerate(labels)},
            "labels": labels, "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser(description="Linear probe of SAR encoder features on SARDet-100K objects")
    ap.add_argument("--config", required=True, help="training config (model.encoder_checkpoint, mars:)")
    ap.add_argument("--sardet", required=True, help="SARDet_100K folder with Annotations/ and JPEGImages/")
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--test-per-class", type=int, default=300)
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--out", default="pilot_studies/linear_probe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import yaml
    from model.mars_adapters import install_mars_adapters
    from model.sar_vlm import build_sar_encoder

    root = os.path.expanduser(args.sardet)
    cfg = yaml.safe_load(open(args.config))
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {}
    for split, n in (("train", args.per_class), ("val", args.test_per_class)):
        coco = json.load(open(os.path.join(root, "Annotations", f"{split}.json")))
        objs = pick_objects(coco, n, args.per_image, rng)
        print(f"{split}: {collections.Counter(o['cls'] for o in objs)}")
        splits[split] = (coco, objs, os.path.join(root, "JPEGImages", split))

    enc = build_sar_encoder(cfg["model"]["encoder_checkpoint"], freeze=True).to(device).eval()
    results = {}
    for name in ("plain_MaRS", "SE-MaRS"):
        if name == "SE-MaRS":
            install_mars_adapters(enc, **(cfg.get("mars") or {}))
            enc = enc.to(device).eval()
        print(f"== {name}")
        feats = {s: extract(enc, c, o, d, device, random.Random(args.seed)) for s, (c, o, d) in splits.items()}
        results[name] = {"object": probe(feats["train"], feats["val"], 0),
                         "background": probe(feats["train"], feats["val"], 1)}

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "linear_probe.json"), "w"), indent=1)
    print("\n" + "=" * 70)
    for name, r in results.items():
        for kind in ("object", "background"):
            s = r[kind]
            per = "  ".join(f"{k} {v:.2f}" for k, v in s["per_class"].items())
            print(f"{name:11s} {kind:10s} acc={s['acc']:.3f} (test n={s['n_test']})  {per}")
    print("\nconfusion (rows = true, cols = predicted), SE-MaRS object:", results["SE-MaRS"]["object"]["labels"])
    for row in results["SE-MaRS"]["object"]["confusion"]:
        print("  ", row)
    print(f"saved {os.path.join(args.out, 'linear_probe.json')}")


if __name__ == "__main__":
    main()
