"""spot_check.py: draw QA pairs on their images (use in Jupyter).

  from core.spot_check import show
  show('configs/ogsod.yaml', category='object_classification', n=6)
  show('configs/ogsod.yaml', category='region_grounding', n=6, save='output/ogsod/reports/spot_grounding.png')
"""
import os
import random

import matplotlib.pyplot as plt
from PIL import Image

from .io import load_config, out_path, read_jsonl


def _rect(ax, box, W, H, color):
    x1, y1, x2, y2 = box
    ax.add_patch(plt.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False, ec=color, lw=2))


def show(config_path, category=None, source=None, split="train", n=6, seed=None, save=None):
    cfg = load_config(config_path)
    recs = [r for r in read_jsonl(out_path(cfg, "data_dir", f"{split}.jsonl"))
            if (category is None or r["category"] == category)
            and (source is None or r["ground_truth_facts"]["source"] == source)]
    recs = random.Random(seed).sample(recs, min(n, len(recs)))
    rows = (len(recs) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6.5 * rows), squeeze=False)
    for ax, r in zip(axes.flat, recs):
        im = Image.open(os.path.join(cfg["data"]["root"], r["image"])).convert("L")
        W, H = im.size
        ax.imshow(im, cmap="gray")
        gt = r["ground_truth_facts"]
        if gt.get("bbox"):
            _rect(ax, gt["bbox"], W, H, "cyan")
        for rg in ([gt["region"]] if gt.get("region") else []) + gt.get("regions", []):
            if rg in cfg["grid_regions"]:
                _rect(ax, cfg["grid_regions"][rg], W, H, "yellow")
        for i, p in enumerate(gt.get("points") or ([gt["point"]] if gt.get("point") else [])):
            ax.plot(p[0] * W, p[1] * H, "o", ms=10, mfc="none", mec=["red", "lime"][i % 2], mew=2)
        q, a = (c["value"] for c in r["conversations"])
        ax.set_title(f"{r['id']} | {gt['source']}\nQ: {q}\nA: {a}", fontsize=8, wrap=True)
        ax.axis("off")
    for ax in list(axes.flat)[len(recs):]:
        ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100, bbox_inches="tight")
    plt.show()
