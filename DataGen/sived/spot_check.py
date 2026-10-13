"""spot_check.py: draw QA pairs on their images for visual checking (use in Jupyter).

  from spot_check import show
  show('config_sived.yaml', category='orientation', n=6)
"""
import json
import os
import random

import matplotlib.pyplot as plt
import yaml
from PIL import Image


def show(config_path, category=None, source=None, split="train", n=6, seed=None, save=True):
    cfg = yaml.safe_load(open(config_path))
    path = os.path.join(cfg["output"]["base_dir"], cfg["output"]["data_dir"], f"{split}.jsonl")
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if (category is None or r["category"] == category)
            and (source is None or r["ground_truth_facts"]["source"] == source)]
    recs = random.Random(seed).sample(recs, min(n, len(recs)))

    fig, axes = plt.subplots((len(recs) + 2) // 3, 3, figsize=(18, 6.5 * ((len(recs) + 2) // 3)), squeeze=False)
    for ax, r in zip(axes.flat, recs):
        im = Image.open(os.path.join(cfg["data"]["root"], r["image"]))
        W, H = im.size
        ax.imshow(im, cmap="gray")
        gt = r["ground_truth_facts"]
        if gt.get("bbox"):
            x1, y1, x2, y2 = gt["bbox"]
            ax.add_patch(plt.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False, ec="cyan", lw=2))
        if gt.get("region") and gt["region"] in cfg["grid_regions"]:
            x1, y1, x2, y2 = cfg["grid_regions"][gt["region"]]
            ax.add_patch(plt.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False, ec="yellow", lw=2))
        for i, p in enumerate(gt.get("points") or ([gt["point"]] if gt.get("point") else [])):
            ax.plot(p[0] * W, p[1] * H, "o", ms=10, mfc="none", mec=["red", "lime"][i % 2], mew=2)
        for rg in gt.get("regions", []):
            x1, y1, x2, y2 = cfg["grid_regions"][rg]
            ax.add_patch(plt.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False, ec="yellow", lw=2))
        q, a = (c["value"] for c in r["conversations"])
        ax.set_title(f"{r['id']} | {gt['source']}\nQ: {q}\nA: {a}", fontsize=8, wrap=True)
        ax.axis("off")
    for ax in list(axes.flat)[len(recs):]:
        ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100, bbox_inches="tight")
    plt.show()