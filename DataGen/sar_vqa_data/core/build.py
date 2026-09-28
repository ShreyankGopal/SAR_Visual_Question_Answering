"""build.py: candidates -> balance -> per-image cap -> train/val/test JSONL + summary."""
import json

import numpy as np
import pandas as pd

from .balance import balance
from .generators import generate_candidates
from .io import read_jsonl, out_path, stem_of

SPLIT_NAME = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def run(cfg):
    rng = np.random.default_rng(cfg["pipeline"]["random_seed"])
    obj = pd.read_csv(out_path(cfg, "facts_dir", "objects.csv"))
    img = pd.read_csv(out_path(cfg, "facts_dir", "images.csv")).sort_values("stem").reset_index(drop=True)
    meta = {stem_of(r["image"]): r for r in read_jsonl(cfg["data"]["clean_jsonl"])}
    by_stem = dict(tuple(obj.groupby("stem")))
    size_in = cfg["pipeline"]["model_input_size"]

    # 1. candidates
    rows = []
    for idx, im in img.iterrows():
        for q in generate_candidates(im, by_stem.get(im.stem, obj.iloc[0:0]), cfg, rng):
            rows.append(dict(sample_idx=idx, stem=im.stem, source=im.source, split=SPLIT_NAME[im.split], **q))
    cand = pd.DataFrame(rows)
    print(f"candidates: {len(cand)}")

    # 2. balance, 3. per-image cap
    bal = balance(cand, cfg, rng)
    bal = bal.sample(frac=1, random_state=int(rng.integers(1e9)))
    bal = bal.groupby("stem", group_keys=False).head(cfg["pipeline"]["max_qa_per_image"])
    bal = bal.sort_values(["sample_idx"]).reset_index(drop=True)
    bal["q_idx"] = bal.groupby("sample_idx").cumcount()
    print(f"after balancing and cap: {len(bal)}")

    # 4. write
    img_by_stem = img.set_index("stem")
    prefix = cfg["dataset"].lower()
    files = {s: open(out_path(cfg, "data_dir", f"{s}.jsonl"), "w") for s in ("train", "val", "test")}
    for r in bal.itertuples():
        m, im = meta[r.stem], img_by_stem.loc[r.stem]
        native = float(im.native_gsd_m)
        gsd = native * im.width / size_in               # metres per pixel at model input size
        rec = dict(
            id=f"{prefix}_{r.sample_idx:05d}_{r.q_idx:03d}",
            dataset=cfg["dataset"],
            image=m["image"],                            # relative to data.root
            category=r.category,
            ground_truth_facts=dict(image_id=r.stem, split=r.split, source=m["source"], scenario=m.get("scenario"),
                                    band=m.get("band"), polarization=m.get("polarization"),
                                    gsd_m=round(gsd, 4), native_gsd_m=native, resample_factor=round(native / gsd, 4),
                                    answer_key=r.answer_key, **r.facts),
            conversations=[{"from": "human", "value": r.question}, {"from": "gpt", "value": r.answer}],
        )
        files[r.split].write(json.dumps(rec) + "\n")
    for f in files.values():
        f.close()

    # 5. report
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    lines = ["QA pairs per category x source/split",
             str(pd.crosstab(bal.category, [bal.source, bal.split], margins=True)),
             "\nAnswer distribution per category x source (share)"]
    for (cat, src), g in bal.groupby(["category", "source"]):
        vc = g.answer_key.value_counts(normalize=True).round(3)
        lines.append(f"{cat:22s} {src:8s} " + ", ".join(f"{k}: {v}" for k, v in vc.head(8).items()))
    lines.append(f"\nQA per image: {bal.groupby('stem').size().describe().round(1).to_dict()}")
    report = "\n".join(lines)
    print(report)
    open(out_path(cfg, "reports_dir", "qa_summary.txt"), "w").write(report)
    print(f"\nWrote {out_path(cfg, 'data_dir', '')}{{train,val,test}}.jsonl")