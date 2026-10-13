"""build_sived.py: Phase 3, build train/val/test JSONL for SIVED.

Steps: generate candidates per image -> balance answers per (source, category)
       -> cap each image at max_qa_per_image -> write JSONL in the OEM record format.

Usage:
  python build_sived.py --config config_sived.yaml
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from generate_questions_sived import generate_candidates

SPLIT_NAME = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def cap_share(g, share, rng):
    """Downsample so no answer_key exceeds `share` of the category."""
    c = g.answer_key.value_counts()
    cap = int(c.max())
    while cap > 1 and cap > share * np.minimum(c, cap).sum():
        cap -= 1
    return pd.concat([x.sample(min(len(x), cap), random_state=int(rng.integers(1e9)))
                      for _, x in g.groupby("answer_key")])


def equalise(g, keys, rng):
    """Downsample rows whose answer_key is in `keys` so each key has the same count; other rows kept."""
    yn, rest = g[g.answer_key.isin(keys)], g[~g.answer_key.isin(keys)]
    n = yn.answer_key.value_counts().reindex(keys, fill_value=0).min()
    keep = [yn[yn.answer_key == k].sample(n, random_state=int(rng.integers(1e9))) for k in keys]
    return pd.concat(keep + [rest])


def balance(df, cfg, rng):
    th = cfg["thresholds"]
    out = []
    for (src, cat), g in df.groupby(["source", "category"]):
        if cat in ("regional_vqa", "arrangement", "region_grounding"):
            g = equalise(g, ["Yes", "No"], rng)
        elif cat in ("sar_comparative", "comparative_spatial", "relative_size"):
            g = equalise(g, ["first", "second"], rng)
        elif cat == "orientation":
            g = cap_share(g, th["orient_max_answer_share"], rng)
        elif cat == "object_counting":
            z, nz = g[g.answer_key == "zero"], g[g.answer_key == "nonzero"]
            max_z = int(th["counting_max_zero_share"] / (1 - th["counting_max_zero_share"]) * len(nz))
            g = pd.concat([nz, z.sample(min(len(z), max_z), random_state=int(rng.integers(1e9)))])
        out.append(g)
    return pd.concat(out)


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    rng = np.random.default_rng(cfg["pipeline"]["random_seed"])
    base = cfg["output"]["base_dir"]
    facts = os.path.join(base, cfg["output"]["facts_dir"])
    data_dir = os.path.join(base, cfg["output"]["data_dir"])
    rep_dir = os.path.join(base, cfg["output"]["reports_dir"])
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    obj = pd.read_csv(os.path.join(facts, "objects.csv"))
    img = pd.read_csv(os.path.join(facts, "images.csv")).sort_values("stem").reset_index(drop=True)
    meta = {os.path.splitext(os.path.basename(r["image"]))[0]: r
            for r in map(json.loads, open(cfg["data"]["clean_jsonl"]))}
    by_stem = dict(tuple(obj.groupby("stem")))

    # 1. candidates
    rows = []
    for idx, im in img.iterrows():
        for q in generate_candidates(im, by_stem[im.stem], cfg, rng):
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
    files = {s: open(os.path.join(data_dir, f"{s}.jsonl"), "w") for s in ("train", "val", "test")}
    for r in bal.itertuples():
        m = meta[r.stem]
        rec = dict(
            id=f"sived_{r.sample_idx:05d}_{r.q_idx:03d}",
            dataset="SIVED",
            image=m["image"],                      # relative to data.root
            category=r.category,
            ground_truth_facts=dict(image_id=r.stem, split=r.split, source=m["source"], scenario=m["scenario"],
                                    band=m["band"], polarization=m["polarization"], gsd_m=cfg["gsd_m"][m["source"]],
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
    open(os.path.join(rep_dir, "qa_summary.txt"), "w").write(report)
    print(f"\nWrote {data_dir}/{{train,val,test}}.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)