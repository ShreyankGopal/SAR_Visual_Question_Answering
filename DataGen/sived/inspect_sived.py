"""inspect_sived.py: Phase 1 review. Distributions used to set Phase 2 thresholds.

Prints a text report and saves reports/phase1_distributions.png.
No thresholds are applied here; candidate values are shown side by side.

Usage:
  python inspect_sived.py --config config_sived.yaml
"""
import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def q(s):
    return s.quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_dict()


def pairs_within_images(obj):
    """All vehicle pairs within each image: intensity difference and length ratio."""
    rows = []
    for (stem, src), g in obj.groupby(["stem", "source"]):
        i, j = np.triu_indices(len(g), 1)
        v, L = g.mean_int.to_numpy(), g.length_px.to_numpy()
        rows.append(pd.DataFrame(dict(source=src, int_diff=np.abs(v[i] - v[j]),
                                      len_ratio=np.maximum(L[i], L[j]) / np.minimum(L[i], L[j]))))
    return pd.concat(rows, ignore_index=True)


def cell_pairs(img, cells):
    """All grid-cell pairs within each image: count difference."""
    rows = []
    for a, b in itertools.combinations(cells, 2):
        rows.append(pd.DataFrame(dict(source=img.source,
                                      diff=(img[f"count_{a}"] - img[f"count_{b}"]).abs())))
    return pd.concat(rows, ignore_index=True)


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    base = cfg["output"]["base_dir"]
    facts = os.path.join(base, cfg["output"]["facts_dir"])
    rep = os.path.join(base, cfg["output"]["reports_dir"])
    os.makedirs(rep, exist_ok=True)
    obj = pd.read_csv(os.path.join(facts, "objects.csv"))
    img = pd.read_csv(os.path.join(facts, "images.csv"))
    cells = list(cfg["grid_regions"])
    urban = img.source.isin(["FARAD", "MiniSAR"])

    section("1. Images and vehicles per source / split")
    print(img.groupby(["source", "split"]).size().unstack(fill_value=0))
    print(obj.groupby(["source", "split"]).size().unstack(fill_value=0))

    section("2. Vehicles per image (object_counting, whole image)")
    for s, g in img.groupby("source"):
        print(s, q(g.n_vehicles))

    section("3. Vehicle coverage % (global_quantitative bins)")
    for s, g in img.groupby("source"):
        print(s, q(g.coverage * 100))
    print("\nCandidate bins, share of images:")
    bins = [0, 2, 5, 10, 20, 100]
    print(pd.crosstab(img.source, pd.cut(img.coverage * 100, bins, right=False), normalize="index").round(3))

    section("4. Vehicles per grid cell (counting per cell, regional_vqa yes/no)")
    long = img.melt(id_vars="source", value_vars=[f"count_{c}" for c in cells], value_name="n")
    print(pd.crosstab(long.source, long.n.clip(upper=10), normalize="index").round(3))
    print("(column 10 = 10 or more; column 0 = empty cells, the 'No' pool for regional_vqa)")

    section("5. Cell-pair count differences (comparative_spatial, FARAD + MiniSAR)")
    cp = cell_pairs(img[urban], cells)
    print(pd.crosstab(cp.source, cp["diff"].clip(upper=5), normalize="index").round(3))
    print("(column 0 = ties; a min-diff of 1 or 2 drops the first one or two columns)")

    section("6. Orientation bins (orientation)")
    print(pd.crosstab(obj.source, obj.orient, normalize="index").round(3))
    for m in (3, 5, 10):
        print(f"within {m} deg of a bin edge:", obj.groupby("source").orient_margin.apply(lambda x: round((x < m).mean(), 3)).to_dict())
    for a in (1.2, 1.5):
        print(f"aspect < {a} (near-square, orientation unclear):", obj.groupby("source").aspect.apply(lambda x: round((x < a).mean(), 3)).to_dict())

    section("7. Alignment per image (arrangement, FARAD + MiniSAR; 1 = all parallel)")
    for s, g in img[urban].groupby("source"):
        print(s, q(g.alignment))
    for t in (0.6, 0.7, 0.8, 0.9):
        print(f"alignment >= {t}:", img[urban].groupby("source").alignment.apply(lambda x: round((x >= t).mean(), 3)).to_dict())

    section("8. Vehicle-pair intensity difference (sar_comparative, 8-bit scale)")
    pr = pairs_within_images(obj)
    for s, g in pr.groupby("source"):
        print(s, q(g.int_diff))
    for t in (5, 10, 20):
        print(f"diff >= {t}:", pr.groupby("source").int_diff.apply(lambda x: round((x >= t).mean(), 3)).to_dict())

    section("9. Vehicle-pair length ratio (relative_size)")
    for s, g in pr.groupby("source"):
        print(s, q(g.len_ratio))
    for t in (1.1, 1.2, 1.3):
        print(f"ratio >= {t}:", pr.groupby("source").len_ratio.apply(lambda x: round((x >= t).mean(), 3)).to_dict())

    section("10. Nearest neighbour (object_relations, FARAD + MiniSAR)")
    u = obj[obj.source.isin(["FARAD", "MiniSAR"]) & obj.nn_dir.notna()]
    print(pd.crosstab(u.source, u.nn_dir, normalize="index").round(3))
    print("second-nearest < 1.2x nearest (ambiguous nearest):",
          u.groupby("source").apply(lambda g: round((g.nn2_dist_px < 1.2 * g.nn_dist_px).mean(), 3)).to_dict())
    print("direction within 10 deg of sector edge:",
          u.groupby("source").nn_dir_margin.apply(lambda x: round((x < 10).mean(), 3)).to_dict())

    section("11. Reference ambiguity")
    sep = cfg["reference"]["point_min_sep_px"]
    print(f"point: another centre within {sep}px:", obj.groupby("source").nn_dist_px.apply(lambda x: round((x < sep).mean(), 3)).to_dict())
    print("box: another centre inside the hbox:", obj.groupby("source").n_centres_in_hbox.apply(lambda x: round((x > 0).mean(), 3)).to_dict())

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [(img, "n_vehicles", "vehicles per image"), (img, "coverage", "coverage fraction"),
              (img[urban], "alignment", "alignment (urban)"), (pr, "int_diff", "pair intensity diff"),
              (pr, "len_ratio", "pair length ratio"), (obj, "orient_margin", "deg from orientation edge")]
    for a, (df, col, title) in zip(ax.flat, panels):
        for s, g in df.groupby("source"):
            a.hist(g[col].replace(np.inf, np.nan).dropna(), bins=40, alpha=.5, density=True, label=s)
        a.set_title(title); a.legend()
    plt.tight_layout()
    out = os.path.join(rep, "phase1_distributions.png")
    plt.savefig(out, dpi=110)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
