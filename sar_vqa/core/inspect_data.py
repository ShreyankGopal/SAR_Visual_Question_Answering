"""inspect_data.py: distributions behind every threshold, for any dataset.

Prints a text report and saves reports/distributions.png. Sections that need
rotated boxes (orientation, alignment) are skipped when the dataset has none.
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import out_path, slug, load_gray

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def q(s):
    return s.quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_dict()


def share(s, cond):
    return round(float(cond(s).mean()), 3)


def run(cfg):
    obj = pd.read_csv(out_path(cfg, "facts_dir", "objects.csv"))
    img = pd.read_csv(out_path(cfg, "facts_dir", "images.csv"))
    cells, classes = list(cfg["grid_regions"]), list(cfg["classes"])
    has_rbox = obj.angle.notna().any()

    section("1. Images and objects per source / split / class")
    print(img.groupby(["source", "split"]).size().unstack(fill_value=0))
    print(pd.crosstab([obj.source, obj.category], obj.split))

    section("2. Objects per image, per class (counting)")
    for c in classes:
        print(c, q(img[f"n__{slug(c)}"]))
    if len(classes) > 1:
        print("classes per image:", (img[[f"n__{slug(c)}" for c in classes]] > 0).sum(1).value_counts().to_dict())

    section("3. Coverage % (global_quantitative)")
    print("all:", q(img.coverage * 100))
    bins = [lo for lo, _, _ in cfg["thresholds"]["coverage_bins"]] + [101]
    print(pd.crosstab(img.source, pd.cut(img.coverage * 100, bins, right=False), normalize="index").round(3))
    for c in classes:
        v = img.loc[img[f"n__{slug(c)}"] > 0, f"coverage__{slug(c)}"] * 100
        print(f"{c} where present:", q(v))

    section("4. Objects per grid cell, per class (regional_vqa, counting per cell)")
    for c in classes:
        long = img.melt(id_vars="source", value_vars=[f"count__{slug(c)}__{x}" for x in cells], value_name="n")
        print(c); print(pd.crosstab(long.source, long.n.clip(upper=10), normalize="index").round(3))

    section("5. Cell-pair count differences per class (comparative_spatial)")
    for c in classes:
        d = pd.concat([(img[f"count__{slug(c)}__{a}"] - img[f"count__{slug(c)}__{b}"]).abs()
                       for a, b in itertools.combinations(cells, 2)])
        print(c, d.clip(upper=5).value_counts(normalize=True).sort_index().round(3).to_dict())

    section("6. Same-class pair intensity difference (sar_comparative, 8-bit)")
    rows = []
    for (stem, cls), g in obj.groupby(["stem", "category"]):
        i, j = np.triu_indices(len(g), 1)
        v, L = g.mean_int.to_numpy(), g.length_px.to_numpy()
        rows.append(pd.DataFrame(dict(source=g.source.iloc[0], cls=cls, int_diff=np.abs(v[i] - v[j]),
                                      len_ratio=np.maximum(L[i], L[j]) / np.minimum(L[i], L[j]),
                                      same_winner=(v[i] > v[j]) == (L[i] > L[j]))))
    pr = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["source", "cls", "int_diff", "len_ratio", "same_winner"])
    for (s, c), g in pr.groupby(["source", "cls"]):
        print(s, c, q(g.int_diff), "| >=15:", share(g.int_diff, lambda x: x >= 15), ">=20:", share(g.int_diff, lambda x: x >= 20))
    for t in (15, 20, 30):
        g = pr[(pr.int_diff >= t) & (pr.len_ratio > 1.0)]
        print(f"pairs with diff >= {t}: brighter object is also the longer one in",
              g.groupby("source").same_winner.mean().round(3).to_dict(), "(high = sar_comparative repeats relative_size)")

    section("7. Object size in metres at native GSD (absolute_size, relative_size)")
    obj = obj.merge(img[["stem", "native_gsd_m"]], on="stem")
    obj["diam_m"] = (obj.box_w_px + obj.box_h_px) / 2 * obj.native_gsd_m
    obj["long_m"] = obj[["box_w_px", "box_h_px"]].max(1) * obj.native_gsd_m
    for c in classes:
        o = obj[obj.category == c]
        print(c, "diameter:", q(o.diam_m), "\n   longest side:", q(o.long_m))
    if has_rbox:
        ok = obj[~obj.truncated & (obj.hx1 > 0) & (obj.hy1 > 0) & (obj.hx2 < 1) & (obj.hy2 < 1)]
        print("rotated-box length (m), non-truncated, by native GSD:")
        print(ok.assign(len_m=ok.length_px * ok.native_gsd_m).groupby("native_gsd_m").len_m
              .describe(percentiles=[.1, .25, .5, .75, .9]).round(1))
        print("truncated objects:", round(float(obj.truncated.mean()), 3))
    for (s, c), g in pr.groupby(["source", "cls"]):
        print(s, c, "length ratio >= 1.2:", share(g.len_ratio, lambda x: x >= 1.2))

    if has_rbox:
        section("8. Orientation and alignment (rotated boxes only)")
        print(pd.crosstab(obj.source, obj.orient, normalize="index").round(3))
        print("within 5 deg of a bin edge:", obj.groupby("source").orient_margin.apply(lambda x: share(x, lambda y: y < 5)).to_dict())
        for s, g in img.groupby("source"):
            print(s, "alignment", q(g.alignment))

    section("9. Nearest neighbour and reference ambiguity (object_relations, point / box references)")
    u = obj[obj.nn_dir.notna()]
    print(pd.crosstab(u.source, u.nn_dir, normalize="index").round(3))
    print("second-nearest < 1.2x nearest:", u.groupby("source").apply(lambda g: share(g, lambda x: x.nn2_dist_px < 1.2 * x.nn_dist_px), include_groups=False).to_dict())
    sep = cfg["reference"]["point_min_sep_px"]
    print(f"point: another centre within {sep}px:", obj.groupby("category").nn_dist_px.apply(lambda x: share(x, lambda y: y < sep)).to_dict())
    print("box: another centre inside the hbox:", obj.groupby("category").n_centres_in_hbox.apply(lambda x: share(x, lambda y: y > 0)).to_dict())

    section("10. Bright-pixel fraction per class (sar_observation)")
    print(obj.groupby("category").bright.describe().round(3))

    section("11. Random-region variance (sar_bbox_variance)")
    v = cfg.get("sar_bbox_variance", {"min_size": 0.10, "max_size": 0.35})
    rng = np.random.default_rng(0)
    samp = img.sample(min(1500, len(img)), random_state=0)
    vr = []
    for r in samp.itertuples():
        a = load_gray(cfg, r.image)
        H, W = a.shape
        for _ in range(2):
            w, h = rng.uniform(v["min_size"], v["max_size"], 2)
            x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
            vr.append(dict(source=r.source, var=float(a[round(y * H):round((y + h) * H), round(x * W):round((x + w) * W)].var())))
    vr = pd.DataFrame(vr)
    cuts = vr["var"].quantile([1 / 3, 2 / 3]).round(1).tolist()
    print("all:", q(vr["var"]), "\ntertile cut points (for the config bins):", cuts)
    lab = pd.cut(vr["var"], [-1, *cuts, np.inf], labels=["Low", "Medium", "High"])
    print(pd.crosstab(vr.source, lab, normalize="index").round(3))

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a0 = ax.flat[0]
    for c in classes:
        n = img[f"n__{slug(c)}"]
        a0.hist(n[n > 0], bins=np.arange(0, n.max() + 2) - .5, alpha=.5, label=c)
    a0.set_title("objects per image (images containing the class)"); a0.legend()
    panels = [(img, "coverage", "source", "coverage fraction"),
              (pr, "int_diff", "cls", "same-class pair intensity diff"),
              (obj, "diam_m", "category", "size: mean box side (m)"),
              (obj, "mean_int", "category", "mean intensity"),
              (obj, "bright", "category", "bright-pixel fraction")]
    for a, (df, col, by, title) in zip(list(ax.flat)[1:], panels):
        for s, g in df.groupby(by):
            a.hist(g[col].replace(np.inf, np.nan).dropna(), bins=40, alpha=.5, density=True, label=s)
        a.set_title(title); a.legend()
    plt.tight_layout()
    path = out_path(cfg, "reports_dir", "distributions.png")
    plt.savefig(path, dpi=110)
    print(f"\nSaved {path}")
