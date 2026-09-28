"""generators.py: deterministic QA generators shared by all datasets.

Each gen_* takes (img, o, cfg, rng):
  img  one row of images.csv (pandas Series)
  o    that image's rows of objects.csv (DataFrame)
and returns a list of QA dicts: category, question, answer, answer_key, facts.
answer_key is what balance.py balances on.

Class names come from cfg["classes"] ({Name: {singular, plural}}), so a single-class
dataset (SIVED: vehicle) and a multi-class one (OGSOD: bridge, harbour, tank) share code.
For a single-class dataset the random draws happen in exactly the SIVED v1 order.
"""
import itertools

import numpy as np

from .io import slug, load_gray

BOX_NOTE = "(normalized [x1, y1, x2, y2])"


# ── text helpers ──────────────────────────────────────────────────────────────

def fb(b):
    return "[" + ", ".join(f"{v:.4f}" for v in b) + "]"


def fp(x, y):
    return f"[{x:.4f}, {y:.4f}]"


def rp(r):
    return [round(r.px, 4), round(r.py, 4)]


def rname(cell):
    return cell.replace("_", " ")


def cap(s):
    return f"{s[0].upper()}{s[1:]}"


def classes(cfg):
    return list(cfg["classes"])


def sg(cfg, c):
    return cfg["classes"][c]["singular"]


def pl(cfg, c):
    return cfg["classes"][c]["plural"]


def present_classes(img, cfg):
    """Multi-class datasets: only classes in this image (avoids floods of trivial zero / No answers)."""
    cl = classes(cfg)
    return cl if len(cl) == 1 else [c for c in cl if img[f"n__{slug(c)}"] > 0]


def join(words):
    words = list(words)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def qa(cat, q, a, key, **facts):
    return dict(category=cat, question=q, answer=a, answer_key=key, facts=facts)


def bin_label(value, bins):
    return next((l for lo, hi, l in bins if lo <= value < hi), None)


# ── sampling helpers ──────────────────────────────────────────────────────────

def pick(items, k, rng):
    items = list(items)
    if len(items) <= k:
        return items
    return [items[i] for i in rng.choice(len(items), k, replace=False)]


def point_ok(o, cfg):
    """Objects usable as point references: no other centre too close, not too small to see."""
    r = cfg["reference"]
    return o[(o.nn_dist_px >= r["point_min_sep_px"]) & (o.length_px >= r.get("min_length_px", 0))]


def region_split(o, box):
    """Objects vs a normalized box: centre inside, hbox overlaps."""
    x1, y1, x2, y2 = box
    xe, ye = (x2 + 1e-6 if x2 >= 1 else x2), (y2 + 1e-6 if y2 >= 1 else y2)
    centre = (o.px >= x1) & (o.px < xe) & (o.py >= y1) & (o.py < ye)
    overlap = (o.hx2 > x1) & (o.hx1 < x2) & (o.hy2 > y1) & (o.hy1 < y2)
    return centre, overlap


def random_box(cfg, rng, min_size=None, max_size=None):
    g = cfg["grounding"]
    w, h = rng.uniform(min_size or g["min_size"], max_size or g["max_size"], 2)
    x1, y1 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
    return [round(x1, 4), round(y1, 4), round(x1 + w, 4), round(y1 + h, 4)]


def box_around(cfg, rng, obj):
    """Random-size box that contains obj's centre (position random around it)."""
    g = cfg["grounding"]
    w, h = rng.uniform(g["min_size"], g["max_size"], 2)
    x1 = rng.uniform(max(0, obj.px - w), min(obj.px, 1 - w))
    y1 = rng.uniform(max(0, obj.py - h), min(obj.py, 1 - h))
    return [round(x1, 4), round(y1, 4), round(x1 + w, 4), round(y1 + h, 4)]


def ordered_pairs(df, ok, k, rng):
    """Up to k object pairs passing ok(a, b), each in random A/B order."""
    cands = [(a, b) for a, b in itertools.combinations(df.itertuples(), 2) if ok(a, b)]
    return [(a, b) if rng.random() < .5 else (b, a) for a, b in pick(cands, k, rng)]


# ── categories ────────────────────────────────────────────────────────────────

def gen_global_quantitative(img, o, cfg, rng):
    th = cfg["thresholds"]
    cl = classes(cfg)
    noun = pl(cfg, cl[0]) if len(cl) == 1 else join(pl(cfg, c) for c in cl)
    pct = img.coverage * 100
    label = bin_label(pct, th["coverage_bins"])
    out = [qa("global_quantitative", f"Approximately what percentage of this image is covered by {noun}?",
              f"{cap(label)}.", label, coverage_pct=round(pct, 2))]
    for c in th.get("coverage_per_class", []):
        if img[f"n__{slug(c)}"] == 0:
            continue
        pct = img[f"coverage__{slug(c)}"] * 100
        label = bin_label(pct, th["coverage_bins"])
        out.append(qa("global_quantitative", f"Approximately what percentage of this image is covered by {pl(cfg, c)}?",
                      f"{cap(label)}.", f"{c}: {label}", cls=c, coverage_pct=round(pct, 2)))
    return out


def gen_sar_comparative(img, o, cfg, rng):
    """Same-class pairs only, so the class itself cannot give the answer away."""
    t = cfg["thresholds"]["sar_min_intensity_diff"]
    out = []
    for a, b in ordered_pairs(point_ok(o, cfg),
                              lambda a, b: a.category == b.category and abs(a.mean_int - b.mean_int) >= t,
                              cfg["pipeline"]["max_per_category"], rng):
        win = a if a.mean_int > b.mean_int else b
        out.append(qa("sar_comparative",
                      f"Which {sg(cfg, a.category)} has the stronger average SAR return: the one centred at {fp(a.px, a.py)} "
                      f"or the one centred at {fp(b.px, b.py)}?",
                      f"The one centred at {fp(win.px, win.py)}.", "first" if win is a else "second",
                      cls=a.category, points=[rp(a), rp(b)], mean_int=[round(a.mean_int, 1), round(b.mean_int, 1)]))
    return out


def gen_comparative_spatial(img, o, cfg, rng):
    t = cfg["thresholds"]["spatial_min_count_diff"]          # a number, or {class: number}
    tc = (lambda c: t[c]) if isinstance(t, dict) else (lambda c: t)
    cells = list(cfg["grid_regions"])
    cnt = lambda c, cell: img[f"count__{slug(c)}__{cell}"]
    cands = [(c, a, b) for c in classes(cfg) for a, b in itertools.combinations(cells, 2)
             if abs(cnt(c, a) - cnt(c, b)) >= tc(c)]
    out = []
    for c, a, b in pick(cands, cfg["pipeline"]["max_per_category"], rng):
        if rng.random() < .5:
            a, b = b, a
        win = a if cnt(c, a) > cnt(c, b) else b
        out.append(qa("comparative_spatial",
                      f"Are there more {pl(cfg, c)} in the {rname(a)} region or the {rname(b)} region?",
                      f"The {rname(win)} region.", "first" if win == a else "second",
                      cls=c, regions=[a, b], counts=[int(cnt(c, a)), int(cnt(c, b))]))
    return out


def gen_regional_vqa(img, o, cfg, rng):
    out = []
    for c in present_classes(img, cfg):
        s = slug(c)
        for cell in cfg["grid_regions"]:
            if img[f"count__{s}__{cell}"] > 0:
                ans = "Yes"
            elif img[f"cov__{s}__{cell}"] == 0:
                ans = "No"
            else:
                continue          # only part of an object reaches into the cell
            out.append(qa("regional_vqa", f"Are there any {pl(cfg, c)} in the {rname(cell)} region?", f"{ans}.", ans,
                          cls=c, region=cell, count=int(img[f"count__{s}__{cell}"])))
    n_types = cfg.get("regional", {}).get("types_per_image", 0)
    if n_types:
        for cell in pick(cfg["grid_regions"], n_types, rng):
            present = [c for c in classes(cfg) if img[f"count__{slug(c)}__{cell}"] > 0]
            partial = [c for c in classes(cfg) if img[f"count__{slug(c)}__{cell}"] == 0 and img[f"cov__{slug(c)}__{cell}"] > 0]
            if partial:
                continue
            ans = join(present) if present else "None"
            out.append(qa("regional_vqa", f"Which object types are in the {rname(cell)} region?", f"{ans}.", "types",
                          region=cell, classes=present))
    return out


def gen_region_grounding(img, o, cfg, rng):
    g, th = cfg["grounding"], cfg["thresholds"]
    cl = classes(cfg)
    out = []
    # presence in random boxes: Yes = an object centre inside, No = no object of that class touches the box
    for _ in range(g["num_regions"] * 10):
        if len(out) >= g["num_regions"]:
            break
        if len(cl) > 1:
            c = cl[int(rng.integers(len(cl)))]
            oc = o[o.category == c]
            box = box_around(cfg, rng, oc.iloc[int(rng.integers(len(oc)))]) if len(oc) and rng.random() < .5 \
                else random_box(cfg, rng)
        else:
            c, p_obj = cl[0], g.get("presence_over_object", 0)
            box = box_around(cfg, rng, o.iloc[int(rng.integers(len(o)))]) if p_obj and len(o) and rng.random() < p_obj \
                else random_box(cfg, rng)
        centre, overlap = region_split(o[o.category == c], box)
        if centre.any():
            ans = "Yes"
        elif not overlap.any():
            ans = "No"
        else:
            continue
        out.append(qa("region_grounding", f"Is there a {sg(cfg, c)} in the region {fb(box)} {BOX_NOTE}?", f"{ans}.", ans,
                      subtype="presence", cls=c, bbox=box))

    mode = g.get("referring")
    if mode == "cues" and img.source in g.get("referring_sources", []):
        noun = sg(cfg, cl[0])
        oo = o[o.length_px >= cfg["reference"].get("min_length_px", 0)]      # too-small objects never referred to
        d = np.hypot(oo.px - .5, oo.py - .5)
        cues = [(f"the {noun} closest to the image center", d.sort_values(), "gap"),
                (f"the {noun} closest to the top of the image", oo.py.sort_values(), "gap"),
                (g.get("longest_cue", f"the {noun} with the longest body"), oo.length_px.sort_values(ascending=False), "ratio")]
        for cue, s, rule in cues:
            if len(s) == 0 or (rule == "ratio" and len(s) < 2):
                continue
            if len(s) >= 2:
                v1, v2 = s.iloc[0], s.iloc[1]
                if rule == "gap" and v2 - v1 < th["referring_min_gap"]:
                    continue
                if rule == "ratio" and v1 / v2 < th["size_min_length_ratio"]:
                    continue
            t = o.loc[s.index[0]]
            if t.n_centres_in_hbox > 0 or (rule == "ratio" and t.truncated):
                continue
            box = [round(t.hx1, 4), round(t.hy1, 4), round(t.hx2, 4), round(t.hy2, 4)]
            out.append(qa("region_grounding", f"Give the bounding box of {cue} {BOX_NOTE}.", f"{fb(box)}.",
                          "ref", subtype="referring", cue=cue, bbox=box))
    elif mode == "unique_class":
        for c in cl:
            oc = o[o.category == c]
            if len(oc) != 1:
                continue
            t = oc.iloc[0]
            box = [round(t.hx1, 4), round(t.hy1, 4), round(t.hx2, 4), round(t.hy2, 4)]
            out.append(qa("region_grounding", f"Give the bounding box of the {sg(cfg, c)} {BOX_NOTE}.", f"{fb(box)}.",
                          "ref", subtype="referring", cls=c, bbox=box))

    # which class is in a random region (not an object's own box, so box size can't give the class away)
    n_cls, n_none, made = g.get("class_in_region", 0), 0, 0
    for _ in range(n_cls * 10):
        if made >= n_cls:
            break
        box = random_box(cfg, rng)
        centre, overlap = region_split(o, box)
        if (overlap & ~centre).any():
            continue
        present = [c for c in cl if (centre & (o.category == c)).any()]
        if not present:
            if n_none >= 1:
                continue
            n_none += 1
        ans = join(present) if present else "None"
        out.append(qa("region_grounding", f"What type of object is in the region {fb(box)} {BOX_NOTE}?", f"{ans}.",
                      "class", subtype="class_in_region", bbox=box, classes=present))
        made += 1
    return out


def gen_object_relations(img, o, cfg, rng):
    th = cfg["thresholds"]
    ok = point_ok(o, cfg)
    ok = ok[(ok.nn_id >= 0) & (ok.nn2_dist_px >= th["relation_min_nn2_ratio"] * ok.nn_dist_px)
            & (ok.nn_dir_margin >= th["relation_dir_margin_deg"])]
    return [qa("object_relations",
               f"In which direction is the nearest {sg(cfg, r.nn_cls)} from the {sg(cfg, r.category)} centred at {fp(r.px, r.py)}?",
               f"The nearest {sg(cfg, r.nn_cls)} is {r.nn_dir}.", r.nn_dir,
               cls=r.category, nn_cls=r.nn_cls, point=rp(r), nn_dist_px=round(r.nn_dist_px, 1))
            for r in ok.itertuples()]


def gen_orientation(img, o, cfg, rng):
    th = cfg["thresholds"]
    ok = point_ok(o, cfg)
    ok = ok[ok.orient.notna() & ~ok.truncated & (ok.orient_margin >= th["orient_margin_deg"]) & (ok.aspect >= th["orient_min_aspect"])]
    return [qa("orientation", f"What is the orientation of the {sg(cfg, r.category)} centred at {fp(r.px, r.py)}?",
               f"{cap(r.orient)}.", r.orient, cls=r.category, point=rp(r), angle=round(r.angle, 1))
            for r in ok.itertuples()]


def gen_relative_size(img, o, cfg, rng):
    t = cfg["thresholds"]["size_min_length_ratio"]
    out = []
    ok = point_ok(o, cfg)
    for a, b in ordered_pairs(ok[~ok.truncated],
                              lambda a, b: a.category == b.category and
                              max(a.length_px, b.length_px) / min(a.length_px, b.length_px) >= t,
                              cfg["pipeline"]["max_per_category"], rng):
        win = a if a.length_px > b.length_px else b
        out.append(qa("relative_size",
                      f"Which {sg(cfg, a.category)} is longer: the one centred at {fp(a.px, a.py)} "
                      f"or the one centred at {fp(b.px, b.py)}?",
                      f"The one centred at {fp(win.px, win.py)}.", "first" if win is a else "second",
                      cls=a.category, points=[rp(a), rp(b)], length_m=[round(a.length_m, 2), round(b.length_m, 2)]))
    return out


def gen_arrangement(img, o, cfg, rng):
    th = cfg["thresholds"]
    if img.n_objects < th["arrangement_min_vehicles"]:
        return []
    if img.alignment >= th["arrangement_yes_min"]:
        ans = "Yes"
    elif img.alignment <= th["arrangement_no_max"]:
        ans = "No"
    else:
        return []
    noun = pl(cfg, classes(cfg)[0])
    return [qa("arrangement", f"Are the {noun} in this image mostly aligned in the same direction?", f"{ans}.", ans,
               alignment=round(img.alignment, 3), n_objects=int(img.n_objects))]


def gen_object_counting(img, o, cfg, rng):
    cl = present_classes(img, cfg)
    out = []
    for c in cl:
        n = int(img[f"n__{slug(c)}"])
        out.append(qa("object_counting", f"How many {pl(cfg, c)} are in this image?", f"{n}.",
                      "zero" if n == 0 else "nonzero", subtype="image", cls=c, count=n))
    # grid cells and random boxes, only where no object of that class straddles the border
    regions = [(f"the {rname(c)} region", c, b) for c, b in cfg["grid_regions"].items()]
    regions += [(f"the region {fb(b)} {BOX_NOTE}", None, b) for b in (random_box(cfg, rng) for _ in range(3))]
    for text, cell, box in regions:
        for c in cl:
            centre, overlap = region_split(o[o.category == c], box)
            if (overlap & ~centre).any():
                continue
            n = int(centre.sum())
            out.append(qa("object_counting", f"How many {pl(cfg, c)} are in {text}?", f"{n}.",
                          "zero" if n == 0 else "nonzero", subtype="cell" if cell else "box",
                          cls=c, region=cell, bbox=box, count=n))
    return out


def gen_global_classification(img, o, cfg, rng):
    cl = classes(cfg)
    out = []
    for c in cl:
        ans = "Yes" if img[f"n__{slug(c)}"] > 0 else "No"
        out.append(qa("global_classification", f"Is there a {sg(cfg, c)} in this image?", f"{ans}.", ans, cls=c))
    present = [c for c in cl if img[f"n__{slug(c)}"] > 0]
    out.append(qa("global_classification", "What types of objects are in this image?", f"{join(present)}.",
                  "types", classes=present))
    return out


def gen_object_classification(img, o, cfg, rng):
    return [qa("object_classification", f"What type of object is centred at {fp(r.px, r.py)}?", f"{r.category}.",
               r.category, point=rp(r))
            for r in point_ok(o, cfg).itertuples()]


def gen_absolute_size(img, o, cfg, rng):
    """Size in metres at native resolution: tank diameter, harbour longest side (per config)."""
    spec, margin = cfg["thresholds"]["size_bins"], cfg["thresholds"]["size_margin_m"]
    out = []
    for r in point_ok(o, cfg).itertuples():
        if r.category not in spec:
            continue
        s = spec[r.category]
        if r.hx1 <= 0 or r.hy1 <= 0 or r.hx2 >= 1 or r.hy2 >= 1 or r.truncated:
            continue                      # clipped at the image edge: size unreliable
        if s["measure"] == "length":      # rotated-box long edge (ships)
            val = r.length_px * img.native_gsd_m
            q = f"Approximately how long is the {sg(cfg, r.category)} centred at {fp(r.px, r.py)}?"
        elif s["measure"] == "diameter":
            val = (r.box_w_px + r.box_h_px) / 2 * img.native_gsd_m
            q = f"Approximately what is the diameter of the {sg(cfg, r.category)} centred at {fp(r.px, r.py)}?"
        else:
            val = max(r.box_w_px, r.box_h_px) * img.native_gsd_m
            q = f"Approximately how long is the longest side of the {sg(cfg, r.category)} centred at {fp(r.px, r.py)}?"
        edges = [lo for lo, _, _ in s["bins"][1:]]
        if any(abs(val - e) < margin for e in edges):
            continue
        label = bin_label(val, s["bins"])
        out.append(qa("absolute_size", q, f"{cap(label)}.", f"{r.category}: {label}",
                      cls=r.category, point=rp(r), size_m=round(val, 1)))
    return out


def gen_sar_observation(img, o, cfg, rng):
    """Multi-class images only: which class has the lowest / highest fraction of bright pixels."""
    t = cfg["thresholds"]["observation_min_diff"]
    frac = o.groupby("category").bright.mean().sort_values()
    if len(frac) < 2:
        return []
    out = []
    if frac.iloc[1] - frac.iloc[0] >= t:
        out.append(qa("sar_observation", "Which object type in this image has the lowest fraction of bright SAR pixels?",
                      f"{frac.index[0]}.", frac.index[0], which="lowest", bright={k: round(v, 3) for k, v in frac.items()}))
    if frac.iloc[-1] - frac.iloc[-2] >= t:
        out.append(qa("sar_observation", "Which object type in this image has the highest fraction of bright SAR pixels?",
                      f"{frac.index[-1]}.", frac.index[-1], which="highest", bright={k: round(v, 3) for k, v in frac.items()}))
    return out


def gen_sar_bbox_variance(img, o, cfg, rng):
    v = cfg["sar_bbox_variance"]
    a = load_gray(cfg, img.image)
    H, W = a.shape
    out = []
    for _ in range(v["num_regions"]):
        box = random_box(cfg, rng, v["min_size"], v["max_size"])
        x1, y1, x2, y2 = box
        var = float(a[round(y1 * H):round(y2 * H), round(x1 * W):round(x2 * W)].var())
        label = bin_label(var, v["bins"])
        out.append(qa("sar_bbox_variance", f"How heterogeneous is the SAR response in the region {fb(box)} {BOX_NOTE}?",
                      f"{label} heterogeneity.", label, bbox=box, variance=round(var, 1)))
    return out


GENERATORS = {name[4:]: f for name, f in globals().items() if name.startswith("gen_")}


def generate_candidates(img, o, cfg, rng):
    """All categories allowed for this image's source, at most max_per_category each."""
    k = cfg["pipeline"]["max_per_category"]
    out = []
    for cat in cfg["categories_per_source"][img.source]:
        out += pick(GENERATORS[cat](img, o, cfg, rng), k, rng)
    return out