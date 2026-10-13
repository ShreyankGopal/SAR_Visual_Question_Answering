"""generate_questions_sived.py: Phase 2, deterministic QA generation for one SIVED image.

Each gen_* takes (img, o, cfg, rng):
  img  one row of images.csv (pandas Series)
  o    that image's rows of objects.csv (DataFrame)
and returns a list of QA dicts with category, question, answer, answer_key, facts.
answer_key is what build_sived.py balances on.
"""
import itertools

import numpy as np

BOX_NOTE = "(normalized [x1, y1, x2, y2])"


def fb(b):
    return "[" + ", ".join(f"{v:.4f}" for v in b) + "]"


def fp(x, y):
    return f"[{x:.4f}, {y:.4f}]"


def rp(r):
    return [round(r.px, 4), round(r.py, 4)]


def rname(cell):
    return cell.replace("_", " ")


def qa(cat, q, a, key, **facts):
    return dict(category=cat, question=q, answer=a, answer_key=key, facts=facts)


def pick(items, k, rng):
    items = list(items)
    if len(items) <= k:
        return items
    return [items[i] for i in rng.choice(len(items), k, replace=False)]


def point_ok(o, cfg):
    return o[o.nn_dist_px >= cfg["reference"]["point_min_sep_px"]]


def region_split(o, box):
    """Vehicles vs a normalized box: centre inside, hbox overlaps, hbox fully inside."""
    x1, y1, x2, y2 = box
    xe, ye = (x2 + 1e-6 if x2 >= 1 else x2), (y2 + 1e-6 if y2 >= 1 else y2)
    centre = (o.px >= x1) & (o.px < xe) & (o.py >= y1) & (o.py < ye)
    overlap = (o.hx2 > x1) & (o.hx1 < x2) & (o.hy2 > y1) & (o.hy1 < y2)
    return centre, overlap


def random_box(cfg, rng):
    g = cfg["grounding"]
    w, h = rng.uniform(g["min_size"], g["max_size"], 2)
    x1, y1 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
    return [round(x1, 4), round(y1, 4), round(x1 + w, 4), round(y1 + h, 4)]


def ordered_pairs(df, ok, k, rng):
    """Up to k vehicle pairs passing ok(a, b), each in random A/B order."""
    cands = [(a, b) for a, b in itertools.combinations(df.itertuples(), 2) if ok(a, b)]
    return [(a, b) if rng.random() < .5 else (b, a) for a, b in pick(cands, k, rng)]


# ── categories ────────────────────────────────────────────────────────────────

def gen_global_quantitative(img, o, cfg, rng):
    pct = img.coverage * 100
    label = next(l for lo, hi, l in cfg["thresholds"]["coverage_bins"] if lo <= pct < hi)
    return [qa("global_quantitative", "Approximately what percentage of this image is covered by vehicles?",
               f"{label[0].upper()}{label[1:]}.", label, coverage_pct=round(pct, 2))]


def gen_sar_comparative(img, o, cfg, rng):
    t = cfg["thresholds"]["sar_min_intensity_diff"]
    out = []
    for a, b in ordered_pairs(point_ok(o, cfg), lambda a, b: abs(a.mean_int - b.mean_int) >= t,
                              cfg["pipeline"]["max_per_category"], rng):
        win = a if a.mean_int > b.mean_int else b
        out.append(qa("sar_comparative",
                      f"Which vehicle has the stronger average SAR return: the one centred at {fp(a.px, a.py)} "
                      f"or the one centred at {fp(b.px, b.py)}?",
                      f"The one centred at {fp(win.px, win.py)}.", "first" if win is a else "second",
                      points=[rp(a), rp(b)], mean_int=[round(a.mean_int, 1), round(b.mean_int, 1)]))
    return out


def gen_comparative_spatial(img, o, cfg, rng):
    t = cfg["thresholds"]["spatial_min_count_diff"]
    cells = list(cfg["grid_regions"])
    cands = [(a, b) for a, b in itertools.combinations(cells, 2)
             if abs(img[f"count_{a}"] - img[f"count_{b}"]) >= t]
    out = []
    for a, b in pick(cands, cfg["pipeline"]["max_per_category"], rng):
        if rng.random() < .5:
            a, b = b, a
        win = a if img[f"count_{a}"] > img[f"count_{b}"] else b
        out.append(qa("comparative_spatial",
                      f"Are there more vehicles in the {rname(a)} region or the {rname(b)} region?",
                      f"The {rname(win)} region.", "first" if win == a else "second",
                      regions=[a, b], counts=[int(img[f"count_{a}"]), int(img[f"count_{b}"])]))
    return out


def gen_regional_vqa(img, o, cfg, rng):
    out = []
    for cell in cfg["grid_regions"]:
        if img[f"count_{cell}"] > 0:
            ans = "Yes"
        elif img[f"cov_{cell}"] == 0:
            ans = "No"
        else:
            continue          # only part of a vehicle reaches into the cell
        out.append(qa("regional_vqa", f"Are there any vehicles in the {rname(cell)} region?", f"{ans}.", ans,
                      region=cell, count=int(img[f"count_{cell}"])))
    return out


def gen_region_grounding(img, o, cfg, rng):
    g, th = cfg["grounding"], cfg["thresholds"]
    out = []
    # presence in random boxes: Yes = a vehicle centre inside, No = no vehicle touches the box
    for _ in range(g["num_regions"] * 10):
        if len(out) >= g["num_regions"]:
            break
        box = random_box(cfg, rng)
        centre, overlap = region_split(o, box)
        if centre.any():
            ans = "Yes"
        elif not overlap.any():
            ans = "No"
        else:
            continue
        out.append(qa("region_grounding", f"Is there a vehicle in the region {fb(box)} {BOX_NOTE}?", f"{ans}.", ans,
                      subtype="presence", bbox=box))
    if img.source not in g["referring_sources"]:
        return out
    # referring: return the box of a uniquely identified vehicle
    d = np.hypot(o.px - .5, o.py - .5)
    cues = [("closest to the image center", d.sort_values(), "gap"),
            ("closest to the top of the image", o.py.sort_values(), "gap"),
            ("with the longest body", o.length_px.sort_values(ascending=False), "ratio")]
    for cue, s, rule in cues:
        if rule == "ratio" and len(s) < 2:
            continue
        if len(s) >= 2:
            v1, v2 = s.iloc[0], s.iloc[1]
            if rule == "gap" and v2 - v1 < th["referring_min_gap"]:
                continue
            if rule == "ratio" and v1 / v2 < th["size_min_length_ratio"]:
                continue
        t = o.loc[s.index[0]]
        if t.n_centres_in_hbox > 0:
            continue
        box = [round(t.hx1, 4), round(t.hy1, 4), round(t.hx2, 4), round(t.hy2, 4)]
        out.append(qa("region_grounding", f"Give the bounding box of the vehicle {cue} {BOX_NOTE}.", f"{fb(box)}.",
                      "ref", subtype="referring", cue=cue, bbox=box))
    return out


def gen_object_relations(img, o, cfg, rng):
    th = cfg["thresholds"]
    ok = point_ok(o, cfg)
    ok = ok[(ok.nn_id >= 0) & (ok.nn2_dist_px >= th["relation_min_nn2_ratio"] * ok.nn_dist_px)
            & (ok.nn_dir_margin >= th["relation_dir_margin_deg"])]
    return [qa("object_relations", f"In which direction is the nearest vehicle from the vehicle centred at {fp(r.px, r.py)}?",
               f"The nearest vehicle is {r.nn_dir}.", r.nn_dir, point=rp(r), nn_dist_px=round(r.nn_dist_px, 1))
            for r in ok.itertuples()]


def gen_orientation(img, o, cfg, rng):
    th = cfg["thresholds"]
    ok = point_ok(o, cfg)
    ok = ok[(ok.orient_margin >= th["orient_margin_deg"]) & (ok.aspect >= th["orient_min_aspect"])]
    return [qa("orientation", f"What is the orientation of the vehicle centred at {fp(r.px, r.py)}?",
               f"{r.orient[0].upper()}{r.orient[1:]}.", r.orient, point=rp(r), angle=round(r.angle, 1))
            for r in ok.itertuples()]


def gen_relative_size(img, o, cfg, rng):
    t = cfg["thresholds"]["size_min_length_ratio"]
    out = []
    for a, b in ordered_pairs(point_ok(o, cfg), lambda a, b: max(a.length_px, b.length_px) / min(a.length_px, b.length_px) >= t,
                              cfg["pipeline"]["max_per_category"], rng):
        win = a if a.length_px > b.length_px else b
        out.append(qa("relative_size",
                      f"Which vehicle is longer: the one centred at {fp(a.px, a.py)} or the one centred at {fp(b.px, b.py)}?",
                      f"The one centred at {fp(win.px, win.py)}.", "first" if win is a else "second",
                      points=[rp(a), rp(b)], length_m=[round(a.length_m, 2), round(b.length_m, 2)]))
    return out


def gen_arrangement(img, o, cfg, rng):
    th = cfg["thresholds"]
    if img.n_vehicles < th["arrangement_min_vehicles"]:
        return []
    if img.alignment >= th["arrangement_yes_min"]:
        ans = "Yes"
    elif img.alignment <= th["arrangement_no_max"]:
        ans = "No"
    else:
        return []
    return [qa("arrangement", "Are the vehicles in this image mostly aligned in the same direction?", f"{ans}.", ans,
               alignment=round(img.alignment, 3), n_vehicles=int(img.n_vehicles))]


def gen_object_counting(img, o, cfg, rng):
    n = int(img.n_vehicles)
    out = [qa("object_counting", "How many vehicles are in this image?", f"{n}.", "zero" if n == 0 else "nonzero",
              subtype="image", count=n)]
    # grid cells and random boxes, only where no vehicle straddles the border
    regions = [(f"the {rname(c)} region", c, b) for c, b in cfg["grid_regions"].items()]
    regions += [(f"the region {fb(b)} {BOX_NOTE}", None, b) for b in (random_box(cfg, rng) for _ in range(3))]
    for text, cell, box in regions:
        centre, overlap = region_split(o, box)
        if (overlap & ~centre).any():
            continue
        c = int(centre.sum())
        out.append(qa("object_counting", f"How many vehicles are in {text}?", f"{c}.", "zero" if c == 0 else "nonzero",
                      subtype="cell" if cell else "box", region=cell, bbox=box, count=c))
    return out


GENERATORS = {name[4:]: f for name, f in globals().items() if name.startswith("gen_")}


def generate_candidates(img, o, cfg, rng):
    """All categories allowed for this image's sensor, at most max_per_category each."""
    k = cfg["pipeline"]["max_per_category"]
    out = []
    for cat in cfg["categories_per_source"][img.source]:
        out += pick(GENERATORS[cat](img, o, cfg, rng), k, rng)
    return out