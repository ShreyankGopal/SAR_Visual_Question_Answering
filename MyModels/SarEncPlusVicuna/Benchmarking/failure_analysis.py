#!/usr/bin/env python3
"""
failure_analysis.py -- where, how and why a model fails, from run_benchmark.py
prediction files (predictions_step*.jsonl). CPU only, no model.

The FIRST run is the reference, the LAST run is the one analysed (focus), e.g.
    python failure_analysis.py "Full Model=a.jsonl" "SE-MaRS=b.jsonl" \
        --split val --config ../train_config.yaml --out failure_analysis/

Writes to --out:
  by_category.csv   EM / EM-1w / R-L per dataset x question category for every run,
                    plus paired difference focus - reference with 95% CI
  error_types.csv   every wrong answer of every run sorted into one error type
  confusions.csv    most common (ground truth -> prediction) mistakes, single-word answers
  yesno.csv         yes/no questions: how often the model says yes vs how often yes is right
  counting.csv      numeric answers: distribution of the error size
  gsd.csv, gsd.png  accuracy vs image GSD per dataset (needs --config; GSD from gsd_m)
  examples.csv      sampled cases: regressions (reference right, focus wrong), both wrong,
                    fixed (reference wrong, focus right); with one run: focus wrong
  examples_<dataset>_<case>.png   image + question + ground truth + every run's answer
                    (needs --config for image paths, and matplotlib)
"""
import argparse
import csv
import json
import os
import random
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import ALIASES, mean_ci, normalise, paired_diff  # noqa: E402

NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
ERROR_TYPES = ["empty", "box", "yes_no_flip", "count_off_by_1", "count_off_by_2", "count_off_by_3plus",
               "answer_buried_in_longer_reply", "near_miss", "wrong_answer"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(path: str, split: str) -> dict:
    """predictions file -> {key: row}; key = (dataset, id, question), same as score.py."""
    rows = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (split != "all" and r.get("split") != split) or r.get("error") or r.get("exact_match") is None:
                continue
            r["dataset"] = ALIASES.get(r["dataset"], r["dataset"])
            r["gt_n"], r["pred_n"] = normalise(r["ground_truth"]), normalise(r["prediction"] or "")
            r["single"] = len(r["gt_n"].split()) == 1
            rows[(r["dataset"], r["id"], r["question"])] = r
    return rows


def load_config_records(config_path: str) -> dict:
    """
    {(dataset, image): (gsd_m or None, absolute image path)} from the val/test files
    in config['data'] (dataset names follow run_benchmark.discover_eval_datasets).
    """
    import yaml
    with open(config_path) as f:
        c_data = yaml.safe_load(f)["data"]
    out, n = {}, 1
    while True:
        sfx = "" if n == 1 else f"_{n}"
        files = [c_data[k] for k in (f"val_jsonl{sfx}", f"test_jsonl{sfx}") if c_data.get(k)]
        if not files:
            break
        name = c_data.get(f"dataset_name{sfx}") or Path(files[0]).parent.parent.name or Path(files[0]).stem
        name, root = ALIASES.get(name, name), c_data[f"data_root{sfx}"]
        for p in files:
            if not os.path.exists(p):
                print(f"  (config file not found, skipped: {p})")
                continue
            with open(p) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        gsd = rec.get("ground_truth_facts", {}).get("gsd_m", rec.get("gsd_m"))
                        out[(name, rec.get("image", ""))] = (gsd, os.path.join(root, rec.get("image", "")))
        n += 1
    return out


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

def as_number(text: str):
    if text.isdigit():
        return int(text)
    return NUMBERS.get(text)


def error_type(r: dict) -> str:
    """One label for a wrong answer (checked in this order)."""
    gt, pred = r["gt_n"], r["pred_n"]
    if not pred:
        return "empty"
    if r.get("iou") is not None:
        return "box"
    if gt in ("yes", "no") and pred in ("yes", "no"):
        return "yes_no_flip"
    g, p = as_number(gt), as_number(pred)
    if g is not None and p is not None:
        d = abs(g - p)
        return "count_off_by_1" if d == 1 else "count_off_by_2" if d == 2 else "count_off_by_3plus"
    if f" {gt} " in f" {pred} ":
        return "answer_buried_in_longer_reply"
    if (r.get("rougeL") or 0) >= 0.5:
        return "near_miss"
    return "wrong_answer"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def score_rows(rows):
    """score.py-style rows (key + metrics) for mean_ci / paired_diff."""
    return [{"key": k, "em": r["exact_match"], "em_1w": r["exact_match"] if r["single"] else None,
             "rougeL": r["rougeL"]} for k, r in rows]


def by_category(runs, ref, focus, n_boot, rng):
    groups = defaultdict(lambda: defaultdict(list))   # (ds, cat) -> label -> [(key, row)]
    for label, rows in runs.items():
        for k, r in rows.items():
            groups[(r["dataset"], r["category"])][label].append((k, r))
            groups[(r["dataset"], "ALL")][label].append((k, r))
    out = []
    for (ds, cat) in sorted(groups):
        for label, items in groups[(ds, cat)].items():
            s = score_rows(items)
            row = {"dataset": ds, "category": cat, "run": label, "n": len(s),
                   "n_single_word": sum(x["em_1w"] is not None for x in s)}
            for m in ("em", "em_1w", "rougeL"):
                row[m] = mean_ci([x[m] for x in s], 1, rng)[0]
            if label == focus and focus != ref and ref in groups[(ds, cat)]:
                ref_s = score_rows(groups[(ds, cat)][ref])
                for m in ("em", "em_1w"):
                    d, lo, hi, _ = paired_diff(s, ref_s, m, n_boot, rng)
                    if d is not None:
                        row[f"diff_{m}"], row[f"diff_{m}_lo"], row[f"diff_{m}_hi"] = d, lo, hi
                        row[f"diff_{m}_sig"] = "*" if lo > 0 or hi < 0 else ""
            out.append(row)
    return out


def error_tables(runs):
    types, confusions, yesno, counting = [], [], [], []
    for label, rows in runs.items():
        per = defaultdict(list)
        for r in rows.values():
            per[(r["dataset"], r["category"])].append(r)
        for (ds, cat), rs in sorted(per.items()):
            wrong = [r for r in rs if r["exact_match"] < 1]
            c = Counter(error_type(r) for r in wrong)
            for t in ERROR_TYPES:
                if c[t]:
                    types.append({"run": label, "dataset": ds, "category": cat, "error_type": t,
                                  "count": c[t], "share_of_wrong": c[t] / len(wrong), "share_of_all": c[t] / len(rs)})
            conf = Counter((r["gt_n"], r["pred_n"]) for r in wrong if r["single"] and r["pred_n"])
            for (g, p), k in conf.most_common(10):
                confusions.append({"run": label, "dataset": ds, "category": cat,
                                   "ground_truth": g, "prediction": p, "count": k})
            yn = [r for r in rs if r["gt_n"] in ("yes", "no")]
            if yn:
                yesno.append({"run": label, "dataset": ds, "category": cat, "n": len(yn),
                              "gt_yes_rate": np.mean([r["gt_n"] == "yes" for r in yn]),
                              "pred_yes_rate": np.mean([r["pred_n"] == "yes" for r in yn]),
                              "pred_no_rate": np.mean([r["pred_n"] == "no" for r in yn]),
                              "accuracy": np.mean([r["exact_match"] for r in yn])})
            nums = [(as_number(r["gt_n"]), as_number(r["pred_n"])) for r in rs]
            nums = [(g, p) for g, p in nums if g is not None]
            if nums:
                errs = [p - g for g, p in nums if p is not None]
                counting.append({"run": label, "dataset": ds, "category": cat, "n": len(nums),
                                 "exact": np.mean([g == p for g, p in nums]),
                                 "non_numeric_reply": np.mean([p is None for _, p in nums]),
                                 "off_by_1": np.mean([abs(e) == 1 for e in errs]) if errs else None,
                                 "off_by_2plus": np.mean([abs(e) >= 2 for e in errs]) if errs else None,
                                 "mean_signed_error": np.mean(errs) if errs else None,
                                 "under_count_rate": np.mean([e < 0 for e in errs]) if errs else None})
    return types, confusions, yesno, counting


def gsd_bins(values):
    """Each value's bin label: distinct GSDs (2 significant digits) if <= 8 of them, else 5 quantile bins."""
    rounded = [float(f"{v:.2g}") for v in values]
    if len(set(rounded)) <= 8:
        return [f"{v:g}" for v in rounded]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    return [f"{edges[i]:.3g}-{edges[i + 1]:.3g}" for i in idx]


def gsd_table(runs, records):
    out, coverage = [], {}
    for label, rows in runs.items():
        per_ds = defaultdict(list)
        for r in rows.values():
            per_ds[r["dataset"]].append(r)
        for ds, rs in sorted(per_ds.items()):
            with_gsd = [(records[(ds, r["image"])][0], r) for r in rs
                        if records.get((ds, r["image"]), (None,))[0] is not None]
            coverage[(label, ds)] = (len(with_gsd), len(rs))
            if not with_gsd:
                continue
            bins = gsd_bins([float(g) for g, _ in with_gsd])
            grouped = defaultdict(list)
            for b, (g, r) in zip(bins, with_gsd):
                grouped[b].append((float(g), r))
            for b, items in sorted(grouped.items(), key=lambda kv: np.mean([g for g, _ in kv[1]])):
                rs_b = [r for _, r in items]
                single = [r["exact_match"] for r in rs_b if r["single"]]
                out.append({"run": label, "dataset": ds, "gsd_bin_m": b,
                            "gsd_mean_m": np.mean([g for g, _ in items]), "n": len(rs_b),
                            "em": np.mean([r["exact_match"] for r in rs_b]),
                            "em_1w": np.mean(single) if single else None})
    return out, coverage


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

def pick_examples(runs, ref, focus, per_cat, seed):
    rng = random.Random(seed)
    cases = defaultdict(list)   # (ds, category, case) -> [key]
    for k, r in runs[focus].items():
        right_f = r["exact_match"] >= 1
        if ref == focus:
            if not right_f:
                cases[(r["dataset"], r["category"], "wrong")].append(k)
            continue
        if k not in runs[ref]:
            continue
        right_r = runs[ref][k]["exact_match"] >= 1
        case = ("regression" if right_r and not right_f else "both_wrong" if not right_r and not right_f
                else "fixed" if not right_r and right_f else None)
        if case:
            cases[(r["dataset"], r["category"], case)].append(k)
    picked = []
    for (ds, cat, case), keys in sorted(cases.items()):
        for k in rng.sample(keys, min(per_cat, len(keys))):
            picked.append((ds, cat, case, k))
    return picked


def display_image(path):
    """SAR image -> 2D float array stretched to [0, 1] (2-98 percentile) for plotting.
    Same reading as dataset.load_sar_array (TIFF via tifffile, else PIL), without torch."""
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        img = np.asarray(tifffile.imread(path), dtype=float)
    else:
        from PIL import Image
        img = np.asarray(Image.open(path), dtype=float)
    if img.ndim == 3:
        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
    lo, hi = np.percentile(img, [2, 98])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def plot_examples(picked, runs, records, out_dir, per_fig):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -> example figures skipped (examples.csv still written)")
        return
    groups = defaultdict(lambda: defaultdict(list))
    for ds, cat, case, k in picked:
        groups[(ds, case)][cat].append(k)
    focus = list(runs)[-1]
    for (ds, case), by_cat in sorted(groups.items()):
        # take one example per category in turn, so a figure covers as many categories as possible
        queues = [[(cat, k) for k in ks] for cat, ks in sorted(by_cat.items())]
        items = []
        while len(items) < per_fig and any(queues):
            for q in queues:
                if q and len(items) < per_fig:
                    items.append(q.pop(0))
        cols = min(4, len(items))
        rows_n = (len(items) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 6.0 * rows_n), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (cat, k) in zip(axes.flat, items):
            r = runs[focus][k]
            path = records.get((ds, r["image"]), (None, None))[1]
            try:
                ax.imshow(display_image(path), cmap="gray")
            except Exception as e:   # missing file etc.: keep the text panel
                ax.text(0.5, 0.5, f"image not loaded\n{type(e).__name__}", ha="center", transform=ax.transAxes)
            lines = [f"[{cat}]", "Q: " + r["question"], "GT: " + r["ground_truth"]]
            lines += [f"{label}: {runs[label][k]['prediction'] if k in runs[label] else '--'}" for label in runs]
            text = "\n".join(textwrap.fill(l, 48, max_lines=3, placeholder=" ...") for l in lines)
            ax.set_title(text, fontsize=7, loc="left")
        fig.suptitle(f"{ds}: {case.replace('_', ' ')}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"examples_{ds}_{case}.png", dpi=150)
        plt.close(fig)


def plot_gsd(gsd_rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -> gsd.png skipped (gsd.csv still written)")
        return
    datasets = sorted({r["dataset"] for r in gsd_rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for label in dict.fromkeys(r["run"] for r in gsd_rows):
            pts = [r for r in gsd_rows if r["dataset"] == ds and r["run"] == label and r["em_1w"] is not None]
            ax.plot([r["gsd_mean_m"] for r in pts], [r["em_1w"] for r in pts], "o-", label=label)
        ax.set_title(ds)
        ax.set_xlabel("GSD (m)")
        ax.set_ylabel("EM (single-word)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "gsd.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Failure analysis of run_benchmark prediction files")
    ap.add_argument("runs", nargs="+", help='"Label=path.jsonl"; first = reference, last = analysed')
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--config", help="train/benchmark config whose data: section lists the val/test jsonl "
                                     "and data_root (for GSD and images)")
    ap.add_argument("--out", default="failure_analysis")
    ap.add_argument("--per-category", type=int, default=10, help="examples sampled per dataset/category/case")
    ap.add_argument("--per-figure", type=int, default=8, help="examples per PNG")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs[label] = load_run(path, args.split)
        if not runs[label]:
            sys.exit(f"{label}: no rows for split '{args.split}' in {path}")
        print(f"{label}: {len(runs[label])} questions")
    ref, focus = list(runs)[0], list(runs)[-1]
    rng = np.random.default_rng(args.seed)

    cat_rows = by_category(runs, ref, focus, args.boot, rng)
    write_csv(out / "by_category.csv", cat_rows,
              ["dataset", "category", "run", "n", "n_single_word", "em", "em_1w", "rougeL",
               "diff_em", "diff_em_lo", "diff_em_hi", "diff_em_sig",
               "diff_em_1w", "diff_em_1w_lo", "diff_em_1w_hi", "diff_em_1w_sig"])

    types, confusions, yesno, counting = error_tables(runs)
    write_csv(out / "error_types.csv", types,
              ["run", "dataset", "category", "error_type", "count", "share_of_wrong", "share_of_all"])
    write_csv(out / "confusions.csv", confusions, ["run", "dataset", "category", "ground_truth", "prediction", "count"])
    write_csv(out / "yesno.csv", yesno,
              ["run", "dataset", "category", "n", "gt_yes_rate", "pred_yes_rate", "pred_no_rate", "accuracy"])
    write_csv(out / "counting.csv", counting,
              ["run", "dataset", "category", "n", "exact", "non_numeric_reply", "off_by_1", "off_by_2plus",
               "mean_signed_error", "under_count_rate"])

    records = load_config_records(args.config) if args.config else {}
    if records:
        gsd_rows, coverage = gsd_table(runs, records)
        for (label, ds), (k, n) in sorted(coverage.items()):
            print(f"  GSD known for {k}/{n} questions  [{label}, {ds}]")
        write_csv(out / "gsd.csv", gsd_rows, ["run", "dataset", "gsd_bin_m", "gsd_mean_m", "n", "em", "em_1w"])
        if gsd_rows:
            plot_gsd(gsd_rows, out)
    else:
        print("  no --config: GSD analysis and example images skipped")

    picked = pick_examples(runs, ref, focus, args.per_category, args.seed)
    write_csv(out / "examples.csv",
              [{"dataset": ds, "category": cat, "case": case, "image": runs[focus][k]["image"],
                "question": runs[focus][k]["question"], "ground_truth": runs[focus][k]["ground_truth"],
                **{f"pred_{label}": runs[label][k]["prediction"] if k in runs[label] else "" for label in runs}}
               for ds, cat, case, k in picked],
              ["dataset", "category", "case", "image", "question", "ground_truth"] + [f"pred_{l}" for l in runs])
    if records:
        plot_examples(picked, runs, records, out, args.per_figure)

    # Short console summary: where the analysed run loses most vs the reference
    if ref != focus:
        losses = sorted((r for r in cat_rows if r["run"] == focus and r["category"] != "ALL"
                         and r.get("diff_em_sig") == "*" and r["diff_em"] < 0), key=lambda r: r["diff_em"])
        print(f"\nSignificant EM losses of '{focus}' vs '{ref}' (dataset / category / diff):")
        for r in losses[:15] or []:
            print(f"  {r['dataset']:14s} {r['category']:28s} {r['diff_em'] * 100:+.2f}  (n={r['n']})")
        if not losses:
            print("  none")
    print(f"\nWrote results to {out}/")


if __name__ == "__main__":
    main()
