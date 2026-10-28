#!/usr/bin/env python3
"""
score.py -- one scoring protocol for every run.

Re-scores the per-question prediction files written by run_benchmark.py
(predictions_step*.jsonl) with ONE split and ONE set of metric definitions,
and adds 95% bootstrap confidence intervals. CPU only; no model needed (the
per-question metrics are already in each row).

Per dataset, and overall (micro = all questions pooled, macro = mean over datasets):
    EM        exact match over ALL questions
    EM-1w     exact match over single-word ground truths only (older tables' EM)
    B-1, B-4  BLEU, R-L  ROUGE-L,  mIoU / Acc@0.5 for box answers

Usage:
    python score.py "Full Model=runA/predictions_step115380_x.jsonl" \
                    "SE-MaRS=runB/predictions_step66080_y.jsonl" --split val
    # the FIRST run is the reference: every other run also gets paired
    # differences vs it (same questions), with 95% CIs
    python score.py ... --split val --csv scores.csv --latex
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict

import numpy as np

# The same dataset under different labels across servers / configs.
ALIASES = {"DataGen": "openearthmap", "data": "openearthmap", "oem": "openearthmap"}
DATASET_ORDER = ["openearthmap", "hrsid", "ogsod", "sar_aircraft", "sived"]
METRICS = ["em", "em_1w", "bleu1", "bleu4", "rougeL"]           # bootstrap / paired on these
CI_METRICS = ["em", "em_1w", "rougeL"]


def normalise(text: str) -> str:
    """Same normalisation as run_benchmark.py (lowercase, strip punctuation)."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def load_rows(path: str, split: str) -> tuple:
    """Rows of one predictions file -> ({dataset: [row, ...]}, n_errors)."""
    by_ds, errors = defaultdict(list), 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if split != "all" and r.get("split") != split:
                continue
            if r.get("error") or r.get("exact_match") is None:
                errors += 1
                continue
            ds = ALIASES.get(r["dataset"], r["dataset"])
            single = len(normalise(r["ground_truth"]).split()) == 1
            by_ds[ds].append({
                "key": (ds, r["id"], r["question"]),
                "em": r["exact_match"], "em_1w": r["exact_match"] if single else None,
                "bleu1": r["bleu1"], "bleu4": r["bleu4"], "rougeL": r["rougeL"], "iou": r.get("iou"),
            })
    return by_ds, errors


def mean_ci(values, n_boot: int, rng):
    """Mean and 95% bootstrap CI of a list of numbers (None entries dropped)."""
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None, None
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    return v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def summarise(rows, n_boot: int, rng) -> dict:
    out = {"n": len(rows)}
    for m in METRICS:
        vals = [r[m] for r in rows]
        if m in CI_METRICS:
            out[m], out[m + "_lo"], out[m + "_hi"] = mean_ci(vals, n_boot, rng)
        else:
            v = [x for x in vals if x is not None]
            out[m] = float(np.mean(v)) if v else None
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    out["miou"] = float(np.mean(ious)) if ious else None
    out["acc50"] = float(np.mean([i >= 0.5 for i in ious])) if ious else None
    out["n_box"] = len(ious)
    return out


def paired_diff(rows_a, rows_ref, metric: str, n_boot: int, rng):
    """Mean difference (a - ref) over questions present in both runs, with 95% CI."""
    ref = {r["key"]: r[metric] for r in rows_ref}
    d = [r[metric] - ref[r["key"]] for r in rows_a
         if r["key"] in ref and r[metric] is not None and ref[r["key"]] is not None]
    if not d:
        return None, None, None, 0
    mean, lo, hi = mean_ci(d, n_boot, rng)
    return mean, lo, hi, len(d)


def fmt(x, lo=None, hi=None, pct=True):
    if x is None:
        return "--"
    s = 100 if pct else 1
    return f"{x * s:.2f}" + (f" [{lo * s:.2f}, {hi * s:.2f}]" if lo is not None else "")


def main():
    ap = argparse.ArgumentParser(description="Re-score run_benchmark.py prediction files with one protocol")
    ap.add_argument("runs", nargs="+", help='"Label=path/to/predictions.jsonl" (first = reference)')
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (default 1000)")
    ap.add_argument("--csv", help="write all numbers to this CSV")
    ap.add_argument("--latex", action="store_true", help="print R-L & EM per dataset as LaTeX table rows")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    runs = []
    for spec in args.runs:
        label, _, path = spec.partition("=") if "=" in spec else (spec, "", spec)
        by_ds, errors = load_rows(path, args.split)
        if not by_ds:
            sys.exit(f"{label}: no rows for split '{args.split}' in {path}")
        runs.append((label, by_ds, errors))

    datasets = [d for d in DATASET_ORDER if any(d in r[1] for r in runs)]
    datasets += sorted({d for r in runs for d in r[1]} - set(datasets))
    csv_rows = []

    print(f"Split: {args.split}   (all numbers in %, 95% bootstrap CI in brackets, {args.boot} resamples)\n")
    for label, by_ds, errors in runs:
        print(f"=== {label}" + (f"   ({errors} errored rows skipped)" if errors else ""))
        per_ds = {}
        for ds in datasets:
            if ds not in by_ds:
                continue
            s = per_ds[ds] = summarise(by_ds[ds], args.boot, rng)
            print(f"  {ds:14s} n={s['n']:6d}  EM {fmt(s['em'], s['em_lo'], s['em_hi'])}  "
                  f"EM-1w {fmt(s['em_1w'], s['em_1w_lo'], s['em_1w_hi'])}  "
                  f"R-L {fmt(s['rougeL'], s['rougeL_lo'], s['rougeL_hi'])}  "
                  f"B-1 {fmt(s['bleu1'])}  B-4 {fmt(s['bleu4'])}  "
                  f"mIoU {fmt(s['miou'])} (n={s['n_box']})")
            csv_rows += [{"run": label, "dataset": ds, "metric": k, "value": v}
                         for k, v in s.items() if v is not None]
        micro = summarise([r for ds in per_ds for r in by_ds[ds]], args.boot, rng)
        macro = {m: float(np.mean([per_ds[d][m] for d in per_ds if per_ds[d][m] is not None]))
                 for m in ["em", "em_1w", "rougeL", "bleu1", "bleu4"]}
        print(f"  {'OVERALL micro':14s} n={micro['n']:6d}  EM {fmt(micro['em'], micro['em_lo'], micro['em_hi'])}  "
              f"EM-1w {fmt(micro['em_1w'])}  R-L {fmt(micro['rougeL'])}")
        print(f"  {'OVERALL macro':14s} (mean of datasets)  EM {fmt(macro['em'])}  "
              f"EM-1w {fmt(macro['em_1w'])}  R-L {fmt(macro['rougeL'])}\n")
        csv_rows += [{"run": label, "dataset": "overall_macro", "metric": k, "value": v} for k, v in macro.items()]

        if args.latex:
            cells = [f"{per_ds[d]['rougeL']:.4f} & {per_ds[d]['em']:.4f}" if d in per_ds else "-- & --"
                     for d in datasets]
            print(f"  LaTeX: {label} & " + " & ".join(cells) + r" \\" + "\n")

    if len(runs) > 1:
        ref_label, ref_ds, _ = runs[0]
        print(f"=== Paired differences vs '{ref_label}' (same questions only; CI excluding 0 = significant)")
        for label, by_ds, _ in runs[1:]:
            print(f"  {label}")
            for ds in datasets:
                if ds not in by_ds or ds not in ref_ds:
                    continue
                parts, n_pairs = [], 0
                for m in CI_METRICS:
                    mean, lo, hi, n = paired_diff(by_ds[ds], ref_ds[ds], m, args.boot, rng)
                    n_pairs = n_pairs or n
                    if mean is None:
                        continue
                    sig = "*" if lo > 0 or hi < 0 else " "
                    parts.append(f"{m} {mean * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{sig}")
                    csv_rows.append({"run": f"{label} - {ref_label}", "dataset": ds, "metric": m,
                                     "value": mean, "ci_lo": lo, "ci_hi": hi, "n_paired": n})
                print(f"    {ds:14s} paired n={n_pairs:6d}  " + "  ".join(parts))
        print("  (* = 95% CI excludes zero. If paired n is 0, the runs share no questions: check --split.)")

    if args.csv:
        fields = ["run", "dataset", "metric", "value", "ci_lo", "ci_hi", "n_paired"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
