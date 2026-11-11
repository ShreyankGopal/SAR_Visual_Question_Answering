#!/usr/bin/env python3
"""
make_sarlang_pilot.py -- pilot E: fine-tune the full SE-MaRS on a small SARLANG-1M train subset.

Inputs are the JSONLs from sarlang_to_jsonl.py (train AND test, both parts).
  * train: --per-part questions per train file, one per image, round-robin over image source
           (+ --val-per-part on other images, val loss only)
  * test : --test-per-part questions per test file -> fixed SARLANG test subset (same for before/after)
  * stops if any train image name is also a test image name
Writes <out>/data/*.jsonl, <out>/train_config.yaml (init_from = --init-from), <out>/eval_config.yaml.

  python Benchmarking/make_sarlang_pilot.py --base-config train_config_marsscaleaware.yaml \
      --init-from <SE-MaRS step_66080 folder> \
      --train sarlang_sardet_train.jsonl sarlang_other_train.jsonl \
      --test sarlang_sardet_test.jsonl sarlang_other_test.jsonl \
      --eval-extra blind_test.jsonl sarlang_box_test.jsonl
"""
import argparse
import collections
import copy
import os
import random
from pathlib import Path

import yaml

from make_pilot_subset import DATA_KEYS, load, pick, write


def name(r):
    return r["ground_truth_facts"]["image_name"]


def main():
    ap = argparse.ArgumentParser(description="SARLANG fine-tuning pilot: data subset + configs")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--init-from", required=True, help="step_N folder of the full SE-MaRS run")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--eval-extra", nargs="*", default=[], help="more eval files (blind test, box test)")
    ap.add_argument("--out", default="pilot_studies/E_sarlang_finetune")
    ap.add_argument("--per-part", type=int, default=2500)
    ap.add_argument("--val-per-part", type=int, default=100)
    ap.add_argument("--test-per-part", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    rng = random.Random(args.seed)
    tests = [load(f, "/") for f in args.test]
    test_names = {name(r) for t in tests for r in t}

    train_files, val_files = [], []
    for f in args.train:
        recs = load(f, "/")
        clash = {name(r) for r in recs} & test_names
        if clash:
            raise SystemExit(f"{f}: {len(clash)} train images are also SARLANG test images, e.g. {sorted(clash)[:5]}")
        sub = pick(recs, args.per_part, rng)
        used = {r["image"] for r in sub}
        vsub = pick([r for r in recs if r["image"] not in used], args.val_per_part, rng)
        stem = Path(f).stem
        write(out / "data" / f"train_{stem}.jsonl", sub)
        write(out / "data" / f"val_{stem}.jsonl", vsub)
        train_files.append(str(out / "data" / f"train_{stem}.jsonl"))
        val_files.append(str(out / "data" / f"val_{stem}.jsonl"))
        print(f"{stem}: {len(recs)} questions -> train {len(sub)}, val {len(vsub)}  "
              f"{dict(collections.Counter(r['category'] for r in sub))}")
    print("no train/test image overlap")

    eval_files = []
    for f, recs in zip(args.test, tests):
        sub = pick(recs, args.test_per_part, rng)
        p = out / "data" / f"test_{Path(f).stem}.jsonl"
        write(p, sub)
        eval_files.append(str(p))
        print(f"test {Path(f).stem}: {len(sub)}  {dict(collections.Counter(r['category'] for r in sub))}")
    eval_files += [os.path.abspath(f) for f in args.eval_extra]

    c = yaml.safe_load(open(args.base_config))
    c["data"] = {k: v for k, v in c["data"].items() if not k.startswith(DATA_KEYS)}
    for i, (tr, va) in enumerate(zip(train_files, val_files), 1):
        sfx = "" if i == 1 else f"_{i}"
        c["data"].update({f"train_jsonl{sfx}": tr, f"val_jsonl{sfx}": va, f"data_root{sfx}": "/"})
    c["data"]["train_sample_size"] = None
    t = c["training"]
    for k in ("checkpoint_path", "mlp_checkpoint"):
        t.pop(k, None)
    t.update(epochs=args.epochs, seed=t.get("seed", 42), save_dir=str(out / "checkpoints"),
             log_file=str(out / "training.log"), load_mlp=False, init_from=os.path.abspath(args.init_from))
    yaml.safe_dump(c, open(out / "train_config.yaml", "w"), sort_keys=False)

    e = copy.deepcopy(c)
    e["data"] = {}
    for i, f in enumerate(eval_files, 1):
        sfx = "" if i == 1 else f"_{i}"
        e["data"].update({f"val_jsonl{sfx}": f, f"data_root{sfx}": "/", f"dataset_name{sfx}": Path(f).stem})
    yaml.safe_dump(e, open(out / "eval_config.yaml", "w"), sort_keys=False)
    print(f"\nwrote {out}/train_config.yaml and eval_config.yaml")
    print(f"Train:    python train_sarlang.py --config {out}/train_config.yaml")
    print(f"Evaluate: python Benchmarking/run_benchmark.py --config {out}/eval_config.yaml --split test --batch_size 16")


if __name__ == "__main__":
    main()
