#!/usr/bin/env python3
"""
make_sarlang_pilot.py -- fine-tune the full SE-MaRS on a SARLANG-1M train subset (pilot E, run F).

Inputs are the JSONLs from sarlang_to_jsonl.py (train AND test, both parts).
  * val  : --val-per-part questions per train file, one per image; those images are held out of training
  * train: per train file, --per-part questions, one per image, round-robin over --balance-key
           (-1 = one question for EVERY remaining image); files in --take-all use ALL their questions
  * test : --test-per-part questions per test file -> fixed test subset (0 = none; pass an existing
           subset via --eval-extra instead)
  * stops if any train image name is also a test image name
Writes <out>/data/*.jsonl, <out>/train_config.yaml (init_from = --init-from),
       <out>/eval_config.yaml (subset + --eval-extra), <out>/eval_full_sarlang.yaml (all --test files).
--keep-fractions 0.25 0.5 -> training.keep_steps: those checkpoints are copied to kept_checkpoints/.

Pilot E (5k):
  python Benchmarking/make_sarlang_pilot.py --base-config train_config_marsscaleaware.yaml \
      --init-from <SE-MaRS step_66080> --train <sardet_train> <other_train> --test <sardet_test> <other_test> \
      --eval-extra blind_test.jsonl sarlang_box_test.jsonl
Run F (~153k, 1 epoch):
  ... --per-part -1 --take-all <other_train> --balance-key question_type --val-per-part 300 \
      --test-per-part 0 --epochs 1 --keep-fractions 0.25 0.5 --out pilot_studies/F_sarlang_full \
      --eval-extra <E data/test_*.jsonl> sarlang_box_test.jsonl
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


def keep_steps(n_train: int, micro_batch: int, every: int, fractions: list) -> list:
    """Checkpoint steps at the given fractions of one epoch, rounded to multiples of `every`."""
    total = n_train // micro_batch
    return sorted({max(every, round(f * total / every) * every) for f in fractions})


def eval_cfg(c: dict, files: list) -> dict:
    e = copy.deepcopy(c)
    e["data"] = {}
    for i, f in enumerate(files, 1):
        sfx = "" if i == 1 else f"_{i}"
        e["data"].update({f"val_jsonl{sfx}": f, f"data_root{sfx}": "/", f"dataset_name{sfx}": Path(f).stem})
    return e


def main():
    ap = argparse.ArgumentParser(description="SARLANG fine-tuning: data subset + configs")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--init-from", required=True, help="step_N folder of the full SE-MaRS run")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--eval-extra", nargs="*", default=[], help="more eval files (blind test, box test)")
    ap.add_argument("--out", default="pilot_studies/E_sarlang_finetune")
    ap.add_argument("--per-part", type=int, default=2500, help="-1 = one question per image, all images")
    ap.add_argument("--take-all", nargs="*", default=[], help="train files whose questions are ALL used")
    ap.add_argument("--balance-key", default="category", help="round-robin field: category or question_type")
    ap.add_argument("--val-per-part", type=int, default=100)
    ap.add_argument("--test-per-part", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--keep-fractions", nargs="*", type=float, default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    rng = random.Random(args.seed)
    tests = [load(f, "/") for f in args.test]
    test_names = {name(r) for t in tests for r in t}
    take_all = {os.path.abspath(f) for f in args.take_all}
    key = args.balance_key

    train_files, val_files, n_train = [], [], 0
    for f in args.train:
        recs = load(f, "/")
        clash = {name(r) for r in recs} & test_names
        if clash:
            raise SystemExit(f"{f}: {len(clash)} train images are also SARLANG test images, e.g. {sorted(clash)[:5]}")
        vsub = pick(recs, args.val_per_part, rng, key=key)
        held = {r["image"] for r in vsub}
        rest = [r for r in recs if r["image"] not in held]
        if os.path.abspath(f) in take_all:
            sub = rest[:]
            rng.shuffle(sub)
        else:
            sub = pick(rest, len(rest) if args.per_part < 0 else args.per_part, rng, key=key)
        stem = Path(f).stem
        write(out / "data" / f"train_{stem}.jsonl", sub)
        write(out / "data" / f"val_{stem}.jsonl", vsub)
        train_files.append(str(out / "data" / f"train_{stem}.jsonl"))
        val_files.append(str(out / "data" / f"val_{stem}.jsonl"))
        n_train += len(sub)
        print(f"{stem}: {len(recs)} questions -> train {len(sub)} on {len({r['image'] for r in sub})} images, "
              f"val {len(vsub)} (held-out images)")
        for k in ("category", "question_type"):
            print(f"   {k}: {dict(collections.Counter(r.get(k, '?') for r in sub).most_common())}")
    print(f"no train/test image overlap; total train questions {n_train}")

    eval_files = []
    for f, recs in zip(args.test, tests):
        if args.test_per_part > 0:
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
    if args.keep_fractions:
        t["keep_steps"] = keep_steps(n_train, t["micro_batch_size"], t.get("checkpoint_every", 20), args.keep_fractions)
        print(f"keep_steps {t['keep_steps']} of ~{n_train // t['micro_batch_size']} steps/epoch")
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(c, open(out / "train_config.yaml", "w"), sort_keys=False)
    if eval_files:
        yaml.safe_dump(eval_cfg(c, eval_files), open(out / "eval_config.yaml", "w"), sort_keys=False)
    yaml.safe_dump(eval_cfg(c, [os.path.abspath(f) for f in args.test]), open(out / "eval_full_sarlang.yaml", "w"),
                   sort_keys=False)
    print(f"\nwrote {out}/train_config.yaml, eval_config.yaml, eval_full_sarlang.yaml")
    print(f"Train:    python train_sarlang.py --config {out}/train_config.yaml")
    print(f"Evaluate: python Benchmarking/run_benchmark.py --config {out}/eval_config.yaml --split test --batch_size 16"
          f"  [--checkpoint {out}/kept_checkpoints/step_N]")
    print(f"Full SARLANG test: python Benchmarking/run_benchmark.py --config {out}/eval_full_sarlang.yaml "
          f"--split test --batch_size 16")


if __name__ == "__main__":
    main()
