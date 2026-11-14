#!/usr/bin/env python3
"""
make_pilot_subset.py -- small, balanced training subset + configs for pilot studies.

From every training dataset in --base-config (train_jsonl, train_jsonl_2, ...):
  * up to --per-dataset questions, ONE question per image (max image variety),
    spread evenly over question categories (round-robin);
  * --val-per-dataset questions from that dataset's val file (val loss only).
Image paths are written absolute, so every config uses data_root "/".

Writes (default --out pilot_studies):
  pilot_studies/data/train_<dataset>.jsonl, val_<dataset>.jsonl
  pilot_studies/<experiment>/train_config.yaml   training config (save_dir = <experiment>/checkpoints)
  pilot_studies/<experiment>/eval_config.yaml    benchmark config: --eval-jsonl files (e.g. the blind
                                                 test and the SARLANG box test) -> results land in <experiment>/
Experiments: A_random_mlp (MLP from scratch) and, with --mlp-checkpoint, B_caption_mlp.

  python Benchmarking/make_pilot_subset.py --base-config train_config_marsscaleaware.yaml \
      --mlp-checkpoint /home/shruti/SAR_VQA/checkpoints/mlp_caption_only/step_23913/projector.pth \
      --eval-jsonl /home/shruti/SAR_VQA/data/blind_test.jsonl /home/shruti/SAR_VQA/data/SARLANG-1M/sarlang_box_test.jsonl
"""
import argparse
import collections
import copy
import json
import os
import random
from pathlib import Path

import yaml

DATA_KEYS = ("train_jsonl", "val_jsonl", "test_jsonl", "data_root", "dataset_name", "train_sample_size")


def sources(c_data: dict) -> list:
    """[(name, train_jsonl, val_jsonl, data_root)] in config order."""
    out, n = [], 1
    while True:
        sfx = "" if n == 1 else f"_{n}"
        tr = c_data.get(f"train_jsonl{sfx}")
        if not tr:
            return out
        va = c_data.get(f"val_jsonl{sfx}")
        if va and "test" in Path(va).name:                 # e.g. OEM val_jsonl = test_added.jsonl
            alt = Path(va).with_name(Path(va).name.replace("test", "val"))
            va = str(alt) if alt.exists() else va
        p = Path(tr)
        name = p.parent.parent.name if p.parent.name == "data" else p.parent.name
        out.append((name, tr, va, c_data.get(f"data_root{sfx}", c_data["data_root"])))
        n += 1


def load(path: str, root: str) -> list:
    recs = []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if not os.path.isabs(r["image"]):
                r["image"] = os.path.join(root, r["image"])
            recs.append(r)
    return recs


def answer_key(r: dict) -> str:
    """Answer bucket for balancing: lower-case text; counts of 4 or more share one bucket."""
    a = r["conversations"][-1]["value"].strip().lower().rstrip(".")
    return "4+" if a.isdigit() and int(a) >= 4 else a


def interleave_answers(v: list, rng: random.Random) -> list:
    """Order a category's records so answers come round-robin (pick() pops from the end)."""
    by_ans = collections.defaultdict(list)
    for r in v:
        by_ans[answer_key(r)].append(r)
    buckets = list(by_ans.values())
    for b in buckets:
        rng.shuffle(b)
    rng.shuffle(buckets)
    out = []
    while buckets:
        out += [b.pop() for b in buckets]
        buckets = [b for b in buckets if b]
    return out[::-1]


def pick(recs: list, n: int, rng: random.Random, balance_answers: bool = False, key: str = "category") -> list:
    """Up to n records, one per image, round-robin over r[key] (and over answers within one)."""
    by_cat = collections.defaultdict(list)
    for r in recs:
        by_cat[r.get(key, "?")].append(r)
    for k, v in by_cat.items():
        rng.shuffle(v)
        if balance_answers:
            by_cat[k] = interleave_answers(v, rng)
    queues, used, out = list(by_cat.values()), set(), []
    while len(out) < n and queues:
        nxt = []
        for q in queues:
            while q and q[-1]["image"] in used:
                q.pop()
            if q and len(out) < n:
                r = q.pop()
                used.add(r["image"])
                out.append(r)
            if q:
                nxt.append(q)
        queues = nxt
    return out


def write(path: Path, recs: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Balanced small training subset + pilot configs")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out", default="pilot_studies")
    ap.add_argument("--per-dataset", type=int, default=1000)
    ap.add_argument("--val-per-dataset", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--mlp-checkpoint", default=None, help="caption-pretrained projector.pth -> experiment B")
    ap.add_argument("--eval-jsonl", nargs="+", required=True, help="evaluation files (absolute image paths)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance-answers", action="store_true",
                    help="also balance answers within each question type -> experiment C_balanced_answers")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    base = yaml.safe_load(open(args.base_config))
    rng = random.Random(args.seed)
    train_files, val_files = [], []
    data_dir = out / ("data_balanced" if args.balance_answers else "data")
    for name, tr, va, root in sources(base["data"]):
        recs = load(tr, root)
        sub = pick(recs, args.per_dataset, rng, args.balance_answers)
        n_img = len({r["image"] for r in recs})
        missing = sum(not os.path.exists(r["image"]) for r in sub[:20])
        cats = collections.Counter(r.get("category", "?") for r in sub)
        cnt = collections.Counter(answer_key(r) for r in sub if "count" in r.get("category", "").lower())
        print(f"{name}: {len(recs)} questions on {n_img} images -> picked {len(sub)} "
              f"({len(cats)} categories, {min(cats.values())}-{max(cats.values())} each)"
              + (f"  WARNING {missing}/20 checked images missing" if missing else ""))
        if cnt:
            print(f"   counting answers: {dict(cnt.most_common())}")
        write(data_dir / f"train_{name}.jsonl", sub)
        train_files.append(str(data_dir / f"train_{name}.jsonl"))
        if va:
            vsub = pick(load(va, root), args.val_per_dataset, rng)
            write(data_dir / f"val_{name}.jsonl", vsub)
            val_files.append(str(data_dir / f"val_{name}.jsonl"))

    exps = {"A_random_mlp": None}
    if args.balance_answers:
        exps = {"C_balanced_answers": None}
    elif args.mlp_checkpoint:
        exps["B_caption_mlp"] = args.mlp_checkpoint
    for exp, mlp in exps.items():
        d = out / exp
        c = copy.deepcopy(base)
        c["data"] = {k: v for k, v in c["data"].items() if not k.startswith(DATA_KEYS)}
        for i, f in enumerate(train_files, 1):
            sfx = "" if i == 1 else f"_{i}"
            c["data"][f"train_jsonl{sfx}"] = f
            c["data"][f"data_root{sfx}"] = "/"
            if i <= len(val_files):
                c["data"][f"val_jsonl{sfx}"] = val_files[i - 1]
        c["data"]["train_sample_size"] = None
        t = c["training"]
        t.pop("checkpoint_path", None)
        t.update(epochs=args.epochs, seed=t.get("seed", 42), save_dir=str(d / "checkpoints"),
                 log_file=str(d / "training.log"), load_mlp=bool(mlp))
        if mlp:
            t["mlp_checkpoint"] = mlp
        else:
            t.pop("mlp_checkpoint", None)
        d.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(c, open(d / "train_config.yaml", "w"), sort_keys=False)

        e = copy.deepcopy(c)
        e["data"] = {}
        for i, f in enumerate(args.eval_jsonl, 1):
            sfx = "" if i == 1 else f"_{i}"
            e["data"].update({f"val_jsonl{sfx}": f, f"data_root{sfx}": "/", f"dataset_name{sfx}": Path(f).stem})
        yaml.safe_dump(e, open(d / "eval_config.yaml", "w"), sort_keys=False)
        print(f"wrote {d}/train_config.yaml and eval_config.yaml")

    print("\nTrain:    python train.py --config <out>/<experiment>/train_config.yaml")
    print("Evaluate: python Benchmarking/run_benchmark.py --config <out>/<experiment>/eval_config.yaml "
          "--split test --batch_size 16")


if __name__ == "__main__":
    main()
