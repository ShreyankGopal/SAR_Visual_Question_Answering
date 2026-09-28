"""run.py: single entry point for every dataset.

  python run.py --config configs/ogsod.yaml --stage adapt      # raw files -> clean JSONL
  python run.py --config configs/ogsod.yaml --stage facts      # clean JSONL -> objects.csv / images.csv
  python run.py --config configs/ogsod.yaml --stage inspect    # distributions for thresholds
  python run.py --config configs/ogsod.yaml --stage build      # QA -> train/val/test JSONL
  python run.py --config configs/ogsod.yaml --stage validate
  python run.py --config configs/ogsod.yaml --stage all        # adapt -> facts -> build -> validate
"""
import argparse
import importlib
import sys

from core import build, facts, inspect_data, validate
from core.io import load_config

STAGES = ["adapt", "facts", "inspect", "build", "validate", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    args = ap.parse_args()
    cfg = load_config(args.config)
    adapter = importlib.import_module(f"adapters.{cfg['adapter']}")

    todo = ["adapt", "facts", "build", "validate"] if args.stage == "all" else [args.stage]
    for stage in todo:
        print(f"\n### {cfg['dataset']}: {stage}")
        if stage == "adapt":
            adapter.run(cfg)
        elif stage == "facts":
            facts.run(cfg)
        elif stage == "inspect":
            inspect_data.run(cfg)
        elif stage == "build":
            build.run(cfg)
        elif stage == "validate" and not validate.run(cfg):
            sys.exit(1)


if __name__ == "__main__":
    main()
