"""Tests for Benchmarking/make_pilot_subset.py: one question per image, categories balanced."""
import collections
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Benchmarking"))
import make_pilot_subset as mp  # noqa: E402


def test_pick_unique_images_and_balanced():
    # 3 categories; "big" has 100 questions, the others 20; images shared across questions
    recs = [{"image": f"img{i % 60}", "category": "big"} for i in range(100)]
    recs += [{"image": f"img{i}", "category": c} for c in ("a", "b") for i in range(60, 80)]
    out = mp.pick(recs, 30, random.Random(0))
    assert len(out) == 30
    assert len({r["image"] for r in out}) == 30                  # one question per image
    cats = collections.Counter(r["category"] for r in out)
    assert max(cats.values()) - min(cats.values()) <= 1          # even spread


def test_pick_stops_when_images_run_out():
    recs = [{"image": "same", "category": c} for c in "abc"]
    assert len(mp.pick(recs, 10, random.Random(0))) == 1
