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


def _qa(img, cat, ans):
    return {"image": img, "category": cat, "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": ans}]}


def test_balance_answers_evens_out_counts():
    # 80% of counting answers are "1"; balanced pick should give each bucket the same share
    recs = [_qa(f"i{i}", "counting", "1" if i % 5 else str(2 + (i // 5) % 4)) for i in range(400)]
    plain = mp.pick(recs, 40, random.Random(0))
    bal = mp.pick(recs, 40, random.Random(0), balance_answers=True)
    c = collections.Counter(mp.answer_key(r) for r in bal)
    assert collections.Counter(mp.answer_key(r) for r in plain)["1"] > 20
    assert set(c) == {"1", "2", "3", "4+"} and max(c.values()) - min(c.values()) <= 10
    assert len({r["image"] for r in bal}) == 40
