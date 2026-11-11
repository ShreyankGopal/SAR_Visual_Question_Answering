"""Tests for Benchmarking/linear_probe.py helpers (no model, no GPU)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Benchmarking"))
import linear_probe as lp  # noqa: E402


def test_box_tokens():
    assert lp.box_tokens([0, 0, 512, 512], 512, 512) == list(range(256))      # whole image
    assert lp.box_tokens([0, 0, 32, 32], 512, 512) == [0]                      # one cell
    assert lp.box_tokens([500, 500, 3, 3], 512, 512) == [255]                  # tiny box, last cell
    assert lp.box_tokens([33, 0, 2, 2], 512, 512) == [1]                       # smaller than a cell


def test_background_token_outside_boxes():
    t = lp.background_token([[0, 0, 256, 512]], 512, 512, random.Random(0))    # left half covered
    assert t % 16 >= 8
    assert lp.background_token([[0, 0, 512, 512]], 512, 512, random.Random(0)) is None


def test_pick_objects_caps():
    coco = {"categories": [{"id": 1, "name": "ship"}, {"id": 2, "name": "car"}],
            "annotations": [{"image_id": i // 5, "category_id": 1, "bbox": [0, 0, 5, 5]} for i in range(50)]
            + [{"image_id": 100, "category_id": 2, "bbox": [0, 0, 5, 5]}]}
    objs = lp.pick_objects(coco, per_class=8, per_image=2, rng=random.Random(0))
    ships = [o for o in objs if o["cls"] == "ship"]
    assert len(ships) == 8 and max(sum(o["image_id"] == i for o in ships) for i in range(10)) <= 2
    assert sum(o["cls"] == "car" for o in objs) == 1
