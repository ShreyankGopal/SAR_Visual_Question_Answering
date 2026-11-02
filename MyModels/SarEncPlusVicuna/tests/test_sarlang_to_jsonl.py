"""Tests for Benchmarking/sarlang_to_jsonl.py: name matching, prompt, leak tag, skip count."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Benchmarking"))
import sarlang_to_jsonl as s2j  # noqa: E402


def test_convert():
    items = [
        {"messages": [{"role": "user", "content": "<image>How many ships?"},
                      {"role": "assistant", "content": "3"}],
         "images": ["/your_path/SARimages/0001.png"]},
        {"messages": [{"role": "user", "content": "<image>Q"}, {"role": "assistant", "content": "A"}],
         "images": ["/your_path/SARimages/missing.png"]},
    ]
    idx = {"0001": ("/data/sardet/test/0001.jpg", "SARDet-100K")}
    recs = list(s2j.convert(items, idx, leaked={"0001"}, tag="t"))
    assert len(recs) == 1
    r = recs[0]
    assert r["image"] == "/data/sardet/test/0001.jpg" and r["ground_truth_facts"]["leaked"] is True
    assert r["conversations"][0]["value"] == "Question: How many ships?. Answer the question concisely."
    assert r["conversations"][1]["value"] == "3" and r["dataset"] == "SARLANG-SARDet-100K"
