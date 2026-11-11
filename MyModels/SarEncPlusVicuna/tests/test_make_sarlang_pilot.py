"""Tests for Benchmarking/make_sarlang_pilot.py: subsets, overlap guard, init_from in config."""
import json
import os
import subprocess
import sys

import pytest
import yaml

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Benchmarking", "make_sarlang_pilot.py")


def _jsonl(path, names, src="SARDet-100K"):
    with open(path, "w") as f:
        for i, n in enumerate(names):
            f.write(json.dumps({"id": str(i), "image": f"/img/{n}.png", "category": src,
                                "ground_truth_facts": {"image_name": n, "source": src, "leaked": False},
                                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}) + "\n")


def _run(tmp_path, train_names):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"data": {"train_jsonl": "x", "data_root": "y"},
                                    "training": {"checkpoint_path": "old", "epochs": 9}}))
    _jsonl(tmp_path / "tr.jsonl", train_names)
    _jsonl(tmp_path / "te.jsonl", [f"t{i}" for i in range(30)])
    return subprocess.run([sys.executable, SCRIPT, "--base-config", str(base), "--init-from", "/ck/step_66080",
                           "--train", str(tmp_path / "tr.jsonl"), "--test", str(tmp_path / "te.jsonl"),
                           "--out", str(tmp_path / "E"), "--per-part", "20", "--val-per-part", "5",
                           "--test-per-part", "10"], capture_output=True, text=True)


def test_builds_subsets_and_config(tmp_path):
    r = _run(tmp_path, [f"a{i}" for i in range(50)])
    assert r.returncode == 0, r.stderr
    d = tmp_path / "E"
    tr = [json.loads(l) for l in open(d / "data" / "train_tr.jsonl")]
    va = [json.loads(l) for l in open(d / "data" / "val_tr.jsonl")]
    assert len(tr) == 20 and len(va) == 5 and not {x["image"] for x in tr} & {x["image"] for x in va}
    c = yaml.safe_load(open(d / "train_config.yaml"))
    assert c["training"]["init_from"].endswith("step_66080") and "checkpoint_path" not in c["training"]
    assert c["training"]["epochs"] == 3 and c["data"]["data_root"] == "/"
    e = yaml.safe_load(open(d / "eval_config.yaml"))
    assert sum(1 for _ in open(e["data"]["val_jsonl"])) == 10


def test_stops_on_train_test_overlap(tmp_path):
    r = _run(tmp_path, ["a0", "t3"])
    assert r.returncode != 0 and "also SARLANG test images" in r.stderr
