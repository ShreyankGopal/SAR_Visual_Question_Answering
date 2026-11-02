"""Tests for Benchmarking/sardet_leakage.py: re-encoded copy matches, different image doesn't."""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "Benchmarking", "sardet_leakage.py")


def test_finds_exact_and_reencoded(tmp_path):
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64)) * 255).astype(np.uint8)
    b = (rng.random((64, 64)) * 255).astype(np.uint8)
    (tmp_path / "ours").mkdir(); (tmp_path / "sd").mkdir()
    Image.fromarray(a).save(tmp_path / "ours" / "a.png")
    Image.fromarray(b).save(tmp_path / "ours" / "b.png")
    (tmp_path / "ours" / "train.jsonl").write_text('{"image": "a.png"}\n{"image": "b.png"}\n')
    Image.fromarray(a).save(tmp_path / "sd" / "1.png")                    # exact copy
    Image.fromarray(b).save(tmp_path / "sd" / "2.jpg", quality=95)        # re-encoded copy
    Image.fromarray((rng.random((64, 64)) * 255).astype(np.uint8)).save(tmp_path / "sd" / "3.png")
    out = tmp_path / "o.csv"
    subprocess.run([sys.executable, SCRIPT, "--sardet", str(tmp_path / "sd" / "*"),
                    "--train", str(tmp_path / "ours" / "train.jsonl"), "--thr", "0.9",
                    "--out", str(out)], check=True)
    rows = {l.split(",")[0]: l.split(",")[1:3] for l in out.read_text().splitlines()[1:]}
    assert rows["1.png"] == ["train", "exact"]
    assert rows["2.jpg"] == ["train", "thumb"]
    assert rows["3.png"][0] == "none"
