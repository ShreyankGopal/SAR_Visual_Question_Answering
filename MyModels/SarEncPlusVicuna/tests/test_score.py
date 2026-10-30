"""Tests for Benchmarking/score.py on synthetic prediction files (CPU, no model)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE = os.path.join(HERE, "..", "Benchmarking", "score.py")
sys.path.insert(0, os.path.join(HERE, "..", "Benchmarking"))
import score  # noqa: E402


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(i, dataset, split, gt, pred, correct):
    return {"id": f"{dataset}_{i}", "dataset": dataset, "split": split, "category": "c",
            "question": f"q{i}", "ground_truth": gt, "prediction": pred, "error": None,
            "exact_match": 1.0 if correct else 0.0, "bleu1": 0.5, "bleu4": 0.1,
            "rougeL": 1.0 if correct else 0.2, "iou": None}


def _make(tmp_path, name, n_correct, n=200):
    # 'DataGen' must be treated as openearthmap; half single-word GTs; plus test rows to be filtered out
    rows = [_row(i, "DataGen", "val", "yes" if i % 2 else "two words", "x", i < n_correct) for i in range(n)]
    rows += [_row(i, "DataGen", "test", "yes", "yes", True) for i in range(1000, 1050)]
    rows.append({**_row(9999, "DataGen", "val", "yes", "", False), "error": "boom", "exact_match": None})
    path = str(tmp_path / f"{name}.jsonl")
    _write(path, rows)
    return path


def test_split_alias_and_metrics(tmp_path):
    by_ds, errors = score.load_rows(_make(tmp_path, "a", n_correct=50), "val")
    assert list(by_ds) == ["openearthmap"] and errors == 1
    rows = by_ds["openearthmap"]
    assert len(rows) == 200                                   # test rows and the errored row excluded
    import numpy as np
    s = score.summarise(rows, 200, np.random.default_rng(0))
    assert abs(s["em"] - 0.25) < 1e-9
    assert s["em_lo"] < 0.25 < s["em_hi"]
    # single-word GTs are the odd i; correct ones are i < 50 -> 25 of 100
    assert abs(s["em_1w"] - 0.25) < 1e-9


def test_paired_difference_detects_real_gain(tmp_path):
    import numpy as np
    ref, _ = score.load_rows(_make(tmp_path, "ref", n_correct=40), "val")
    new, _ = score.load_rows(_make(tmp_path, "new", n_correct=120), "val")
    mean, lo, hi, n = score.paired_diff(new["openearthmap"], ref["openearthmap"], "em", 500, np.random.default_rng(0))
    assert n == 200 and abs(mean - 0.40) < 1e-9 and lo > 0      # +40 points, CI excludes zero


def test_cli_runs(tmp_path):
    out = subprocess.run([sys.executable, SCORE, f"Ref={_make(tmp_path, 'r', 40)}",
                          f"New={_make(tmp_path, 'n', 120)}", "--split", "val", "--boot", "200",
                          "--csv", str(tmp_path / "s.csv"), "--latex"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Paired differences" in out.stdout and "LaTeX: New" in out.stdout
    assert os.path.exists(tmp_path / "s.csv")
    micro = [l for l in out.stdout.splitlines() if "OVERALL micro" in l][0]
    assert "B-4" in micro and "Acc@0.5" in micro
    assert "overall_micro" in open(tmp_path / "s.csv").read()
