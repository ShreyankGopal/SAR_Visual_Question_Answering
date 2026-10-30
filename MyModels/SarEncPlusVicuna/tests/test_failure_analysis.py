"""Tests for Benchmarking/failure_analysis.py on synthetic prediction files (CPU, no model)."""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "Benchmarking", "failure_analysis.py")
sys.path.insert(0, os.path.join(HERE, "..", "Benchmarking"))
import failure_analysis as fa  # noqa: E402


def _row(i, gt, pred, correct, iou=None):
    return {"id": f"x{i}", "dataset": "hrsid", "split": "val", "category": f"cat{i % 2}",
            "image": f"img{i % 4}.png", "question": f"q{i}", "ground_truth": gt, "prediction": pred,
            "error": None, "exact_match": 1.0 if correct else 0.0, "bleu1": 0.5, "bleu4": 0.1,
            "rougeL": 1.0 if correct else 0.1, "iou": iou}


def test_error_types():
    def et(gt, pred, **kw):
        r = {"gt_n": fa.normalise(gt), "pred_n": fa.normalise(pred), "rougeL": kw.get("rl", 0.0), "iou": kw.get("iou")}
        return fa.error_type(r)
    assert et("yes", "") == "empty"
    assert et("[1,2,3,4]", "[5,6,7,8]", iou=0.1) == "box"
    assert et("yes", "No.") == "yes_no_flip"
    assert et("3", "four") == "count_off_by_1"
    assert et("2", "7") == "count_off_by_3plus"
    assert et("ship", "Yes, there is a ship.") == "answer_buried_in_longer_reply"
    assert et("residential area", "residential zone", rl=0.5) == "near_miss"
    assert et("forest", "water") == "wrong_answer"


def test_cli(tmp_path):
    # reference right on all; analysed run wrong on the first 40 -> regressions
    ref = [_row(i, "yes", "yes", True) for i in range(100)]
    new = [_row(i, "yes", "no" if i < 40 else "yes", i >= 40) for i in range(100)]
    for name, rows in (("ref", ref), ("new", new)):
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    # config: val file with GSD for 3 of 4 images; folder two levels up names the dataset
    data = tmp_path / "hrsid" / "data"
    data.mkdir(parents=True)
    with open(data / "val.jsonl", "w") as f:
        for k, g in enumerate([0.5, 1.0, 3.0, None]):
            f.write(json.dumps({"image": f"img{k}.png", "gsd_m": g}) + "\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"data:\n  val_jsonl: '{(data / 'val.jsonl').as_posix()}'\n  data_root: '{tmp_path.as_posix()}'\n")

    out = tmp_path / "out"
    p = subprocess.run([sys.executable, SCRIPT, f"Ref={tmp_path / 'ref.jsonl'}", f"New={tmp_path / 'new.jsonl'}",
                        "--config", str(cfg), "--out", str(out), "--boot", "200"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "GSD known for 75/100" in p.stdout

    cats = list(csv.DictReader(open(out / "by_category.csv")))
    all_new = [r for r in cats if r["category"] == "ALL" and r["run"] == "New"][0]
    assert abs(float(all_new["diff_em"]) + 0.40) < 1e-9 and all_new["diff_em_sig"] == "*"

    types = list(csv.DictReader(open(out / "error_types.csv")))
    assert {r["error_type"] for r in types if r["run"] == "New"} == {"yes_no_flip"}

    ex = list(csv.DictReader(open(out / "examples.csv")))
    assert ex and {r["case"] for r in ex} == {"regression"} and len(ex) <= 20

    gsd = list(csv.DictReader(open(out / "gsd.csv")))
    assert {r["gsd_bin_m"] for r in gsd} == {"0.5", "1", "3"}
