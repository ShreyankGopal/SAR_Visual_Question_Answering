"""Tests for Benchmarking/sarlang_to_jsonl.py: question types from SARLANG's CSV."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Benchmarking"))
import sarlang_to_jsonl as s2j  # noqa: E402


def test_question_type_from_csv(tmp_path):
    csv_path = tmp_path / "x.csv"
    csv_path.write_text('Image Name, Question, Answer, Application\n'
                        '0001.png,What is the object visible in the image?,ship,object classification\n'
                        '0001.png,"What is the object present in the area [9.0, 135.0, 53.0, 44.0]?",a harbor,region referring\n')
    types = s2j.load_types(str(csv_path))
    items = [{"images": ["/your_path/SARimages/0001.png"],
              "messages": [{"role": "user", "content": "<image>What is the object visible in the image?"
                                                      "The classes are: tank, car, ship."},
                           {"role": "assistant", "content": "ship"},
                           {"role": "user", "content": "What is the object present in the area [9.0, 135.0, 53.0, 44.0]?"},
                           {"role": "assistant", "content": "a harbor"},
                           {"role": "user", "content": "Something not in the CSV?"},
                           {"role": "assistant", "content": "x"}]}]
    idx = {"0001": ("/d/0001.jpg", "SARDet-100K")}
    recs = list(s2j.convert(items, idx, set(), "t", types))
    assert [r["question_type"] for r in recs] == ["object classification", "region referring", "unknown"]
    assert s2j.load_types(str(tmp_path / "missing.csv")) == {}


def test_image_name_with_dots(tmp_path):
    # DFC2023 names like GF2_AddisAbaba_9.0314_38.7359.png
    csv_path = tmp_path / "x.csv"
    csv_path.write_text("Image Name,Question,Answer,Application\n"
                        "GF2_A_9.03_38.73.png,What type of buildings are visible?,houses,object classification\n")
    items = [{"images": ["/your_path/SARimages/GF2_A_9.03_38.73.png"],
              "messages": [{"role": "user", "content": "<image>What type of buildings are visible?"},
                           {"role": "assistant", "content": "houses"}]}]
    idx = {"GF2_A_9.03_38.73": ("/d/GF2_A_9.03_38.73.png", "DFC2023")}
    recs = list(s2j.convert(items, idx, set(), "t", s2j.load_types(str(csv_path))))
    assert recs[0]["question_type"] == "object classification"
