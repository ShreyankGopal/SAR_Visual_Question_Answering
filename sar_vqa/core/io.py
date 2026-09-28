"""io.py: config, paths, JSONL and image loading shared by every stage."""
import json
import os

import numpy as np
import yaml
from PIL import Image


def load_config(path):
    return yaml.safe_load(open(path))


def out_path(cfg, key, *parts):
    """Path under output.base_dir/<output[key]>/..., directory created."""
    d = os.path.join(cfg["output"]["base_dir"], cfg["output"][key])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, *parts)


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(recs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def load_gray(cfg, rel_path):
    """Image as float32 grayscale; relative paths resolve against data.root."""
    return np.asarray(Image.open(os.path.join(cfg["data"]["root"], rel_path)).convert("L"), dtype=np.float32)


def stem_of(rel_path):
    return os.path.splitext(os.path.basename(rel_path))[0]


def slug(name):
    return name.lower().replace(" ", "_")


def native_gsd(cfg, rec):
    """Native metres per pixel: from the clean record, else from the config per source."""
    return rec.get("native_gsd_m") or cfg["native_gsd_m"][rec["source"]]
