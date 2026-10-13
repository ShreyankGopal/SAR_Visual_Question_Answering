# SAR VQA data generation

Builds visual question answering (VQA) datasets from SAR object-detection datasets. Every question and answer is
computed deterministically from the annotations and pixels , so a fixed seed always reproduces the same data.

| Dataset | Objects | Native GSD | QA pairs (train / val / test) |
|---|---|---|---|
| SIVED | vehicles (rotated boxes) | 0.1 / 0.3 m | 16,741 (13,394 / 1,680 / 1,667) |
| OGSOD | bridges, harbours, tanks | 3 m | 262,898 (184,001 / 26,297 / 52,600) |
| HRSID | ships (polygons → rotated boxes) | 0.5 / 1 / 3 m | 77,547 (55,818 / 7,044 / 14,685) |
| SAR-AIRcraft-1.0 | aircraft (512 px tiles) | 0.5 m | 156,916 (112,060 / 16,020 / 28,836) |

Question types per dataset, with the rule behind each answer: see `DATA_QUESTIONTYPES.md`.

## Quick start

```bash
pip install numpy pandas pillow pyyaml scipy matplotlib
python run.py --config configs/ogsod.yaml --stage all      # adapt -> facts -> build -> validate
```

Edit the paths at the top of the config first (`data.root`, `data.clean_jsonl`, `output.base_dir`).

| Stage | What it does | Output |
|---|---|---|
| `adapt` | Reads the raw dataset and writes one common format (cleaning, splits, tiling) | `clean.jsonl` |
| `facts` | Measures every object and image (position, size, grid cell, intensity, neighbours) | `intermediate_facts/*.csv` |
| `inspect` | Prints the distributions used to choose thresholds and bins | `reports/distributions.png` |
| `build` | Generates questions, balances answers, caps each image at 22 questions | `data/{train,val,test}.jsonl` |
| `validate` | Checks fields, coordinates, split leakage, and recomputes count/presence answers | `reports/validation_report.json` |

To look at generated questions drawn on their images (in Jupyter, from this folder):

```python
from core.spot_check import show
show('configs/hrsid.yaml', category='orientation', n=6)
```

## Layout

```
run.py              single entry point (--config, --stage)
configs/<name>.yaml everything dataset-specific: paths, classes, categories, thresholds, balancing, GSD
adapters/<name>.py  raw files -> clean.jsonl (the only dataset-specific code)
core/               shared by all datasets: geometry, facts, generators, balance, build, validate, inspect, spot_check
tools/              compare_qa.py / compare_clean.py: check that a rebuild matches a previous one
```

## Output record

```json
{"id": "hrsid_00012_004", "dataset": "HRSID", "image": "HRSID_JPG/JPEGImages/P0001_0_800_7200_8000.jpg",
 "category": "orientation",
 "ground_truth_facts": {"image_id": "...", "split": "train", "source": "Sentinel-1B",
                        "gsd_m": 4.6875, "native_gsd_m": 3.0, "resample_factor": 0.64,
                        "answer_key": "horizontal", "point": [0.41, 0.63], "...": "facts behind the answer"},
 "conversations": [{"from": "human", "value": "What is the orientation of the ship centred at [0.4100, 0.6300]?"},
                   {"from": "gpt", "value": "Horizontal."}]}
```

- `image` is relative to `data.root` in the config.
- Coordinates are normalised to [0, 1], so they stay valid when images are resized.
- `native_gsd_m`: metres per pixel of the stored image. `gsd_m`: metres per pixel at the model input size
  (`pipeline.model_input_size`, 512). `resample_factor` = native / gsd.

## Design choices 

- **Balanced answers.** Yes/No 50/50, A/B answers 50/50 first/second, capped shares for multi-answer categories,
  zero counts capped at 20% (rules per category under `balance:` in each config).
- **No shortcut references.** Objects are referred to by their centre point `[x, y]` in shape, size and brightness
  questions, so the numbers in the question cannot give away the answer.
- **Ambiguous cases are skipped**, not guessed: ties, objects cut by a region border or the image edge,
  angles near a bin edge, objects too small to see.
- **No split leakage.** HRSID is re-split by panorama (the official split shares overlapping tiles);
  SAR-AIRcraft tiles are cut inside each image of the official split.

## Adding a dataset

1. Write `adapters/<name>.py` that writes `clean.jsonl`: one line per image with `image`, `split`, `source`,
   `width`, `height`, `native_gsd_m` and `objects` (`category`, `hbox` in pixels, optional `rbox`, optional `truncated`).
2. Copy a config, set paths, `classes`, `categories_per_source` and `balance`.
3. Run `adapt`, `facts`, `inspect`; set thresholds and bins from the inspect output.
4. Run `build`, `validate`, then spot-check a few categories.
