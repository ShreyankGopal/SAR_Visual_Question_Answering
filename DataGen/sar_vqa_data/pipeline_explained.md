# SAR VQA Data Pipeline, Explained

There are **two separate, independent VQA-generation pipelines** in this
repo. §1–§6 describe `DataGen/sar_vqa_data/`, which handles the four object-
detection-style SAR datasets (SIVED, HRSID, OGSOD, SAR-AIRcraft). §7 describes
`DataGen/dataset_generation/`, an earlier, separate pipeline that generates
land-cover VQA from OpenEarthMap (OEM) SAR tiles. They share no code and run
independently via different entry points.

## Part A: SIVED / HRSID / OGSOD / SAR-AIRcraft — `DataGen/sar_vqa_data/`

This describes `DataGen/sar_vqa_data/` end to end: how raw dataset files become
`train.jsonl` / `val.jsonl` / `test.jsonl` VQA records, what every question type
actually asks and why, how each of the four source datasets (SIVED, HRSID,
OGSOD, SAR-AIRcraft) is handled, and — in detail — how every numeric threshold
that decides a class/category/answer was chosen.

Entry point: `python run.py --config configs/<dataset>.yaml --stage {adapt,facts,inspect,build,validate,all}`.

---

## 1. The five stages

```
raw dataset files
   │
   ▼
┌─────────┐   clean.jsonl (common schema: image, split, source, width,
│  adapt  │ → height, native_gsd_m, objects[{category, hbox, rbox?, truncated?}])
└─────────┘
   │
   ▼
┌─────────┐   objects.csv (per-object geometry/intensity/neighbour facts)
│  facts  │ → images.csv  (per-image totals, per-class counts, per-grid-cell
└─────────┘                counts/coverage)
   │                       meta.json ({bright_threshold})
   ▼
┌──────────┐   (optional, human-in-the-loop) prints distribution stats and
│ inspect  │   quantile cut points, saves reports/distributions.png — this is
└──────────┘   how the threshold numbers in the config yamls were chosen.
   │
   ▼
┌─────────┐   generates every candidate question/answer per image, balances
│  build  │ → the mix per category, caps questions per image, writes the
└─────────┘   final train/val/test.jsonl + reports/qa_summary.txt
   │
   ▼
┌──────────┐   re-derives answers from images.csv independently and checks
│ validate │   them against what build.py wrote, checks file existence,
└──────────┘   coordinate ranges, split leakage, duplicates → validation_report.json
```

**adapt** — each dataset has its own adapter (`adapters/sived.py`,
`hrsid.py`, `ogsod.py`, `sar_aircraft.py`) that reads that dataset's native
annotation format (XML, COCO JSON, YOLO txt, VOC XML — all different) and
converts it into one shared, simple schema. This is the only stage that knows
anything dataset-specific about file formats.

**facts** — walks every image and, before any questions are written, computes
all the raw numbers a question might need: how much of the image each class
covers, how many objects of each class sit in each of 9 grid regions, each
object's SAR brightness, its nearest neighbour, its orientation, etc. It also
computes one global number per dataset: the "bright SAR pixel" threshold (the
90th percentile pixel intensity across a sample of training images) used later
by the `sar_observation` question type.

**inspect** — not part of the automated build. A human runs this, reads the
printed quantiles/crosstabs, and manually copies interesting cut points (e.g.
"the size tertiles are 37.3 and 125.1") into that dataset's yaml config. This
is literally how several of the thresholds below were chosen — see §4.

**build** — for every image, runs all 15 question-generator functions, each
producing zero or more candidate question/answer pairs. Then it *balances*
the pile of candidates (so e.g. "Yes" doesn't drown out "No" 20:1, or one easy
category doesn't dominate the dataset), caps each image at 22 questions, and
writes the final JSONL files plus a human-readable summary report.

**validate** — a sanity pass. It doesn't trust that `build.py` got everything
right; it independently recomputes some answers straight from `images.csv`
and flags any mismatch, plus checks for missing images, bad coordinates,
duplicate questions, and data leaking across train/val/test splits.

---

## 2. Question types — what each one asks and how

Every generator follows the same shape: look at one image's facts, decide
whether a question *can* be asked unambiguously (this is where thresholds
come in — see §4), and if so, produce a question/answer pair. All 15 are
registered in a `GENERATORS` dict and a config (`categories_per_source`)
decides which generators run for which dataset.

### global_quantitative
*"Approximately what percentage of this image is covered by vehicles?"*
→ a percentage-range label (e.g. "2–5%"). Based on how much of the image's
pixels fall inside any object of that class (or, for multi-class images, the
union of classes). The percentage is binned into ranges (see `coverage_bins`
in §4) rather than given as an exact number, because exact pixel coverage
isn't something a model should be expected to state precisely — a range is
the honest level of precision.

### sar_comparative
*"Which vehicle has the stronger average SAR return: the one centred at (x,y)
or the one at (x,y)?"* → picks the two with higher/lower mean pixel intensity
inside their box. Restricted to objects of the *same* class, specifically so
the class name itself can't be used to guess the answer — the model has to
actually look at brightness.

### comparative_spatial
*"Are there more bridges in the top-left region or the top-right region?"*
→ compares object counts between two of the image's 9 grid cells.

### regional_vqa
Two forms: *"Are there any ships in the center region?"* (Yes/No presence per
class per grid cell), and, for OGSOD only, *"Which object types are in the
middle-left region?"* (lists all classes present there).

### region_grounding
Three sub-types, all asking about a randomly generated bounding box drawn on
the image:
- **presence**: *"Is there a tank in the region [x1,y1,x2,y2]?"*
- **referring**: *"Give the bounding box of the vehicle closest to the image
  center?"* / *"...closest to the top?"* / *"...with the longest body?"*
  (SIVED/HRSID/SAR-AIRcraft), or *"Give the bounding box of the harbour?"*
  when a class has exactly one instance in the image (OGSOD).
- **class_in_region**: *"What type of object is in the region [x1,y1,x2,y2]?"*
  (OGSOD only, since it's the only multi-class dataset where this is
  interesting).

### object_relations
*"In which direction is the nearest ship from the ship centred at (x,y)?"*
→ one of 8 compass directions, from the nearest-neighbour object (any class).

### orientation
*"What is the orientation of the vehicle centred at (x,y)?"* → horizontal /
vertical / one of two diagonal labels, from the object's rotated box angle.

### relative_size
*"Which ship is longer: the one centred at (x,y) or the one at (x,y)?"* →
compares two same-class objects' lengths.

### arrangement
*"Are the vehicles in this image mostly aligned in the same direction?"* →
Yes/No, from how parallel all the objects' angles are (SIVED only — this is
specifically about vehicle convoys/parking, which doesn't apply to ships,
tanks, bridges, harbours, or aircraft).

### object_counting
*"How many ships are in this image?"* / *"...in the center region?"* /
*"...in the region [x1,y1,x2,y2]?"* → an exact count.

### global_classification / object_classification
OGSOD only (the only multi-class dataset where "what kind of object is this"
is a real question): *"Is there a tank in this image?"*, *"What types of
objects are in this image?"*, *"What type of object is centred at (x,y)?"*.

### absolute_size
*"Approximately how long is the ship centred at (x,y)?"* → a size range in
metres (e.g. "50–100m"), using the dataset's known ground sample distance to
convert pixel length to real-world metres. Only defined for classes where
real-world size is a meaningful, answerable thing (ships' length, tanks'
diameter, harbours' extent) — see §4 for exactly which classes and bins.

### sar_observation
OGSOD only: *"Which object type in this image has the lowest fraction of
bright SAR pixels?"* / *"...highest..."* → ranks classes by what fraction of
their pixels are "bright" (above the dataset's 90th-percentile threshold from
the facts stage).

### sar_bbox_variance
*"How heterogeneous is the SAR response in the region [x1,y1,x2,y2]?"* →
Low/Medium/High, from the raw pixel variance inside a random box, binned by
dataset-specific tertiles (see §4).

**Correction:** `DATA_QUESTIONTYPES.md` also documents a `surrounding_classes`
question type tied to an "OEM" (OpenEarthMap land-cover) dataset. That
generator does not exist in *this* pipeline's `generators.py` — but OEM is
real and is generated by a separate, earlier pipeline at
`DataGen/dataset_generation/`, covered in full in §7 below.

---

## 3. Per-dataset handling

### SIVED (vehicles)
Reads per-image XML annotation files (rotated boxes: center, width, height,
angle) plus a parallel set of DOTA-format polygon files. Before trusting the
XML's angle, the adapter re-derives each box from its polygon and compares —
if they disagree by more than 2 pixels at any corner, it **refits the box
from the polygon** instead (481 boxes were corrected this way). Single class,
`Vehicle`. Images come from three different sensors (FARAD, MiniSAR, MSTAR)
with two different native resolutions (0.1 m and 0.3 m); sensor name is kept
as `source`, and whether it's an MSTAR scene (vs. FARAD/MiniSAR) sets a
`scenario` field used to pick which questions apply (e.g. `arrangement` only
makes sense for convoy/urban scenes). Uses its own official train/val/test
split as-is.

### HRSID (ships)
Reads one COCO-format JSON with polygon ship outlines. Each image's sensor,
imaging mode, polarization, and ground resolution are looked up from a table
in the config, keyed by the first 5 characters of the filename (a panorama
ID, e.g. `P0001`) — one panorama ID (`P0137`) isn't in that table and is
dropped entirely rather than guessed. Polygons are converted to tight rotated
boxes. Because the *official* HRSID split shares the same underlying
panoramas (and 200px overlapping tiles) across train and test — which would
leak data — the adapter **re-splits by whole panorama**: a few specific
high-resolution panoramas are hand-assigned to train/test, and the rest are
greedily assigned to whichever split (train 70% / val 10% / test 20%) is
furthest below its target share. Native resolution varies by panorama (0.5m,
1m, or 3m) depending on which satellite took it (Sentinel-1B, TerraSAR-X,
TanDEM-X) — that satellite name is what balancing stratifies by.

### OGSOD (bridges, harbours, tanks)
Reads YOLO-format normalized label files, matched to PNG images (the
dataset's parallel optical/RGB imagery is explicitly ignored — SAR only).
Three classes via a fixed id→name table. Degenerate boxes under 2 pixels wide
or tall are dropped (40 of them). The dataset ships with only train/test, so
val is carved out of train by randomly moving 12.5% of images from each
*stratum* (defined by exactly which combination of classes appears in that
image) into val — stratifying this way keeps rare class combinations
represented in val instead of accidentally landing entirely in train. Single
sensor, single native resolution (3 m).

### SAR-AIRcraft (aircraft)
Reads VOC-style XML boxes over large (800–1500px) images, with 7
fine-grained aircraft type names in the raw data (A220, A320/321, A330,
ARJ21, Boeing 737, Boeing 787, "other"). Because individual aircraft type
isn't something you could actually verify by eye in a SAR image, these are
kept only as metadata, not used as VQA classes — everything becomes one
`Aircraft` class. Large images are **tiled** into 512×512 pieces (evenly
spaced, e.g. a 1000px image becomes a 2×2 grid of tiles) strictly *within*
each image's own official split, so no tile ever crosses from train into test.
Aircraft boxes are clipped per tile; an aircraft cut down to less than 25% of
its original area is dropped as a sliver, and partially-cut ones are flagged
`truncated`. Tiles with zero aircraft on them are capped at 15% of that
split's kept tiles (extra empty tile image files are actually deleted from
disk) so the dataset isn't mostly empty sky. The dataset's stated resolution
is 1 m/px, but the adapter uses 0.5 m/px for actual size-in-metres
calculations, based on cross-checking real aircraft sizes against pixel
sizes — the stated 1 m value is kept separately as `resolution_m` metadata.

---

## 4. How every threshold was chosen

Every threshold lives in that dataset's YAML config under `thresholds:`,
`reference:`, `grounding:`, `sar:`, or `sar_bbox_variance:` — none are
hidden/hardcoded inside the Python (the only real exceptions are a handful of
genuinely fixed geometric facts, noted at the end of this section). Three
different *kinds* of reasoning produced these numbers:

**(a) Derived from the data itself, via the `inspect` stage.** You run
`--stage inspect`, it prints quantiles (and for one specific threshold,
tertiles) computed over a sample of real images, and a human copies those
numbers into the config. This is explicitly how the `sar_bbox_variance` bins
were set: `inspect_data.py` samples 1500 images (2 random regions each),
computes the pixel variance in each, and prints the 1/3 and 2/3 quantiles —
those exact numbers were then pasted into each dataset's config with a
comment naming them ("HRSID tertiles", "OGSOD tertiles", etc.):

| Dataset | Low | Medium | High (cutoffs, raw pixel variance) |
|---|---|---|---|
| HRSID | 0 – 37.3 | 37.3 – 125.1 | 125.1 – ∞ |
| OGSOD | 0 – 1428.8 | 1428.8 – 2430.6 | 2430.6 – ∞ |
| SAR-AIRcraft | 0 – 60.4 | 60.4 – 545.5 | 545.5 – ∞ |

The large gap between datasets (OGSOD's numbers are ~30x HRSID's) reflects
real differences in each dataset's raw pixel value range/contrast stretching
— they're dataset-specific precisely because a fixed global cutoff wouldn't
mean the same thing across sensors.

The `absolute_size` bins for OGSOD's `Harbour` class are commented
`# from inspect section 7` in the config — same process, applied to
object-extent distributions instead of pixel variance.

**(b) A reasoned margin around a known measurement error, to avoid asking
ambiguous questions.** Several thresholds exist purely to *skip* a question
when the true answer is too close to a bin boundary to be confident about,
given known pixel-level imprecision:
- `size_margin_m` (absolute_size): skip if the computed real-world size is
  within this many metres of a bin edge. HRSID uses 3m ("one 3m pixel" —
  i.e. don't trust a size estimate that could flip bins from a single pixel
  of measurement noise). OGSOD uses 1.5m ("half a native pixel").
- `size_min_length_ratio` (relative_size / referring "longest"): one object
  must be at least this many times longer than another to call it "longer."
  1.2 for SIVED/HRSID. **1.45 for SAR-AIRcraft specifically**, with the
  reasoning spelled out in the config comment: a box around a *diagonal*
  object is up to 1.41x larger than one around the same object aligned to
  the axes, purely from the rotation — so a ratio threshold below ~1.41
  could be comparing boxes that only *look* different because of alignment,
  not because the aircraft are actually different sizes.
- `relation_min_nn2_ratio` (object_relations) = 1.2 for every dataset: the
  second-nearest neighbour must be at least 20% farther away than the
  nearest one, so "the nearest X" isn't a coin-flip between two nearly-tied
  candidates.
- `relation_dir_margin_deg` = 7.5° for every dataset: the nearest neighbour's
  direction must be at least 7.5° away from a 45°-sector boundary (e.g. not
  sitting right on the line between "above" and "above-right"), so the
  compass-direction answer isn't borderline.
- `orient_margin_deg` = 5° (SIVED/HRSID only, where `orientation` applies):
  same idea, margin from the 0°/22.5°/67.5°/90° bin edges.
- `orient_min_aspect` = 1.3 (SIVED/HRSID): an object's length/width ratio
  must be at least 1.3 before "orientation" is even a meaningful question —
  a near-square object doesn't really have an orientation.
- `referring_min_gap` = 0.02 (normalized image-diagonal units, used
  everywhere `referring` cues apply): the winning candidate for "closest to
  center" / "closest to top" must beat the runner-up by at least this much.

**(c) A reasoned threshold based on what's visually/physically distinguishable,
tuned per class because the classes themselves differ in scale and density.**
- `spatial_min_count_diff` (comparative_spatial) — how many more objects one
  region must have than another before "more X in region A" is a fair
  question. SIVED uses 2 (vehicles are dense, so a 1-object gap could be
  SIVED's labeling resolution rather than a visible difference). HRSID and
  SAR-AIRcraft use 1 (ships/aircraft are comparatively sparse, so even a
  single extra object is a visible difference). OGSOD sets this **per
  class**, explained directly in the config: `Bridge: 1, Harbour: 1` ("sparse
  and large, 1 vs 0 is clear"), `Tank: 2` (tanks cluster tightly, so a
  1-object difference is noise, not a real pattern).
- `point_min_sep_px` (`reference`, controls which objects are usable as
  "the X centred at (x,y)" reference points): 20px for SIVED/HRSID/
  SAR-AIRcraft, but **6px for OGSOD's tanks specifically** — the config
  comment explains tanks are only ~7–8px wide and sit in tight clusters, so
  a 20px separation rule would exclude almost every tank from ever being
  referenced; 6px is set just below one tank-width so points still land
  inside a single tank unambiguously.
- `arrangement_min_vehicles` = 4 and the yes/no cutoffs `arrangement_yes_min`
  = 0.9 / `arrangement_no_max` = 0.6 (SIVED only): need at least 4 vehicles
  before "alignment" is meaningful, and the 0.6–0.9 middle zone is
  deliberately left unanswered (skipped) rather than forced into Yes/No,
  because that's genuinely the "somewhat aligned" zone where a confident
  answer isn't honest.
- `observation_min_diff` = 0.05 (OGSOD's `sar_observation` only): the
  lowest-brightness and second-lowest-brightness classes (or highest/second)
  must differ by at least 5 percentage points of "bright pixel fraction"
  before ranking them is meaningful.

**(d) Marked explicitly as provisional, pending real data.** Two thresholds
carry comments admitting they weren't derived from inspection yet:
`sar_min_intensity_diff` (sar_comparative — minimum mean-intensity gap to
call one object's SAR return "stronger") is 15 for SIVED and SAR-AIRcraft, 20
for HRSID/OGSOD, with HRSID's comment reading `# provisional: inspect section
6` and SAR-AIRcraft's reading `# provisional: stretched tiles change the
scale` — i.e. these are best-guess starting values the author flagged as
needing a proper inspect-stage check before being trusted.

**`sar.bright_percentile` = 90** (HRSID, OGSOD) defines what counts as a
"bright" SAR pixel at all (used by `sar_observation` and per-object
`bright` fraction facts): the top 10% of pixel intensity values, sampled from
up to 500 training images (every 16th pixel, for speed) — a percentile
rather than an absolute pixel value specifically so it adapts to each
dataset's own intensity range instead of assuming a shared brightness scale
across sensors.

### The few genuinely hardcoded (not config) numbers
A handful of constants are fixed geometry, not tunable thresholds, so they
live directly in `core/geometry.py` rather than any config:
- Orientation bins split at 22.5° and 67.5° — these are exactly 1/4 and 3/4
  of 90°, i.e. the only sensible way to cut a quarter-turn into 4 equal bins.
- The 8 compass directions are 8 equal 45° sectors — again the only sensible
  even split.
- `bright_threshold`'s sampling (500 images, every 16th pixel) is a
  performance/precision tradeoff, not a scientific choice.

### `max_per_category` / `max_qa_per_image` / balance caps
Separately from "is this specific question answerable," every config also
limits *volume*, same values across all four datasets: at most 3 candidate
questions kept per generator per image (`pipeline.max_per_category`), at most
22 total questions per image after mixing all categories
(`pipeline.max_qa_per_image`), and a balance pass afterward that caps any one
(category, answer) combination at a configured share of that category's
total — e.g. `share: 0.35` for orientation everywhere, `share: 0.2` for
object_counting's zero-vs-nonzero split everywhere — so one easy, common
answer (like "No" or "0") can't dominate a category 10:1.

---

## 5. Key numbers

| Dataset | Images used | Classes | Native GSD | Effective GSD (@512 input) | Total QA (train / val / test) |
|---|---|---|---|---|---|
| SIVED | 1,044 | Vehicle | 0.1 m (FARAD/MiniSAR) or 0.3 m (MSTAR) | same (already 512px, resample ×1.0) | 16,741 (13,394 / 1,680 / 1,667) |
| HRSID | 5,495 (1 panorama dropped) | Ship | 0.5 / 1 / 3 m (varies by panorama) | 0.78 / 1.56 / 4.69 m (resample ×0.64, a real downsample) | 77,547 (55,818 / 7,044 / 14,685) |
| OGSOD | 18,331 | Bridge, Harbour, Tank | 3 m | 1.5 m (resample ×2.0) | 262,898 (184,001 / 26,297 / 52,600) |
| SAR-AIRcraft | 13,552 tiles (from 4,368 images) | Aircraft | 0.5 m (stated 1 m) | 0.5 m (resample ×1.0) | 156,916 (112,060 / 16,020 / 28,836) |

Fixed across every dataset's config: `model_input_size = 512`,
`max_qa_per_image = 22`, `max_per_category = 3`, `random_seed = 42` (so every
run — sampling, box randomization, split assignment, balancing — is
reproducible).

Dataset-specific cleaning counts: SIVED refit 481 boxes from DOTA polygons
(XML/polygon disagreement); OGSOD dropped 40 degenerate (<2px) boxes and
clipped 253; HRSID dropped 1 panorama (109 tiles) with no metadata entry, 12
unusable polygons, and flagged 2,005 ships (12%) as truncated; SAR-AIRcraft
flagged 6,762 of 24,858 aircraft instances as truncated from tiling, and
capped empty tiles at 15% per split.

HRSID's re-split by panorama: 3,960 / 489 / 1,046 images, 11,774 / 1,628 /
3,391 ships (train/val/test), targeting a 70/10/20 share. OGSOD's val is
12.5% of official train, carved out stratified by which class-combination
each image contains (12,830 / 1,834 / 3,667 images).

---

## 6. `ground_truth_facts` — what's recorded with every answer

Every output record carries a `ground_truth_facts` block with these fixed
fields on every single record: `image_id`, `split`, `source`, `scenario`
(may be null), `band` (may be null), `polarization` (may be null), `gsd_m`,
`native_gsd_m`, `resample_factor`, `answer_key`.

On top of that, each question category adds its own category-specific
evidence fields — e.g. `point`/`points` (object centre coordinates) for
questions that reference a specific object, `bbox`/`region`/`regions` for
grounding/counting questions, and numeric facts like `coverage_pct`,
`length_m`, `mean_int`, `variance`, `count`, `alignment`, `nn_dist_px`,
depending on what that category's answer was actually computed from. The
`validate` stage spot-checks some of these (class name, region, subtype)
against `images.csv` independently, to catch any mismatch between what a
generator claims and what's actually true of the image.

---

## Part B: OEM / land-cover — `DataGen/dataset_generation/`

A separate, earlier pipeline, generating land-cover VQA from OpenEarthMap SAR
tiles — no adapters, no `run.py`, no `--stage` flags; it's one script.
Entry point: `python dataset_generation/build_dataset.py --config
dataset_generation/config.yaml [--no-vis]`.

### 7.1 Pipeline flow

```
OpenEarthMap SAR tiles (1024×1024) + label masks
   │
   ▼
Phase 1  inspect: scan every train tile/mask pair, record dimensions/dtype/
         mask values, class pixel distribution → reports/dataset_statistics.json
   │
Phase 2b compute the single global "bright SAR pixel" threshold T: sample up
         to 5,000,000 pixels (seed 0) across all sampled tiles, take the 90th
         percentile
   │
Phase 8  split the TILE list (not patches) into train/val/test (~70/15/15,
         seed 42) — val/test are both carved out of the official train tiles,
         since OEM's own val split has SAR images but no label masks
   │
   ▼  for every tile: split into four 512×512 patches (non-overlapping)
   │
   ▼  for every patch:
       Phase 2  whole-patch per-class pixel stats, dominant/2nd/smallest class
       Phase 2b per-class mean_sar_response / bright_pixel_fraction (using T)
       Phase 3  3×3 grid-cell stats (same per-class stats, per region)
       Phase 4-7 generate all 9 question categories, shuffle, cap at 22/image
       Phase 9  write the record to {split}.jsonl
       (optional) save a 3-panel SAR | label | QA-text visualisation PNG
   │
   ▼
Phase 7  save category/class/region distribution reports
Phase 10 validate: file existence, category validity, conversation shape,
         duplicate IDs/questions, bbox validity, cross-split ID leakage →
         reports/validation_report.json (exit 1 on failure)
```

### 7.2 Question categories (9, up to 22 per image)

- **global_classification**: dominant class, a duplicate-phrasing "largest
  area" question, which classes are visible, how many distinct classes are
  present, and Yes/No presence for up to 2 random classes.
- **global_quantitative**: percentage-range coverage of the dominant class
  and one random non-dominant present class; up to 2 random class-pair "which
  occupies more area" comparisons (only if the gap is real — see thresholds);
  "what are the two most prevalent classes."
- **regional_vqa**: per grid cell, dominant class and which classes are
  present there; plus, for up to 3 random classes, "which region has the
  highest proportion of X" — answered with a human-readable merged region
  name when multiple cells are close to tied (see `delta_fraction` below).
- **region_grounding**: random free-form boxes (not grid-aligned) — dominant
  class in the box, and a one-sentence description of what's in it.
- **comparative_spatial**: "is there more X in region A or B," and a Yes/No
  variant "is X more prevalent in A than B" — both gated by a minimum
  proportion gap.
- **sar_observation**: which class has the strongest/weakest average SAR
  return and the highest/lowest bright-pixel fraction (whole image); plus,
  per up to 3 random classes, its exact average SAR response and bright-pixel
  percentage.
- **sar_comparative**: pairwise "does A or B have the stronger SAR return,"
  and "which has more bright pixels" — both gated by minimum-gap thresholds.
- **sar_bbox_variance**: random boxes — dominant class plus a heterogeneity
  label (low/moderate/high) from the pixel variance inside the box.
- **surrounding_classes**: for up to 2 regions, what classes are directly
  adjacent (8-connected) to that region's dominant class, with each
  neighbour's share of the boundary.

### 7.3 How the thresholds were chosen

Same three flavours as Part A — a minimum-gap rule to avoid asking
unanswerable-close comparisons, a presence floor so near-zero traces of a
class don't count, and one set of fixed bins. Values, from `config.yaml`
unless noted as hardcoded:

| Threshold | Value | Why |
|---|---|---|
| `present_min_proportion` | 0.5% | A class only counts as "present" in a region/image, or gets `sar_class_facts` computed at all, once it covers ≥0.5% of the patch's pixels — below that, it's skipped entirely rather than reported as a near-zero number. |
| `dominant_margin` | 5 pts | The top class must beat the 2nd-ranked class by ≥5 percentage points to be called "dominant" with real confidence (the generator still answers with the top class either way — this margin isn't a hard gate, more a documented confidence bar). |
| `comparison_min_diff` | 2 pts | Minimum proportion gap required before asking any "which has more area" question (global_quantitative and both comparative_spatial questions). |
| `regional_vqa.delta_fraction` | 10 pts | When answering "which region has the most X," every grid cell within 10 percentage points of the single best cell is folded into the answer together (so a near-tie across 2–3 cells gets a merged, honest answer like "the top region" instead of arbitrarily picking one). |
| `sar.bright_pixel_percentile` | 90th | Defines what counts as a "bright" SAR pixel at all — same percentile-based reasoning as Part A: a relative cutoff within this dataset's own pixel range, not a physically calibrated value. Computed once, dataset-wide, from up to 5,000,000 sampled pixels. |
| `sar.sar_mean_comparison_min_diff` | 5.0 (0–255 scale) | Minimum average-SAR-response gap before asking "which has the stronger return" — "prevents trivial comparisons" per the config comment. |
| `sar.sar_bright_fraction_min_diff` | 5 pts | Same idea, for the bright-pixel-fraction comparison. |
| `region_grounding` box size | 8%–30% of image | Random free-form box size range for region_grounding questions. |
| `sar_bbox_variance` box size | 10%–35% of image | Larger than region_grounding's box range — larger boxes give a more stable variance estimate. |
| `surrounding_classes.max_questions` | 2 | Config comment: "as requested by the user, limit to 2 regions." |
| SAR variance heterogeneity bins | var < 50 → "low", 50 ≤ var < 150 → "moderate", var ≥ 150 → "high" | **Hardcoded directly in `generate_questions.py`, not in config** — no comment explaining how 50/150 were picked (unlike Part A's `sar_bbox_variance`, which derived its bins from actual data quantiles via the `inspect` stage, this pipeline's bins look like fixed guesses). |
| `concentrated_threshold` (0.40) | — | Defined in the config with a comment ("≥40% in a region = concentrated there") but **not referenced anywhere in the code** — a leftover/unused setting. |
| `np.random.seed(0)` for threshold sampling | hardcoded | A separate, fixed seed just for subsampling pixels when computing the bright-pixel threshold T, independent of the pipeline's main `random_seed: 42`. |

The percentage-range bins for `global_quantitative` answers are fixed and
not something a class-specific threshold choice: <5%, 5–15%, 15–30%, 30–50%,
50–70%, 70–90%, >90%.

### 7.4 Land-cover classes (8, 1-indexed mask values)

| ID | Class |
|---|---|
| 1 | Bareland |
| 2 | Rangeland |
| 3 | Developed Space |
| 4 | Road |
| 5 | Tree |
| 6 | Water |
| 7 | Agriculture Land |
| 8 | Building |

No merging/remapping — all 8 raw mask values are used as-is everywhere
(global, regional, SAR, and adjacency facts all share this same class map).

### 7.5 Tiles and patches

Each 1024×1024 source tile (SAR + label mask) is split into four
non-overlapping 512×512 patches — `(row,col)` → `top_left`/`top_right`/
`bottom_left`/`bottom_right`, with a `p{row}{col}` suffix (`p00`, `p01`,
`p10`, `p11`) appended to the tile's filename to form each patch's ID (e.g.
`TrainArea_001_p00`). All facts and questions are computed on the 512×512
patch, matching the model's 512 input size directly — no resampling factor
involved, since native resolution is already 0.5 m/px at that patch size.

### 7.6 Counts / splits

Splitting happens at the **tile** level (`random.Random(seed=42)`, shuffled),
not the patch level: ~15% of tiles → test, ~15% → val, remaining ~70% →
train, each tile then contributing its 4 patches to whichever split it
landed in. Both val and test are carved out of OEM's own `train/` tiles
(OEM's real `val/` split has SAR imagery but no label masks, so it can't be
used to compute ground truth facts).

A previously-documented real run (per `DATA_QUESTIONTYPES.md`): 17,328 total
patches (~4,332 tiles × 4), split 12,136 / 2,596 / 2,596 (train/val/test),
producing 381,216 QA pairs total (266,992 / 57,112 / 57,112) — exactly 22
per patch, the configured max.

### 7.7 `ground_truth_facts` schema

```
image_id, original_tile, patch_suffix, patch_position,
dominant_class, classes_present, n_classes_present,
class_proportions: { class: proportion, ... }        # present classes only
sar_class_facts: { class: { mean_sar_response, bright_pixel_fraction }, ... }
# added only for region/bbox-based question categories:
region, region_dominant, region_proportions           # grid-region questions
bbox                                                   # free-form-box questions
```

`sar_class_facts` only includes classes meeting `present_min_proportion`
(0.5%); `mean_sar_response` is the plain average raw 8-bit SAR pixel value
for that class's pixels (explicitly documented as a relative statistic, not
a calibrated physical radar measurement), and `bright_pixel_fraction` is the
share of that class's pixels at or above the global bright threshold T.
