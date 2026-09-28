# SAR VQA: question types per dataset

Pipeline: `DataGen/sar_vqa` (shared core, one adapter + config per dataset).

## Overview

| Category | OEM (reference) | SIVED | OGSOD | HRSID | SAR-AIRcraft |
|---|---|---|---|---|---|
| global_classification | 42,260 | ✗ (single class) | 41,953 | ✗ (single class) | ✗ (aircraft types not visually verifiable) |
| object_classification | n/a | ✗ (single class) | 8,311 | ✗ (single class) | ✗ (aircraft types not visually verifiable) |
| object_counting | n/a | 944 | 16,098 | 5,139 | 10,497 |
| regional_vqa | 144,961 | 1,872 | 26,444 | 7,370 | 16,192 |
| comparative_spatial | 19,250 | 1,780 | 54,273 | 16,271 | 34,319 |
| region_grounding | 42,200 | 1,975 | 31,349 | 15,210 | 38,072 |
| object_relations | n/a | 1,655 | 16,216 | 4,472 | 14,156 |
| orientation | n/a | 2,548 | ✗ (no angles) | 6,866 | ✗ (no angles) |
| relative_size | n/a | 2,414 | ✗ (boxes too small) | 3,423 | 1,414 |
| absolute_size | n/a | ✗ (mostly all in the same size bin) | 4,451 | 4,817 | ✗ (diagonal boxes with the same object appear longer although they aren't) |
| arrangement | n/a | 228 | ✗ (no angles) | ✗ (too few parallel groups) | ✗ (no angles) |
| global_quantitative | 29,295 | 707 | 14,283 | ✗ (ships ~0.1% of a tile) | 13,551 |
| sar_comparative | 16,911 | 2,618 | 12,425 | 3,673 | 1,611 |
| sar_observation | 60,274 | ✗ (single class) | 689 | ✗ (single class) | ✗ (single class) |
| sar_bbox_variance | 13,926 | ✗ (mostly dark background only object is bright) | 36,406 | 10,306 | 27,104 |
| surrounding_classes | 12,139 | ✗ (no land-cover map) | ✗ | ✗ | ✗ |
| **Total** | **381,216** | **16,741** | **262,898** | **77,547** | **156,916** |

**All five SAR datasets: 895,318.**
## Metdata

| | OEM | SIVED | OGSOD | HRSID | SAR-AIRcraft |
|---|---|---|---|---|---|
| Images | 17,328 patches of OpenEarthMap | 1,044 | 18,331 | 5,495 (P0137 dropped) | 13,552 tiles from 4,368 images |
| Image size | 512 | 512 | 256 | 800 | 512 tiles (from 800–1,500) |
| Classes | 8 land-cover | Vehicle | Bridge, Harbour, Tank | Ship | Aircraft (7 types kept as metadata) |
| Box type | segmentation mask | rotated | horizontal | polygon → rotated | horizontal |
| Native GSD | 0.5m | 0.1 / 0.3 m | 3 m | 0.5 / 1 / 3 m | 0.5 m (stated resolution 1 m) |
| gsd_m at 512 input | 0.5m | 0.1 / 0.3 m | 1.5 m | 0.78 / 1.56 / 4.69 m | 0.5 m |
| Splits | carved from train (12,136 / 2,596 / 2,596 images) | official | official test, val carved | re-split by panorama | official, tiled within each image |


---

## OEM (OpenEarthMap)

| Category | Example question → answer | Train | Val | Test | Total |
|---|---|---|---|---|---|
| global_classification | Is agriculture land present in this image? → No. | 29,568 | 6,346 | 6,346 | 42,260 |
| global_quantitative | Approximately what percentage of this image is covered by bareland? → about 50–70%. | 20,387 | 4,378 | 4,530 | 29,295 |
| comparative_spatial | Is there more rangeland in the center or bottom right region? → The bottom right region. | 13,499 | 2,890 | 2,861 | 19,250 |
| regional_vqa | Which land-cover classes are present in the bottom right region? → Rangeland, Developed Space, and Road. | 101,442 | 21,877 | 21,642 | 144,961 |
| region_grounding | What is the dominant land-cover type in the region [0.6756, 0.0000, 0.8989, 0.2041]? → Developed Space. | 29,660 | 6,387 | 6,153 | 42,200 |
| surrounding_classes | What land-cover classes surround the dominant class in the top right region? → Rangeland; adjacent: Developed Space (62.9%), Road (37.1%). | 8,525 | 1,818 | 1,796 | 12,139 |
| sar_comparative | Does Bareland or Road have the stronger average SAR response in this image? → Road. | 11,831 | 2,424 | 2,656 | 16,911 |
| sar_observation | Which land-cover class contains the lowest fraction of bright SAR pixels? → Bareland. | 42,305 | 8,889 | 9,080 | 60,274 |
| sar_bbox_variance | Given the region [0.1982, 0.7001, 0.3473, 0.9973], what is the dominant class and how heterogeneous is the SAR response? → Rangeland; variance 894.51, high heterogeneity. | 9,775 | 2,103 | 2,048 | 13,926 |
| **Total** | | **266,992** | **57,112** | **57,112** | **381,216** |

- 17,328 image patches (12,136 / 2,596 / 2,596), exactly 22 QA pairs per image.
- Answer balance (not balanced the way the SAR sets are): sar_comparative answers lean to Building (29%), then Tree, Rangeland and Developed Space (17–19% each); global_classification Yes/No answers are No 18% vs Yes 15% of the category.
- Free-text answers: sar_bbox_variance (13,708 distinct answers, exact variance values), surrounding_classes (8,371 distinct) and sar_observation (1,438 distinct) are mostly unique strings, so they need text metrics (BLEU/ROUGE) rather than exact match - LETS SEE IF WE CAN MODIFY THIS .

---

## SIVED (

| # | Category | Question template → example answer | Applies to | How the answer is decided | QA pairs |
|---|---|---|---|---|---|
| 1 | global_quantitative | Approximately what percentage of this image is covered by vehicles? → About 2–5%. | FARAD, MiniSAR | Area of all rotated vehicle boxes ÷ image area, binned: <2%, 2–5%, 5–10%, >10% | 707 |
| 2 | sar_comparative | Which vehicle has the stronger average SAR return: the one centred at [x, y] or the one centred at [x, y]? → The one centred at [x, y]. | FARAD, MiniSAR, MSTAR | Mean pixel intensity inside each rotated box; pair used only if the difference is ≥15 (8-bit scale) | 2,618 |
| 3 | comparative_spatial | Are there more vehicles in the center region or the bottom right region? → The bottom right region. | FARAD, MiniSAR | Vehicles counted per 3×3 grid cell by box centre; pair used only if counts differ by ≥2 | 1,780 |
| 4 | region_grounding | Is there a vehicle in the region [x1, y1, x2, y2]? → Yes. / Give the bounding box of the vehicle closest to the image center. → [0.4312, 0.4705, 0.5498, 0.5211]. | Presence: all three. Referring: FARAD, MiniSAR | Presence: Yes if a vehicle centre is in the box, No if no vehicle touches it, partial overlaps skipped. Referring cues: closest to centre, closest to top, longest body; used only when one vehicle clearly wins | 1,975 |
| 5 | regional_vqa | Are there any vehicles in the top left region? → No. | FARAD, MiniSAR | Yes if a vehicle centre is in the grid cell, No if no part of any vehicle is in it, otherwise skipped | 1,872 |
| 6 | object_relations | In which direction is the nearest vehicle from the vehicle centred at [x, y]? → The nearest vehicle is to its right. | FARAD, MiniSAR | 8 directions between box centres; used only if the second-nearest vehicle is ≥1.2× farther and the direction is ≥7.5° from a sector boundary | 1,655 |
| 7 | orientation | What is the orientation of the vehicle centred at [x, y]? → Diagonal, top-left to bottom-right. | FARAD, MiniSAR, MSTAR | Long-edge angle of the rotated box: horizontal, vertical or one of two diagonals. This is the vehicle's axis, not its heading. Skipped if within 5° of a bin edge or near-square | 2,548 |
| 8 | relative_size | Which vehicle is longer: the one centred at [x, y] or the one centred at [x, y]? → The one centred at [x, y]. | FARAD, MiniSAR, MSTAR | Long edge of the rotated boxes; pair used only if the ratio is ≥1.2 | 2,414 |
| 9 | arrangement | Are the vehicles in this image mostly aligned in the same direction? → Yes. | FARAD, MiniSAR | Alignment score over all vehicle angles (1 = parallel); Yes ≥0.9, No ≤0.6, in-between skipped; needs ≥4 vehicles | 228 |
| 10 | object_counting | How many vehicles are in this image? → 11. / … in the top left region? → 3. / … in the region [x1, y1, x2, y2]? → 2. | FARAD, MiniSAR | Exact count by box centre; regions skipped if any vehicle straddles the border | 944 |
| | **Total** | | | | **16,741** |

- Coordinates are normalised to [0, 1]. Vehicles are referred to by centre point [x, y] in categories 2, 6, 7 and 8, so the numbers in the question don't reveal the vehicle's shape or size. Boxes [x1, y1, x2, y2] are used only where the box is the point of the question (4 and 10).
- Answers are balanced per sensor: Yes/No 50/50; A/B answers 50/50 first/second; no orientation answer above 35%; zero counts capped at 20%; at most 22 questions per image.
- MSTAR vehicles are pasted on a fixed 4×4 grid (always 16), so position- and count-based questions (1, 3, 5, 6, 9, 10 and referring grounding) are excluded for it.
- GSD: native 0.1 m for FARAD and MiniSAR, 0.3 m for MSTAR; images are already 512, so resample_factor = 1.0.
- Splits: official train/valid/test; 13,394 / 1,680 / 1,667 QA pairs.
- Cleaning: 481 XML box angles corrected from the DOTA polygons.

---

## OGSOD (done)

| # | Category | Question template → example answer | Applies to | How the answer is decided | QA pairs |
|---|---|---|---|---|---|
| 1 | global_classification | Is there a harbour in this image? → No. / What types of objects are in this image? → Bridge and Tank. | Bridge, Harbour, Tank | Presence from the annotations; Yes/No balanced by asking about absent classes | 41,953 |
| 2 | object_classification | What type of object is centred at [x, y]? → Tank. | Bridge, Harbour, Tank | Class of the object at that point; point used only if no other centre is within 6 px; no class above 40% | 8,311 |
| 3 | object_counting | How many tanks are in this image? → 12. / … in the top left region? → 3. / … in the region [x1, y1, x2, y2]? → 2. | Classes present in the image | Exact count by box centre; regions skipped if an object of that class straddles the border; zero answers capped at 20% | 16,098 |
| 4 | regional_vqa | Are there any bridges in the bottom right region? → Yes. / Which object types are in the center region? → Tank. | Classes present in the image | Yes if an object centre is in the grid cell, No if no part of that class is in it, otherwise skipped; one "types" question per image | 26,444 |
| 5 | comparative_spatial | Are there more tanks in the top left region or the center region? → The center region. | Bridge, Harbour, Tank | Objects counted per 3×3 cell by box centre; pair used only if counts differ by ≥1 (bridge, harbour) or ≥2 (tank) | 54,273 |
| 6 | region_grounding | Is there a tank in the region [x1, y1, x2, y2]? → Yes. / Give the bounding box of the harbour. → [0.2411, 0.4357, 0.6109, 0.6150]. / What type of object is in the region [x1, y1, x2, y2]? → Bridge. | Bridge, Harbour, Tank | Presence: Yes if a centre of that class is inside, No if none touches the box, half the boxes placed over an object. Referring: only when exactly one object of that class exists. Class-in-region: random box, not an object's own box | 31,349 |
| 7 | absolute_size | Approximately what is the diameter of the tank centred at [x, y]? → About 18–27 m. / … the longest side of the harbour …? → About 200–400 m. | Tank, Harbour | Box size × 3 m native GSD. Tank: mean box side, bins <18 / 18–27 / >27 m. Harbour: longest side, bins <100 / 100–200 / 200–400 / >400 m. Skipped within 1.5 m of a bin edge or when clipped by the image border | 4,451 |
| 8 | object_relations | In which direction is the nearest tank from the bridge centred at [x, y]? → The nearest tank is below and to its left. | Bridge, Harbour, Tank | 8 directions between box centres; used only if the second-nearest object is ≥1.2× farther and the direction is ≥7.5° from a sector boundary | 16,216 |
| 9 | global_quantitative | Approximately what percentage of this image is covered by bridges, harbours and tanks? → About 1–2%. / … covered by harbours? → About 5–20%. | All objects; Harbour separately | Area of all boxes ÷ image area, bins <1 / 1–2 / 2–5 / 5–20 / >20%; no answer above 35% | 14,283 |
| 10 | sar_comparative | Which tank has the stronger average SAR return: the one centred at [x, y] or the one centred at [x, y]? → The one centred at [x, y]. | Same-class pairs only | Mean pixel intensity inside each box; difference ≥20 (8-bit scale) | 12,425 |
| 11 | sar_observation | Which object type in this image has the lowest (or highest) fraction of bright SAR pixels? → Harbour. | Multi-class images only | Share of box pixels above the dataset's 90th-percentile brightness (216); lowest and second-lowest (or highest) must differ by ≥0.05 | 689 |
| 12 | sar_bbox_variance | How heterogeneous is the SAR response in the region [x1, y1, x2, y2]? → High heterogeneity. | Any region | Pixel variance in a random box, bins Low <1,429 / Medium 1,429–2,431 / High >2,431 (OGSOD tertiles) | 36,406 |
| | **Total** | | | | **262,898** |

- Splits: 12,830 / 1,834 / 3,667 images (train / val / test), giving 184,001 / 26,297 / 52,600 QA pairs. Val is 12.5% of the official train split, stratified by which classes each image contains; the official test set is unchanged.
- Coordinates normalised to [0, 1]; categories 2, 7, 8 and 10 refer to objects by centre point so box size can't reveal the class or size.
- Dropped: orientation and arrangement (horizontal boxes only), relative_size (boxes too small).
- GSD: native_gsd_m 3.0, gsd_m 1.5 at 512 input, resample_factor 2.0.
- Cleaning: 40 degenerate boxes (<2 px) dropped, 253 boxes clipped to the image edge.
- Caveats: tank sizes span a few pixels (±1 px ≈ ±3 m); harbour coverage is overstated by horizontal boxes; sar_observation is small because only 581 images contain more than one class.

---

## HRSID 

| # | Category | Question template → example answer | Applies to | How the answer is decided | QA pairs |
|---|---|---|---|---|---|
| 1 | object_counting | How many ships are in this image? → 4. / … in the top left region? → 1. / … in the region [x1, y1, x2, y2]? → 2. | All sensors | Exact count by box centre; regions skipped if a ship straddles the border; zero answers capped at 20% | 5,139 |
| 2 | regional_vqa | Are there any ships in the bottom right region? → No. | All sensors | Yes if a ship centre is in the grid cell, No if no ship pixel is in it, otherwise skipped; balanced Yes/No | 7,370 |
| 3 | comparative_spatial | Are there more ships in the center region or the top left region? → The top left region. | All sensors | Ships counted per 3×3 cell by box centre; pair used if counts differ by ≥1 | 16,271 |
| 4 | region_grounding | Is there a ship in the region [x1, y1, x2, y2]? → Yes. / Give the bounding box of the ship closest to the image center. → [0.3643, 0.4987, 0.4376, 0.5707]. | All sensors | Presence: Yes if a ship centre is inside, No if none touches the box, half the boxes placed over a ship, balanced. Referring cues: closest to centre, closest to top, longest ship; only when one ship clearly wins; ships under 10 px never referred to | 15,210 |
| 5 | object_relations | In which direction is the nearest ship from the ship centred at [x, y]? → The nearest ship is to its left. | All sensors | 8 directions between centres; second-nearest ≥1.2× farther; ≥7.5° from a sector edge | 4,472 |
| 6 | orientation | What is the orientation of the ship centred at [x, y]? → Diagonal, top-left to bottom-right. | All sensors | Long-edge angle of the rotated box fitted to the ship polygon; skipped within 5° of a bin edge, aspect < 1.3, or truncated; no answer above 35% | 6,866 |
| 7 | relative_size | Which ship is longer: the one centred at [x, y] or the one centred at [x, y]? → The one centred at [x, y]. | All sensors | Rotated-box length; ratio ≥1.2; truncated ships skipped | 3,423 |
| 8 | absolute_size | Approximately how long is the ship centred at [x, y]? → About 100–200 m. | All sensors | Rotated-box length × native GSD (0.5 / 1 / 3 m); bins <50 / 50–100 / 100–200 / >200 m; skipped within 3 m of a bin edge, truncated, or touching the border; no answer above 40% | 4,817 |
| 9 | sar_comparative | Which ship has the stronger average SAR return: the one centred at [x, y] or the one centred at [x, y]? → The one centred at [x, y]. | All sensors | Mean intensity in the rotated box; difference ≥20 (8-bit). Brighter ship is also the longer one in only 48–63% of pairs, so not a size proxy | 3,673 |
| 10 | sar_bbox_variance | How heterogeneous is the SAR response in the region [x1, y1, x2, y2]? → Low heterogeneity. | All sensors | Pixel variance in a random box; bins Low <37.3 / Medium 37.3–125.1 / High >125.1 (HRSID tertiles); no answer above 40% per sensor | 10,306 |
| | **Total** | | | | **77,547** |

- Splits (re-split by whole panorama): 3,960 / 489 / 1,046 images, 11,774 / 1,628 / 3,391 ships, 55,818 / 7,044 / 14,685 QA pairs (train / val / test). The official split was not used: 136 of 137 panoramas were in both train and test, and 5,726 train/test tile pairs overlapped (tiles overlap by 200 px). High-res panoramas assigned by hand: 0.5 m P0124, P0125, P0130 train, P0131 test; 1 m P0128 train, P0123 test.
- `source` is the sensor (Sentinel-1B, TerraSAR-X, TanDEM-X), so balancing runs per sensor. TanDEM-X is one test-only panorama (342 QA pairs): per-sensor results for it are indicative only.
- GSD: native 0.5 / 1 / 3 m from HRSID Table 2 by panorama prefix; gsd_m at 512 input = native × 1.5625 (0.78 / 1.56 / 4.69 m); resample_factor 0.64 (a real downsample). Ship lengths confirm the GSDs (median 134 m at 3 m, 77 m at 1 m, 81 m at 0.5 m).
- Coordinates normalised to [0, 1]; categories 5–9 refer to ships by centre point.
- Cleaning: P0137 (109 tiles, not in Table 2) dropped; 12 unusable polygons dropped; 2,005 ships (12%) touching the tile border flagged truncated and skipped for shape and size questions.
- Dropped: global_quantitative (ships ~0.1% of a tile), arrangement (few parallel groups); global_classification, object_classification and sar_observation not applicable (single class).
- Caveat: "less than 50 m" answers come mostly from 0.5–1 m imagery (small boats are rarely annotated at 3 m), so report absolute_size per GSD.

---

## SAR-AIRcraft-1.0 

| # | Category | Question template → example answer | Applies to | How the answer is decided | QA pairs |
|---|---|---|---|---|---|
| 1 | object_counting | How many aircraft are in this image? → 3. / … in the top left region? → 1. / … in the region [x1, y1, x2, y2]? → 2. | All tiles | Exact count by box centre; regions skipped if an aircraft straddles the border; zero answers capped at 20% | 10,497 |
| 2 | regional_vqa | Are there any aircraft in the center region? → Yes. | All tiles | Yes if an aircraft centre is in the grid cell, No if no aircraft pixel is in it, otherwise skipped; balanced Yes/No | 16,192 |
| 3 | comparative_spatial | Are there more aircraft in the top left region or the center region? → The center region. | All tiles | Aircraft counted per 3×3 cell by box centre; pair used if counts differ by ≥1 | 34,319 |
| 4 | region_grounding | Is there an aircraft in the region [x1, y1, x2, y2]? → No. / Give the bounding box of the aircraft closest to the image center. → […]. / … of the largest aircraft. → […]. | All tiles | Presence: Yes if an aircraft centre is inside, No if none touches the box, half the boxes placed over an aircraft, balanced. Referring: closest to centre, closest to top, largest (size ratio ≥1.45); only when one aircraft clearly wins | 38,072 |
| 5 | object_relations | In which direction is the nearest aircraft from the aircraft centred at [x, y]? → The nearest aircraft is to its left. | All tiles | 8 directions between centres; second-nearest ≥1.2× farther; ≥7.5° from a sector edge | 14,156 |
| 6 | global_quantitative | Approximately what percentage of this image is covered by aircraft? → About 2–5%. | All tiles | Area of all aircraft boxes ÷ tile area, bins <2 / 2–5 / 5–10 / >10%; no answer above 35% | 13,551 |
| 7 | sar_bbox_variance | How heterogeneous is the SAR response in the region [x1, y1, x2, y2]? → Medium heterogeneity. | All tiles | Pixel variance in a random box; bins Low <60.4 / Medium 60.4–545.5 / High >545.5 (tertiles on raw tiles); no answer above 40% | 27,104 |
| 8 | sar_comparative | Which aircraft has the stronger average SAR return: the one centred at [x, y] or the one centred at [x, y]? → The one centred at [x, y]. | Tiles with ≥2 aircraft | Mean intensity in each box; difference ≥15 (8-bit). Brighter aircraft is the larger one in only 34% of pairs, so not a size proxy. Small because raw aircraft intensities differ little | 1,611 |
| 9 | relative_size | Which aircraft is larger: the one centred at [x, y] or the one centred at [x, y]? → The one centred at [x, y]. | Tiles with ≥2 aircraft | Longest side of the horizontal boxes; only when the ratio is ≥1.45, which a heading change alone (≤1.41×) cannot produce; truncated aircraft skipped | 1,414 |
| | **Total** | | | | **156,916** |

- Splits: official train / val / test lists kept; tiles cut within each image so no tile crosses splits. 9,686 / 1,371 / 2,495 tiles and 112,060 / 16,020 / 28,836 QA pairs (train / val / test). Checked for leakage: consecutive images overlap no more than random pairs.
- Tiling: 512 px tiles at native resolution, evenly spaced with overlap (800 and 1,000 px images 2×2, 1,200 and 1,500 px 3×3). Raw pixel values kept (no contrast stretch), consistent with the other datasets. Tiles without aircraft capped at 15% (2,033 kept) for genuine "0" / "No" answers.
- Aircraft cut by a tile edge kept if ≥25% visible and flagged truncated (6,762 of 24,858 aircraft instances; 16,463 unique aircraft appear in several overlapping tiles).
- GSD: stated resolution 1 m, but known aircraft sizes imply ~0.5 m per pixel (e.g. Boeing 737 ~85 px, ARJ21 ~66 px), so native_gsd_m = 0.5, gsd_m = 0.5, resample_factor 1.0; the stated value is kept as resolution_m. Median aircraft size at 0.5 m: 38.5 m.
- One class "aircraft": the 7 fine-grained types (A220, A320/321, A330, ARJ21, Boeing 737, Boeing 787, other) could not be told apart by eye, so they are kept only as metadata in clean.jsonl.
- Dropped: object/global classification (types not visually verifiable), orientation and arrangement (no angles), absolute_size (a diagonal aircraft's horizontal box is up to 1.41× larger), sar_observation (single class).
- Coordinates normalised to [0, 1]; categories 5, 8 and 9 refer to aircraft by centre point; aircraft under 10 px never referred to.