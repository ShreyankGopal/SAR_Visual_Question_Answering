# sar_vqa: shared SAR VQA generation pipeline

One core, one adapter and one config per dataset.

    run.py                 python run.py --config configs/<name>.yaml --stage adapt|facts|inspect|build|validate|all
    configs/<name>.yaml    paths, classes, categories per source, thresholds, balancing, GSD
    adapters/<name>.py     raw dataset files -> common clean JSONL (only dataset-specific code)
    core/                  geometry, io, facts, generators, balance, build, validate, inspect_data, spot_check
    tools/                 compare_qa.py, compare_clean.py (regression checks)

Clean JSONL record (adapter output, core input):

    {"dataset", "image" (relative to data.root), "split" (train/val/test), "source",
     "width", "height", "native_gsd_m", "band", "polarization", "scenario",
     "objects": [{"category", "hbox": [x1, y1, x2, y2] px, "rbox": [cx, cy, w, h, angle] or null}]}

Every QA record carries gsd_m (at pipeline.model_input_size), native_gsd_m and resample_factor.

Adding a dataset: write adapters/<name>.py, copy a config, run adapt -> facts -> inspect,
set thresholds, then build -> validate -> spot-check.

