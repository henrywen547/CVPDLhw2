# Q4 validation figure procedure

The user supplied the original slide wording: “Q4 (10%): Visualize 8 images from the validation set with the predicted and ground-truth labels.” Earlier derived handoff guidance broadly said not to read `data/` independently. For this report requirement, we apply a narrow **post-evaluation validation-only** exception: select four correct and four incorrect images using the already completed official Tent `val_pred.csv` and validation ground truth, then open only those eight validation PNGs to create the report figure. This does not affect the adaptation method, its validation result, or the single official test pass. No test files or labels are read.

The report-only script is `scripts/build_q4_figure.py`. It uses the order in the official prediction CSV, picks one correct and one incorrect candidate near each quarter of the validation stream when available, prefers different true classes, and writes an eight-panel PNG plus JSON provenance. Its predictions come from the official Tent validation run at commit `b6f2a6d886259421a1a53089dbfcc838f35f0083`; validation labels are opened **only after** that run, outside `starter/`. The script was checked on synthetic files before actual validation use.

From `CVPDLhw2`:

```bash
../.venv/bin/python scripts/build_q4_figure.py --allow-val-files-for-q4 --pred runs/tent_lr001/val_pred.csv --val-root ../data/val --classes starter/classes.txt --out runs/q4_tent/q4_examples.png
```

Keep the generated figure local and include it in the final report PDF. Do not put `data/` or validation PNGs into the submission ZIP.

`scripts/collect_report_materials.py` gathers the figure, its metadata, the report text, Q3 CSV, and copies of only the eight selected source PNGs into ignored `runs/report_materials/` for the user's own layout work. The individual PNGs are local source material; they are not extra files for the COOL ZIP.
