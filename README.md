# CVPDL 2026 HW2

This repository tracks the official starter and validation experiments. The original assignment files, `data/`, and `source_resnet50_160.pth` are one directory above this repository. Do not read `data/` directly; use the official `starter/run.py` and `starter/harness.py`.

## Environment

- Host: Ubuntu 24.04, Python 3.12.3, NVIDIA GeForce GTX 1080 Ti (11 GiB), driver 580.178.04.
- Runtime: PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.4.4, pandas 2.3.3, Pillow 11.3.0. CUDA is available, GPU compute capability 6.1.
- The local virtual environment is `../.venv`, as requested. It currently uses a `.pth` link to packages already installed at `/home/weal222c/71hw/part1/myvenv/lib/python3.12/site-packages`. This is local reuse, not a portable installation. For a new machine, install the versions in `requirements.txt` from a compatible PyTorch CUDA wheel index and verify `torch.cuda.is_available()`.
- Re-run timing and VRAM checks on an RTX 3090 before submission. The present GPU is different from the grading GPU.

## Baseline

From this repository root:

```bash
cd starter
../../.venv/bin/python run.py --method source --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/source_val --seed 0
```

`source` runs in evaluation mode and uses `torch.no_grad()`. The official harness writes `val_pred.csv`, `val_run_log.json`, and prints validation accuracy. The run uses the official batch size of 64 and ImageNet mean/std normalization. `harness.py` remains unchanged.

On the GTX 1080 Ti, the baseline achieved **0.5454545 Top-1 (1,152/2,112)** with 2.7 seconds of harness stream time and 706 MiB sampled peak VRAM. These values came from starter commit `0b26fb09acd2749098eb20222d0cc947f9e1034b`. The supplied checkpoint SHA-256 is `8ec867598014a03c00237f61145d311ff245d8abeb650c06e070866efa48363f`.

## Validation diagnostics

The optional `source_diag` method receives the same normalized batches from the official harness and writes prediction and image summaries without labels or IDs. It is for validation exploration only; it adds logging overhead and should not be used for final timing or test submission.

```bash
cd starter
CVPDL_DIAG_JSONL=../runs/source_diag/image_summaries.jsonl ../../.venv/bin/python run.py --method source_diag --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/source_diag --seed 0
cd ..
../.venv/bin/python scripts/analyze_diagnostics.py --diagnostics runs/source_diag/image_summaries.jsonl --baseline-pred runs/source_val/val_pred.csv --diagnostic-pred runs/source_diag/val_pred.csv --out runs/source_diag/summary.json
../.venv/bin/python scripts/permutation_adjacent.py --diagnostics runs/source_diag/image_summaries.jsonl --out runs/source_diag/permutation_adjacent.json --reps 10000 --seed 0
```

The analysis verifies that diagnostic predictions exactly match the baseline. The findings and limitations are in [docs/data_exploration.md](docs/data_exploration.md).

## Online adaptation comparisons

`tent` and `tent_come` each make one forward pass and one SGD update per served batch. They return the current batch's predictions from **before** that update, so the next batch is the first to use newly adapted parameters. Only BatchNorm2d affine weight and bias receive gradients; batch statistics come from the current batch. Both methods start from the supplied checkpoint on each invocation. The defaults below are local experiment settings, not claims about the authors' published hyperparameters.

```bash
cd starter
../../.venv/bin/python run.py --method bn_batch --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/bn_batch --seed 0
../../.venv/bin/python run.py --method tent --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/tent_lr001 --seed 0 --tta-lr 0.001 --tta-momentum 0.9
../../.venv/bin/python run.py --method tent_come --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/come_lr001 --seed 0 --tta-lr 0.001 --tta-momentum 0.9 --come-k 1000
../../.venv/bin/python run.py --method tent_gate --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/tent_gate_040 --seed 0 --tta-lr 0.001 --tta-momentum 0.9 --max-class-share 0.4
../../.venv/bin/python run.py --method tent_gate --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/tent_gate_025_timed --seed 0 --tta-lr 0.001 --tta-momentum 0.9 --max-class-share 0.25
../../.venv/bin/python run.py --method gold_sf --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/gold_sf_r8 --seed 0 --tta-lr 0.01 --tta-momentum 0.9 --gold-rank 8 --gold-alpha 0.1 --gold-confidence-threshold 0.5
```

`bn_batch` isolates the effect of using current-batch BatchNorm statistics without parameter updates. `tent_gate` is a custom P2 ablation: it skips an update if the current batch's largest predicted class exceeds the selected threshold. Each threshold is fixed before its run and does not use batch index, image ID, or labels. For validation-only confidence diagnostics, set `CVPDL_DIAG_JSONL=../runs/<run-name>/diagnostics.jsonl` before invoking a TTA method; this adds logging overhead and should be left unset for timing or final testing. The comparison script reads only official output and optional diagnostic logs.

`gold_sf` is a source-free **GOLD-inspired subspace ablation**, not the full author method. It uses only the supplied classifier weights and the current batch's predictions to maintain a rank-8 classifier-sensitive subspace. It trains an 8-value scaling vector with entropy minimization while freezing the backbone. Its eigenbasis is refreshed from current-batch information without a batch-index schedule. See the compliance review for omitted GOLD components.

Paper code and compliance review: [docs/paper_method_review.md](docs/paper_method_review.md). Experiment outcomes are recorded in [docs/experiments.md](docs/experiments.md).
The same-code Tent diagnostic and bounded learning-rate round is recorded in [docs/tent_diagnostic_round.md](docs/tent_diagnostic_round.md). It selected lr .003 by official validation Top-1.

After the user reported a 0.72005 public score for Tent lr .003, [docs/round3_plan.md](docs/round3_plan.md) recorded a bounded multiview validation round. `bn_flip` averages original and horizontal-flip probabilities with current-batch BN; `tent_post` predicts the current batch after one Tent update. The complete results are in [docs/q3_results.csv](docs/q3_results.csv).

The bounded follow-up reached **1759/2112 (.832860)** with `bn_view10`, which averages original, horizontal-flip, and four fixed center-crop scales with flips. It uses current-view BN statistics and no optimizer update. Its separate full local test took 108.5 seconds and 706 MiB; that output is preserved but is not the current submission.

The ten-view paired candidate reached **1763/2112 (.834754)** on official validation in 19.0 seconds and 1164 MiB on the GTX 1080 Ti. The user reported public **0.76627** for it. The eight-view method had lower validation accuracy, **1722/2112 (.815341)**, but the user reported a slightly higher public **0.76822**. Both are preserved as earlier candidates.

[Round 4](docs/round4_unmix_plan.md) tested an independent normalization-only implementation of the UnMix-TNS paper's online BN statistics. It scored **1413/2112 (.669034)**, below BN-only, so it was not used for the current submission.

The [LAME-inspired round](docs/round4_lame_plan.md) tested parameter-free current-batch output correction on one-view BN and six-view BN. `lame_view6` reached **1772/2112 (.839015)** in 11.7 seconds and 808 MiB on official validation.

Its official full test completed in **66.2 seconds** and **808 MiB** on the GTX 1080 Ti. The user reported **public 0.81266**. Its log-verified Kaggle file is `runs/final_lame_view6/submission.csv` (SHA-256 `b60a57c286eb1eab643b3ac880f6e9a764495aeb5110f7cfb85525273a238398`); the root `submission.csv` and `runs/submission_staging/` now correspond to this method.

A [bounded view-count check](docs/round5_view_sensitivity.md) found an eight-view LAME-inspired candidate at **1827/2112 (.865057)** validation, compared with 1772/2112 for the current six-view version. Its separate full local test took **87.5 seconds / 848 MiB**. The verified upload file is `runs/final_lame_view8/submission.csv`; public score is pending. The root submission remains the six-view method until public feedback supports changing it.

## Official test artifacts and current submission

The earlier Tent lr .001 test pass is preserved at `runs/final_tent_001/`. It scored 1528/2112 on validation; its exact command, from `CVPDLhw2/starter` with `CVPDL_DIAG_JSONL` unset, was:

```bash
../../.venv/bin/python run.py --method tent --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_tent_001 --seed 0 --tta-lr 0.001 --tta-momentum 0.9
```

The official harness wrote matching `test_pred.csv` and `test_run_log.json` there. This earlier run completed 12,288 test images in 28.2 seconds with 3470 MiB peak VRAM on the GTX 1080 Ti; test accuracy is unavailable.

The bounded second validation round selected **Tent lr .003**, 1530/2112 on validation, for a separate official test pass from the original checkpoint. Its exact command was:

```bash
../../.venv/bin/python run.py --method tent --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_tent_003 --seed 0 --tta-lr 0.003 --tta-momentum 0.9
```

This earlier test run completed 12,288 images in 28.3 seconds with 3470 MiB peak VRAM on the GTX 1080 Ti. Its official CSV SHA-256 is `85d4d77e4b2876b158e75c844bdc2db857befacedb6f36b3c073cb429bb76775`, equal to its paired run log. The user reported a **0.72005 public score** for this artifact. The earlier lr .001 outputs remain intact; 166 test predictions differ between those Tent runs.

The earlier ten-view paired artifact was run from `CVPDLhw2/starter` at commit `a01bef5ec76176f6f9d8cfc5131e3268d87b5496`:

```bash
../../.venv/bin/python run.py --method bn_view10_pair --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_bn_view10_pair --seed 0
```

This official local test completed 12,288 images in **109.3 seconds** with **1164 MiB** peak VRAM on the GTX 1080 Ti. Its paired CSV/log hash is `02baf91683a615f0af84b568a4a6699529dc5f8965e9c998938c53103ddb70f1`. The local time exceeds 90 seconds; the grading limit is measured on RTX 3090, which is not available here, so time compliance is **unverified**. The user reported public **0.76627** for this artifact. Its CSV/log remain preserved.

The earlier `bn_view8` artifact was run from `CVPDLhw2/starter` at commit `57c209a056ecbba04d3d0cf000194d8fd196635d`:

```bash
../../.venv/bin/python run.py --method bn_view8 --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_bn_view8 --seed 0
```

It scored **1722/2112 (.815341)** on validation and completed the full local test in **87.3 seconds**, 706 MiB. Its paired official files are in `runs/final_bn_view8/`; the CSV SHA-256 is `9460c10feeebae78946484bbea9ad28fc43731aebdbe07059c7bb796930d8330`. The user reported public **0.76822**. These earlier files remain preserved.

The current `lame_view6` artifact was run from `CVPDLhw2/starter` at commit `c6d18e32b6206abb2a021797e64dc013b3579dcc`:

```bash
../../.venv/bin/python run.py --method lame_view6 --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_lame_view6 --seed 0
```

It scored **1772/2112 (.839015)** on validation and **public 0.81266** as reported by the user. The full local test took **66.2 seconds** and **808 MiB** peak VRAM. Its paired official CSV/log hash is `b60a57c286eb1eab643b3ac880f6e9a764495aeb5110f7cfb85525273a238398`. RTX 3090 runtime and private accuracy remain unmeasured.

To regenerate the Kaggle upload filename from the current `lame_view6` official run, without rerunning the test stream, run from the repository root:

```bash
../.venv/bin/python scripts/make_submission.py
```

The script verifies the paired run-log hash, `id,label` header, 12,288 unique IDs, and label range, then copies the exact CSV bytes to `submission.csv`. Use `--pred`, `--run-log`, and `--out` to select another official run or destination. See [docs/experiments.md](docs/experiments.md) for the full record and remaining submission items.

The Q4 report-only validation-image procedure is documented in [docs/q4_procedure.md](docs/q4_procedure.md). It is separate from the official adaptation and test code. Report source files and the Q4 figure are collected under `runs/report_materials/`; the PDF is left for the user to compose.
The same local folder also contains the original COME and GOLD paper PDFs for offline Q1 reading; their links and checksums are in its `README.md`.

```bash
../.venv/bin/python scripts/collect_report_materials.py --allow-val-images-for-q4 --q4-figure runs/q4_lame_view6/q4_examples.png --q4-metadata runs/q4_lame_view6/q4_examples.json --val-images ../data/val/images --classes starter/classes.txt --out runs/report_materials
```

## Rules

Only the supplied checkpoint may be used. Adaptation may only use the current and previously served unlabeled batches. Validation labels are reserved for official evaluation and method selection; Q4 report preparation follows the narrow post-evaluation procedure above. Never modify the harness or view test labels.
