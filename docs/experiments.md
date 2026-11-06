# Official validation experiments

All rows use the official `starter/run.py --split val`, the unchanged official `harness.py`, the same supplied checkpoint, seed 0, original stream order, batch size 64, and GTX 1080 Ti (11 GiB). Each command loads the source checkpoint afresh. The harness time begins after model construction and data loader setup; it is not the full process startup time. The grading RTX 3090 and full 12,288-image test resource limits remain unverified.

| Method | Git commit used | Validation Top-1 | Correct / 2112 | Harness seconds | Peak VRAM MiB | Setting |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| source | `0b26fb09acd2749098eb20222d0cc947f9e1034b` | .545455 | 1152 | 2.7 | 706 | eval, no update |
| tent | `b6f2a6d886259421a1a53089dbfcc838f35f0083` | .723485 | 1528 | 5.2 | 3470 | BN affine, 1 SGD step/batch, lr .001, momentum .9 |
| tent_come | `b6f2a6d886259421a1a53089dbfcc838f35f0083` | .722538 | 1526 | 5.3 | 3470 | Same as Tent; opinion entropy, K=1000 |
| tent_gate, threshold .4 | `f3edf49af806057b8367bd0a5760b92022961cd8` | .723485 | 1528 | 5.2 | 3470 | Custom P2 rule; 33/33 batches updated, so gate was inactive |
| tent_gate, threshold .25 | `0197bb0567388a83b2faffb4a57c032fb175fb2e` | .723485 | 1528 | 5.0 | 3370 | Custom P2 rule; 31/33 batches updated, 2 predictions changed, no accuracy gain |
| bn_batch | `b013fe81faa444b9065936e224ae89a29a175828` | .722538 | 1526 | 2.6 | 706 | Current-batch BN statistics only, no parameter updates |
| gold_sf | `b013fe81faa444b9065936e224ae89a29a175828` | .722538 | 1526 | 2.7 | 754 | Source-free GOLD-inspired subspace ablation, rank 8, alpha .1, confidence threshold .5, scale SGD lr .01 |

The COME loss replacement scored 2 fewer correct images than Tent at this initial setting. This is a retained negative result for P1; it does not rule out other settings or paper settings. Returned predictions come from the current batch before its optimizer update; later batches use the updated state. No method sees validation labels.

P2 prediction diagnostics, verified against formal-run CSVs, show source / Tent / Tent+COME / gate-.4 adjacent predicted-class rates of `.203 / .134 / .133 / .134`. Their average batch maximum predicted-class shares are `.296 / .185 / .185 / .185`; average model confidences are `.714 / .459 / .454 / .459`. The lower Tent confidence accompanies higher **overall** accuracy in this validation run; no per-window accuracy claim follows from it. A fixed .4 gate was too high for the Tent prediction distribution (maximum batch share .28125), so a single active threshold comparison used .25. This threshold was chosen from already-seen validation predictions and is not a batch-index rule.

The .25 gate skipped 2/33 updates and changed two predicted labels compared with Tent, but did not change official Top-1. Its prediction diagnostics and timing run produced identical CSVs. The data-driven skip did not improve this validation result, so P3 comparison continues to use plain Tent as the primary adapted baseline.

The P3 `gold_sf` output differs from `bn_batch` on 22 images but has exactly the same overall accuracy. Its inference and memory costs are close to BN-only because the backbone is frozen and only an eight-value scale is trained. This is a **modified source-free component**, not the author's full GOLD result. It offers no validation gain over BN-only or Tent at the initial setting. The full author's implementation is excluded because it requires source prototypes and source warm-up forbidden by HW2.

The diagnostic rerun confirmed `gold_sf` updated the subspace on **33/33 batches**, with scale-vector norm increasing from .00251 after batch 1 to .62142 after batch 33. Its official prediction CSV exactly matched the timed run. This supports treating the unchanged accuracy as an observed negative result, not an inactive adapter. No per-window true accuracy is available.

## Selected validation version

The current best official validation Top-1 is plain `tent`, 1528/2112. `tent_gate` at .25 ties it with extra logic, so plain Tent is selected for one official test pass. The 2-image lead over BN-only is small; the choice is based on the recorded validation protocol, not on unseen test labels. The test pass will start afresh from the official source checkpoint. We will not use `source_diag`, validation labels, or any saved validation state in test adaptation.

## Commands and outputs

Run from `CVPDLhw2/starter`:

```bash
../../.venv/bin/python run.py --method source --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/source_val --seed 0
../../.venv/bin/python run.py --method tent --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/tent_lr001 --seed 0 --tta-lr 0.001 --tta-momentum 0.9
../../.venv/bin/python run.py --method tent_come --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/come_lr001 --seed 0 --tta-lr 0.001 --tta-momentum 0.9 --come-k 1000
../../.venv/bin/python run.py --method tent_gate --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/tent_gate_040 --seed 0 --tta-lr 0.001 --tta-momentum 0.9 --max-class-share 0.4
../../.venv/bin/python run.py --method tent_gate --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/tent_gate_025_timed --seed 0 --tta-lr 0.001 --tta-momentum 0.9 --max-class-share 0.25
../../.venv/bin/python run.py --method bn_batch --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/bn_batch --seed 0
../../.venv/bin/python run.py --method gold_sf --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/gold_sf_r8 --seed 0 --tta-lr 0.01 --tta-momentum 0.9 --gold-rank 8 --gold-alpha 0.1 --gold-confidence-threshold 0.5
```

The ignored `runs/<run-name>/val_run_log.json` and `val_pred.csv` files contain the official metrics, exact command arguments, and predictions. `docs/data_exploration.md` contains the source model EDA. The incoming paper plan is at `../CVPDL_2026_HW2_Paper_Implementation_Plan.md`.

## Single official test pass

With selected Git commit `1e4ee135fef0b29d870f8f608d82be82ad7c437c`, from `CVPDLhw2/starter`:

```bash
../../.venv/bin/python run.py --method tent --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_tent_001 --seed 0 --tta-lr 0.001 --tta-momentum 0.9
```

The official harness completed 12,288 images in **28.2 seconds**, with **3470 MiB** peak sampled VRAM and **3294 MiB** maximum PyTorch reserved memory, on the local GTX 1080 Ti. The official output files are `runs/final_tent_001/test_pred.csv` and `runs/final_tent_001/test_run_log.json`. CSV validation found header `id,label`, exactly 12,288 unique IDs, integer labels in 0–19, and SHA-256 `d6ee738231872c47d1c9c744110dd12700a2797cee520569ffe51144d22c06e2`, equal to the log's `pred_sha256`. The test log contains no accuracy. This was one online test pass; no test labels were sought or used. Timing and VRAM still need confirmation on the grading RTX 3090.

The Kaggle and COOL uploads are external submission steps. The original Q4 slide wording provided by the user requires eight validation images with predicted and true labels. The report-only Q4 selection and its narrow validation-data access are documented in `docs/q4_procedure.md`; no validation truth entered adaptation. A local submission staging directory contains the official test CSV/log, code, and reproduction Readme; the report PDF is generated separately and should be checked before packaging.
