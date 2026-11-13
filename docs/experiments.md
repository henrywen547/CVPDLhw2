# Official validation experiments

All rows use the official `starter/run.py --split val`, the unchanged official `harness.py`, the same supplied checkpoint, seed 0, original stream order, batch size 64, and GTX 1080 Ti (11 GiB). Each command loads the source checkpoint afresh. The harness time begins after model construction and data loader setup; it is not the full process startup time. The full 12,288-image test was timed locally, but resource limits on the grading RTX 3090 remain unverified.

| Method | Git commit used | Validation Top-1 | Correct / 2112 | Harness seconds | Peak VRAM MiB | Setting |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| source | `0b26fb09acd2749098eb20222d0cc947f9e1034b` | .545455 | 1152 | 2.7 | 706 | eval, no update |
| tent | `b6f2a6d886259421a1a53089dbfcc838f35f0083` | .723485 | 1528 | 5.2 | 3470 | BN affine, 1 SGD step/batch, lr .001, momentum .9 |
| tent, lr .003 (earlier test choice) | `96d5b578119115f78e9ac2fcbb632d5ad4657fb2` | .724432 | 1530 | 5.1 | 3470 | Same one-step Tent; lr .003, momentum .9; used for earlier public 0.72005 CSV |
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

## Historical selection before the LR round

Before the bounded LR round, the best official validation Top-1 was plain `tent` at lr .001, 1528/2112. `tent_gate` at .25 tied it with extra logic, so plain Tent was selected for the earlier official test pass. The 2-image lead over BN-only was small. That pass started afresh from the official source checkpoint and used no validation labels or saved validation state in test adaptation. The later lr .003 result and current test artifact are recorded below.

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

The ignored `runs/<run-name>/val_run_log.json` and `val_pred.csv` files contain the official metrics, exact command arguments, and predictions. `docs/data_exploration.md` contains the source model EDA. The new adjustment plan is at `../CVPDL_2026_HW2_Paper_Implementation_Plan (1).md`.

## Earlier official test pass

With selected Git commit `1e4ee135fef0b29d870f8f608d82be82ad7c437c`, from `CVPDLhw2/starter`:

```bash
../../.venv/bin/python run.py --method tent --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_tent_001 --seed 0 --tta-lr 0.001 --tta-momentum 0.9
```

The official harness completed 12,288 images in **28.2 seconds**, with **3470 MiB** peak sampled VRAM and **3294 MiB** maximum PyTorch reserved memory, on the local GTX 1080 Ti. The official output files are `runs/final_tent_001/test_pred.csv` and `runs/final_tent_001/test_run_log.json`. CSV validation found header `id,label`, exactly 12,288 unique IDs, integer labels in 0–19, and SHA-256 `d6ee738231872c47d1c9c744110dd12700a2797cee520569ffe51144d22c06e2`, equal to the log's `pred_sha256`. The test log contains no accuracy. This was one online test pass; no test labels were sought or used. Timing and VRAM still need confirmation on the grading RTX 3090.

The Kaggle and COOL uploads are external submission steps. The original Q4 slide wording provided by the user requires eight validation images with predicted and true labels. The report-only Q4 selection and its narrow validation-data access are documented in `docs/q4_procedure.md`; no validation truth entered adaptation. A local submission staging directory contains the official test CSV/log, code, and reproduction Readme. Report materials are collected separately; the user will compose the PDF.

## Later Tent diagnostic round

The same-code P0/P1 comparison and full commands are in [tent_diagnostic_round.md](tent_diagnostic_round.md). It reproduced 1528/2112 at lr .001 and found 1530/2112 at lr .003. A later separate official test pass selected lr .003, preserving the earlier lr .001 output. The user reported that Tent lr .003 scored **0.72005 public**. That is historical feedback; the current staged test pair, root `submission.csv`, and Q4 metadata are updated in the later multiview round below. No new Kaggle upload was made by the agent.

### Current lr .003 test pass

From `CVPDLhw2/starter`, at Git commit `d244f10ac391523f360c52f4281dd397defceb1d`:

```bash
../../.venv/bin/python run.py --method tent --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_tent_003 --seed 0 --tta-lr 0.003 --tta-momentum 0.9
```

The official test log records 12,288 images, 28.3 seconds, 3470 MiB peak sampled VRAM, and 3294 MiB PyTorch maximum reserved memory on the local GTX 1080 Ti. The CSV has `id,label`, 12,288 unique IDs, labels 0–19, and SHA-256 `85d4d77e4b2876b158e75c844bdc2db857befacedb6f36b3c073cb429bb76775`, matching its paired log. It differs from the preserved lr .001 test CSV on 166 predictions. Test accuracy is unavailable, and the RTX 3090 runtime has not been measured locally.

## Multiview BN round after public feedback

The bounded method and resource results are in [round3_plan.md](round3_plan.md); all official validation rows, including negative results, are in [q3_results.csv](q3_results.csv). Horizontal-flip and center-crop views gave much larger validation gains than immediate Tent updates. The selected `bn_view10_pair` scored **1763/2112 (.8347538)**, compared with Tent lr .003's 1530/2112 (.7244318). It uses only the current unlabeled batch, ten views across five scales, current-batch BN statistics, and no trainable update. The corresponding Q4 figure uses that method's official validation prediction CSV.

At Git commit `a01bef5ec76176f6f9d8cfc5131e3268d87b5496`, from `CVPDLhw2/starter`, the selected full test command was:

```bash
../../.venv/bin/python run.py --method bn_view10_pair --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_bn_view10_pair --seed 0
```

The paired official test outputs are `runs/final_bn_view10_pair/test_pred.csv` and `test_run_log.json`. The full local GTX 1080 Ti stream took **109.3 seconds**, with **1164 MiB** sampled peak VRAM; the CSV SHA-256 is `02baf91683a615f0af84b568a4a6699529dc5f8965e9c998938c53103ddb70f1`, matching its log. The local time is above 90 seconds. The user reported public **0.76627** for this artifact. Whether it passes the grading RTX 3090 limit is not yet measured. The older Tent run and CSV remain intact.

### Lower-cost eight-view fallback

At commit `57c209a056ecbba04d3d0cf000194d8fd196635d`, the same official test stream was run independently with `--method bn_view8 --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_bn_view8 --seed 0`, from `CVPDLhw2/starter`. Its local GTX 1080 Ti test took **87.3 seconds**, **706 MiB** peak VRAM, and produced a paired official CSV/log. The separately generated `runs/final_bn_view8/submission.csv` matches the official CSV (SHA-256 `9460c10feeebae78946484bbea9ad28fc43731aebdbe07059c7bb796930d8330`). Its validation result was **1722/2112 (.8153409)** and the user reported public **0.76822**, slightly higher than ten-view's public 0.76627. This earlier method remains preserved in its own run folder. It is below 90 seconds locally; RTX 3090 performance and private accuracy remain unmeasured.

## Paper-inspired normalization and output correction

The independent normalization-only UnMix-TNS component, described in [round4_unmix_plan.md](round4_unmix_plan.md), scored **1413/2112 (.669034)** and was stopped before test. The independently implemented [LAME-inspired output correction](round4_lame_plan.md) scored **1594/2112 (.754735)** with one current-batch-BN view and **1772/2112 (.839015)** with six views. All are official validation runs from the supplied checkpoint. The six-view graph method was selected for a separate official full test pass.

From `CVPDLhw2/starter`, at Git commit `c6d18e32b6206abb2a021797e64dc013b3579dcc`:

```bash
../../.venv/bin/python run.py --method lame_view6 --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_lame_view6 --seed 0
```

The full GTX 1080 Ti stream took **66.2 seconds** and **808 MiB** sampled peak VRAM. `runs/final_lame_view6/submission.csv` is a verified copy of the official test CSV (12,288 unique IDs, labels 0–19; SHA-256 `b60a57c286eb1eab643b3ac880f6e9a764495aeb5110f7cfb85525273a238398`, matching the paired official log). The user reported **public 0.81266**, an improvement of **0.04444** over the prior eight-view artifact. This is now the root `submission.csv` and staged COOL method. Private accuracy remains unknown.

## Bounded LAME-inspired view-count check

At commit `902672e5812783ea917af4e0a5e3b55d4ac87a68`, the same fixed output graph was evaluated with four and eight views through the official validation harness. Four views scored **1704/2112 (.806818)** in **8.1 seconds / 768 MiB**, below the selected six-view method. Eight views scored **1827/2112 (.865057)** in **15.3 seconds / 848 MiB**, 55 more correct images than six views. The four-view branch was stopped. Details are in [round5_view_sensitivity.md](round5_view_sensitivity.md).

The eight-view method received one official full test pass from `CVPDLhw2/starter`:

```bash
../../.venv/bin/python run.py --method lame_view8 --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_lame_view8 --seed 0
```

The 12,288-image local GTX 1080 Ti pass took **87.5 seconds / 848 MiB**. The paired official CSV/log and verified upload copy `runs/final_lame_view8/submission.csv` have SHA-256 `629858e1bbc0fe859fbac45e1846ed841ee9b451b164ef4309e377902ebc9229`. This was first kept as an independent candidate while the root `submission.csv` remained the user-tested six-view method.

### Selected eight-view artifact

The user reported **public 0.81835** for `runs/final_lame_view8/submission.csv`, versus **0.81266** for six views. The public gain (0.00569) is much smaller than the validation gain (55 images, 2.6 points). The method remains 0.02165 below the 0.84 strong baseline. `lame_view8` is now the selected method:

- `scripts/make_submission.py` defaults to `runs/final_lame_view8/test_pred.csv`. The root `submission.csv` was regenerated and verified against the paired log (SHA-256 `629858e1bbc0fe859fbac45e1846ed841ee9b451b164ef4309e377902ebc9229`).
- `runs/submission_staging/` holds the eight-view `test_pred.csv`, `test_run_log.json`, `Readme.md`, and a `code/` snapshot identical to `starter/`, with the original `harness.py`.
- The Q4 figure was rebuilt from the official `lame_view8` validation predictions (`runs/q4_lame_view8/`).

The six-view and all earlier artifacts remain preserved. Private accuracy and grading RTX 3090 runtime are unmeasured; the local full stream is only 2.5 seconds under 90 seconds on the GTX 1080 Ti.

## Round 6: LAME graph variants at fixed eight-view cost

At commit `e4a3478`, three predeclared graph variants of `lame_view8` were compared with a same-commit control (1827/2112, identical predictions to round 5). `lame_view8_k3` scored 1779, `lame_view8_k10` scored **1901/2112 (.900095)**, and the previous-batch fixed-memory variant `lame_view8_mem` scored 1769. All took 15.3–15.4 s / 848 MiB on validation. Only `k10` met the ≥10-image gain rule. Its one full test pass took **87.5 s / 848 MiB** locally, and its verified upload copy is `runs/final_lame_view8_k10/submission.csv` (SHA-256 `98c48cb5…8df15`). The user reported **public 0.87483** (+0.05648 over `lame_view8`), so `lame_view8_k10` is now the selected method for the root `submission.csv`, staged COOL files, and Q4 figure. Details: [round6_lame_graph_plan.md](round6_lame_graph_plan.md).

## Round 7: neighbor count beyond ten

At commit `7010ae4`, the same-commit k = 10 control reproduced 1901/2112 exactly. k = 15, 20, and 30 scored 1967, **1987 (.940814)**, and 1974, all at 15.4 s / 848 MiB on validation, so validation peaks near k = 20. `lame_view8_k20` received the single full test pass (**87.5 s / 848 MiB** locally). The verified upload copy is `runs/final_lame_view8_k20/submission.csv` (SHA-256 `062625d7…4697f`). The user reported **public 0.87906** (+0.00423 over k = 10), so `lame_view8_k20` is now the selected method. Details: [round7_lame_k_plan.md](round7_lame_k_plan.md).

## Round 8: robust BN statistics

At commit `bf99b69`, k = 20 LAME with custom BN statistics: an α = 1 equivalence check reproduced 1987/2112. Source-mixed α = 0.9 / 0.7 scored 1995 / 2011; per-view EMA momentum 0.3 / 0.1 scored **2019 (.955966)** / 1962. All custom-BN runs took 16.1 s on validation versus 15.3 s, above the predeclared 15.6 s guard. At the user's direction, `lame8k20_ema03` still received one full test pass: **91.6 s / 848 MiB** locally, slightly above 90 s on this GPU. The upload copy is `runs/final_lame8k20_ema03/submission.csv` (SHA-256 `195be44f…e69cf`). The user reported **public 0.89143** (+0.01237 over k = 20, above μ = 0.890748), so `lame8k20_ema03` is now the selected method. Details and speed probes: [round8_robust_bn_plan.md](round8_robust_bn_plan.md).
