# New Tent diagnosis and bounded learning-rate round

The user-supplied `CVPDL_2026_HW2_Paper_Implementation_Plan (1).md` asks to diagnose the existing Tent before adding methods. Its statement that run logs and prediction CSVs are missing is stale for this workspace: `docs/experiments.md` records each completed official validation run, and ignored `runs/` contains their official CSV/JSON files. We will still repeat the P0 comparison under one committed code version to eliminate version drift, then test only the predeclared learning-rate grid below. No official test stream is rerun in this round, and `submission.csv` remains the earlier Tent test output until a later validation decision.

## P0: same-code comparison

From `CVPDLhw2/starter`, with the official source checkpoint, seed 0, batch size 64, SGD momentum 0.9, and one update per served batch:

| Method | Command suffix after `run.py` | Diagnostic purpose |
| --- | --- | --- |
| source | `--method source --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/round2_source --seed 0` | Fixed source baseline |
| bn_batch | `--method bn_batch --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/round2_bn_batch --seed 0` | Current-batch BN effect without parameter updates |
| tent | `--method tent --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/round2_tent_001 --seed 0 --tta-lr 0.001 --tta-momentum 0.9` | Reproduce 1528/2112 |
| tent_gate | `--method tent_gate --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/round2_gate_025 --seed 0 --tta-lr 0.001 --tta-momentum 0.9 --max-class-share 0.25` | Count skips and compare all 2112 predictions |

Formal timing runs leave `CVPDL_DIAG_JSONL` unset. Separate diagnostic reruns set that variable and write to their own run directories; compare their official CSV bytes to the timing runs. The TTA diagnostic JSONL records per-image prediction/confidence/entropy, per-batch update decision and BN-affine update L2 norm. It does not expose labels to the method. Source diagnostics already record input and prediction summaries, and will be rerun under the same commit.

P0 will report official Top-1, harness seconds, peak VRAM, 2112-prediction difference count, number of optimizer updates, mean confidence/entropy, and mean update norm. The current code returns the current batch's prediction before its optimizer update. `source` uses frozen running BN statistics; `bn_batch` and Tent use current-batch statistics. No method is reset by batch index or image ID.

## P1: fixed small grid

Change only Tent's SGD learning rate to **0.0003** and **0.003**, retaining 0.001 as the P0 control. Momentum remains 0.9; BN and prediction timing remain identical. Each candidate starts from the official source checkpoint in a fresh process. This is a three-point validation grid, not a search over the official test set. Record both positive and negative results with the exact commit and commands. Select by official overall validation Top-1; proxy confidence and entropy only explain behavior. If the best score ties, prefer the less complex existing 0.001 setting unless resource evidence favors another.

COME and the source-free GOLD-inspired ablation were already tested at the prior setting, scoring 1526/2112 each. They remain comparison context, not this round's primary changes. The original author's full GOLD classification implementation uses source prototypes and warm-up and is not eligible as-is. Public leaderboard 0.84 and an informal reported 0.91 refer to different test subsets, so they are not comparable to the 0.72348 validation score.
