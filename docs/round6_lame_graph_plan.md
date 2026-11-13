# Round 6: LAME graph variants at fixed eight-view cost

Selected method: `lame_view8`, validation 1827/2112 (.865057), user-reported public **0.81835**, full local GTX 1080 Ti test 87.5 s / 848 MiB. That is 0.02165 below the 0.84 strong baseline. Adding views is not viable because the local test is only 2.5 s under the limit. This round changes only the output-correction graph. The eight views and forward passes stay fixed, so the added cost is a few 64×64 or 64×128 matrix operations per batch.

## Predeclared runs

All runs use the official validation harness, the supplied checkpoint, seed 0, diagnostics off, and one code commit:

| Run | Change from `lame_view8` | Motivation |
| --- | --- | --- |
| `lame_view8` (control) | none | Same-commit rerun, which rules out nondeterminism (expect 1827) |
| `lame_view8_k3` | 3 nearest neighbors instead of 5 | Sparser graph; LAME's affinity is its only hyperparameter |
| `lame_view8_k10` | 10 nearest neighbors | Denser graph |
| `lame_view8_mem` | previous served batch's averaged features and final assignments added as **fixed** graph nodes | The validation permutation test indicated temporally correlated predictions, so same-class neighbors may sit across the batch boundary |

`lame_view8_mem` uses only the immediately preceding batch, which was already served and answered. Memory nodes are candidate neighbors for the current samples. They contribute `A_cm · z_mem` to each update but are never updated, so earlier predictions cannot change. The method uses no batch index, image ID, label, or future batch.

## Stopping rule

- A variant needs **≥ 10 more correct images** than the same-commit control to count as a gain.
- If both the best `k` variant and `mem` gain, one combined variant (best `k` + memory) may be added for a single validation run.
- At most **one** full test pass this round, for the best variant that gains, and only if the local test stays under 90 s.
- Because validation gains have transferred only partially to public (`lame_view6` → `lame_view8`: +55 validation images, +0.00569 public), any new test CSV stays a separate candidate until the user reports its public score. The root `submission.csv` is replaced only if that score exceeds 0.81835.

## Results

All runs used code commit `e4a3478`, seed 0, diagnostics off, on the local GTX 1080 Ti:

| Run | Correct / 2112 | Top-1 | Δ vs. control | Harness s | Peak VRAM MiB | Val predictions differing from control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lame_view8` (control) | 1827 | .865057 | — | 15.3 | 848 | 0 (identical to the round-5 run) |
| `lame_view8_k3` | 1779 | .842330 | −48 | 15.3 | 848 | — |
| `lame_view8_k10` | **1901** | **.900095** | **+74** | 15.3 | 848 | 98 |
| `lame_view8_mem` | 1769 | .837595 | −58 | 15.4 | 848 | — |

- The control reproduced the round-5 predictions exactly, so the comparison is not affected by run-to-run noise.
- Accuracy rises with neighbor count (3 → 5 → 10). Following the predeclared rule, no further `k` values were tried this round.
- The one-batch fixed memory hurt by 58 images; the branch is stopped and no combined variant was run. A plausible reason is that the previous batch's refined labels add confirmation bias, including across domain changes, but this was not diagnosed.

`lame_view8_k10` received the round's single full test pass at the same commit:

```bash
../../.venv/bin/python run.py --method lame_view8_k10 --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_lame_view8_k10 --seed 0
```

It took **87.5 s / 848 MiB** locally, the same as `lame_view8`, since graph density costs nothing measurable. The verified upload copy is `runs/final_lame_view8_k10/submission.csv` (SHA-256 `98c48cb5b4001f0f810d04ae13f4402d6b68c36265a688cba21516667bf8df15`, matching its log); the reproduction snapshot is in `runs/lame_view8_k10_candidate/`. Against `lame_view8`, **1068/12,288 (8.7%)** test predictions differ, versus 98/2112 (4.6%) on validation, so the denser graph changes the test stream more. No test labels were read. Public score pending; the root `submission.csv` stays `lame_view8` until the user's feedback.

The user subsequently reported **public 0.87483** for this file, versus 0.81835 for `lame_view8`. The gain is **+0.05648**, and the score now exceeds the 0.84 strong baseline. Unlike the view-count change in round 5, most of the validation gain transferred to public. `lame_view8_k10` is now selected for the root `submission.csv`, `runs/submission_staging/`, and the Q4 figure (`runs/q4_lame_view8_k10/`). Private accuracy and RTX 3090 runtime remain unmeasured.
