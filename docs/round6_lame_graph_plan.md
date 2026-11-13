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

Pending.
