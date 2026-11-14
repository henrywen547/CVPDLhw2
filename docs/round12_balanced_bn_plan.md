# Round 12: class-balanced batch statistics (simplified TRIBE balanced BN)

Selected method: `lame8k20_ema03`, validation 2019/2112, user-reported public **0.89143**, local full test 91.6 s. Round 11's T3A collapse showed that class-correlated batches are a central difficulty of this stream. Per-view EMA BN was the most recent component to transfer to public. This round therefore targets the class bias **inside** each batch's BN statistics.

[TRIBE](https://arxiv.org/html/2309.14949v2) (Su, Xu, Jia, AAAI 2024) argues that adapting normalization to class-imbalanced test batches is a main failure cause. It proposes balanced BN that models per-class statistics and derives the global statistics from them. [NOTE](https://arxiv.org/abs/2208.05117) (NeurIPS 2022) makes a related argument for temporally correlated streams.

## Implementation (`LameBalancedBN`, independent and simplified)

- View 0 (original image) uses the plain per-view EMA BN. Its argmax provides pseudo-labels for the current batch.
- Views 1–7 compute batch statistics with per-sample weights `w_i ∝ count(pred_i)^(−γ)`, normalized to sum to 1. The weighted mean and variance come from per-sample means and variances via the law of total variance.
- The weighted statistics feed the same per-view EMA (momentum 0.3). Weights are cleared after each batch.
- TRIBE's per-class running statistics and tri-net self-training are **not** used.
- Checks: uniform weights reproduce plain batch statistics (max |Δ| 4e-6 on a random ResNet-18). Measured cost is +4.3% per batch versus the control (459 → 479 ms on one 64-image batch).

## Predeclared runs

| Run | γ | Role |
| --- | ---: | --- |
| `lame8k20_ema03` | — | same-commit control, expect 2019 |
| `lame_bbn05` | 0.5 | partial balancing |
| `lame_bbn10` | 1.0 | each predicted class contributes equally |

## Stopping rule

- A candidate needs **≥ 10 more correct images** than the control.
- A test pass needs validation time **≤ 16.5 s** (≈ 94 s locally), so that the overrun stays slight per the user's policy. A qualifying but slower candidate is reported to the user for a time/accuracy decision.
- At most one test pass. A new CSV stays separate until its public score exceeds 0.89143.

## Results

All runs used code commit `7493429`, seed 0, diagnostics off, on the local GTX 1080 Ti:

| Run | Correct / 2112 | Top-1 | Δ vs. control | Harness s | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lame8k20_ema03` (control) | 2019 | .955966 | — | 16.1 | 848 |
| `lame_bbn05` | 2019 | .955966 | 0 | 16.8 | 850 |
| `lame_bbn10` | 2015 | .954072 | −4 | 16.7 | 850 |

No candidate qualifies, and both exceed the 16.5 s time guard, so **no test pass** was run. Per-view EMA already averages statistics over several correlated batches, which may leave little within-batch class bias for re-weighting to remove. Pseudo-labels from a single view are also noisier than the final LAME labels. The selected method stays `lame8k20_ema03`.
