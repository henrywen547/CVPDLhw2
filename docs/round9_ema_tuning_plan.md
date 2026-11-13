# Round 9: EMA momentum and neighbor count around the selected method

Selected method: `lame8k20_ema03`, validation 2019/2112, user-reported public **0.89143** (rank 57/79, above the grading mean μ = 0.890748), local full test 91.6 s / 848 MiB. Per-view EMA BN transferred to public (+0.01237), so this round tunes its two settings **at identical cost**: no new views, layers, or kernels.

## Predeclared runs

Official validation harness, supplied checkpoint, seed 0, diagnostics off, one commit:

| Run | k | EMA momentum | Role |
| --- | ---: | ---: | --- |
| `lame8k20_ema03` | 20 | 0.3 | same-commit control, expect 2019 |
| `lame8k20_ema02` | 20 | 0.2 | slower EMA (0.1 already hurt) |
| `lame8k20_ema05` | 20 | 0.5 | faster EMA |
| `lame8k15_ema03` | 15 | 0.3 | k re-check: robust BN changes the features |
| `lame8k30_ema03` | 30 | 0.3 | k re-check |

## Stopping rule

- Many settings have now been chosen on the same 2112-image validation stream, so a candidate needs **≥ 10 more correct images** than the control; smaller differences are treated as noise.
- At most **one** full test pass, for the best qualifying candidate. Local runtime is expected to equal the control's (≈ 91.6 s), which is within the user's "slight overrun is acceptable" policy.
- A new CSV stays separate until the user reports a public score above 0.89143.

## Results

Pending.
