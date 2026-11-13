# Round 10: hybrid BN statistics (per-view EMA + source prior)

Selected method: `lame8k20_ema03`, validation 2019/2112, user-reported public **0.89143**, local full test 91.6 s. Round 8 found two separate gains over native current-batch BN: per-view EMA, momentum 0.3 (+32), and a 30% source-statistics prior (+24). Round 9 showed that EMA momentum and k are at a local optimum. This round tests whether the two BN remedies combine.

Each view keeps its EMA of **pure** batch statistics (unchanged), and normalization uses `(1 − s) · EMA + s · source running statistics` for both mean and variance. Synthetic checks on a random ResNet-18: s = 0 skips the new code path; s = 1 equals source BN exactly; the stored EMA is identical with and without blending. Views, k = 20, and output updates are unchanged; the extra cost is two small per-channel operations per BN layer.

## Predeclared runs

Official validation harness, supplied checkpoint, seed 0, diagnostics off, one commit:

| Run | Source share s | Role |
| --- | ---: | --- |
| `lame8k20_ema03` | 0 | same-commit control, expect 2019 |
| `lame8k20_ema03_src01` | 0.1 | candidate |
| `lame8k20_ema03_src02` | 0.2 | candidate |
| `lame8k20_ema03_src03` | 0.3 | candidate |

## Stopping rule

- A candidate needs **≥ 10 more correct images** than the control.
- At most **one** full test pass, for the best qualifying candidate, provided its validation time does not rise noticeably above the control's 16.0–16.1 s (user policy: slight local overrun acceptable, not severe).
- A new CSV stays separate until the user reports a public score above 0.89143.

## Results

Pending.
