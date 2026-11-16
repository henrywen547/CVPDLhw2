# Round 16: public screening of validation-tied candidates

On 2026-11-16 the user chose to spend unused daily Kaggle uploads on existing candidates. Validation is saturated: rounds 9–15 left the top candidates within a few images of each other, so validation cannot separate them. Public feedback selects among already-validated variants. It is not used to choose new hyperparameter values.

## Existing test CSVs not yet uploaded

Only `final_bn_view10` and `final_tent_001` exist without a public score. Both are early, weak methods; their siblings scored 0.76627 and 0.72005. They are not worth upload slots.

## Candidates

| Run | Validation | Mechanism versus the selected `src02` |
| --- | ---: | --- |
| `lame8k20_ema03_src03` | 2017 | 30% instead of 20% source statistics |
| `lame8k20_ema02` | 2019 | slower EMA (0.2), no source statistics |
| `lame_bbn05` | 2019 | EMA 0.3 + class-balanced batch statistics (γ = 0.5), no source statistics |
| `lame8k20_ema02_src02` | **1997** | EMA 0.2 + 20% source; −21 versus the control, so **excluded** (no test pass) |

`lame8k20_ema02_src02` was added at `ce0e0b8` and run once on validation (16.2 s).

## Test passes (local GTX 1080 Ti, code `ce0e0b8`)

| Run | Harness s | Peak VRAM MiB | Predictions changed vs. `src02` | Changed vs. `ema03` | submission.csv SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `lame8k20_ema03_src03` | 92.3 | 848 | 283 | 806 | `867ad8d7…a95ea0` |
| `lame8k20_ema02` | 91.9 | 848 | 580 | 605 | `9dfd8769…80cc4e` |
| `lame_bbn05` | **95.6** | 850 | 621 | 295 | `693ebf8f…2923c` |

`lame_bbn05` overruns 90 s by 5.6 s locally, more than the other candidates. It should be selected only if its public gain is clear, and only with the RTX 3090 risk in mind.

## Selection rule

The selected artifact switches only to a candidate whose user-reported public score exceeds 0.89257. Differences of one or two public images (about 0.0002–0.0003) are within noise. Picking the best of several near-ties on public inflates the public score without improving the expected private score.
