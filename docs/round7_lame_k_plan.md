# Round 7: neighbor count beyond ten

Round 6 found validation rising with graph density at fixed eight-view cost: k = 3, 5, 10 scored 1779, 1827, 1901/2112. The selected `lame_view8_k10` reached user-reported **public 0.87483** (+0.05648 over k = 5), so this validation gain transferred to public, unlike the round-5 view-count gain. The next grading threshold the user named is μ = 0.890748 (μ + σ = 0.986811).

## Predeclared runs

Official validation harness, supplied checkpoint, seed 0, diagnostics off, one code commit:

| Run | Neighbors |
| --- | ---: |
| `lame_view8_k10` (same-commit control, expect 1901) | 10 |
| `lame_view8_k15` | 15 |
| `lame_view8_k20` | 20 |
| `lame_view8_k30` | 30 (about half of a 64-image batch) |

Everything else is unchanged: eight views, cosine similarity with negative values clamped to zero, symmetrization, and ten updates.

## Stopping rule

- A variant needs **≥ 10 more correct images** than the same-commit k = 10 control.
- At most **one** full test pass, for the best qualifying variant.
- No larger k is tried this round, even if k = 30 is best; that trend would be recorded for a later predeclared round.
- A new test CSV stays a separate candidate until the user reports a public score above 0.87483.

## Results

All runs used code commit `7010ae4`, seed 0, diagnostics off, on the local GTX 1080 Ti:

| Run | Correct / 2112 | Top-1 | Δ vs. k = 10 control | Harness s | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lame_view8_k10` (control) | 1901 | .900095 | — (identical predictions to round 6) | 15.3 | 848 |
| `lame_view8_k15` | 1967 | .931345 | +66 | 15.4 | 848 |
| `lame_view8_k20` | **1987** | **.940814** | **+86** | 15.4 | 848 |
| `lame_view8_k30` | 1974 | .934659 | +73 | 15.4 | 848 |

Validation peaks near k = 20 and declines at k = 30. All three variants qualify; following the rule, only the best (`k20`) received the round's single full test pass:

```bash
../../.venv/bin/python run.py --method lame_view8_k20 --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_lame_view8_k20 --seed 0
```

It took **87.5 s / 848 MiB** locally. The verified upload copy is `runs/final_lame_view8_k20/submission.csv` (SHA-256 `062625d79f05c638bc92bda13956e10af2ea5dd22307eb5a0358dd37bc24697f`, matching its log); the reproduction snapshot is in `runs/lame_view8_k20_candidate/`. Against k = 10, **1468/12,288 (11.9%)** test predictions differ, versus 141/2112 (6.7%) on validation. No test labels were read. Public score pending; the root `submission.csv` stays `lame_view8_k10`.

The user subsequently reported **public 0.87906** for this file, versus 0.87483 for k = 10. The gain is only **+0.00423**, against +86 validation images (+4.1 points). Most of the validation gain did not transfer, so neighbor count is saturating on the test stream. `lame_view8_k20` is now selected for the root `submission.csv`, staged COOL files, and Q4 figure (`runs/q4_lame_view8_k20/`). For context, the user reported the public leaderboard's top score as 0.9567, with this submission ranked 63/77.
