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

Pending.
