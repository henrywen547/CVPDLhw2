# Round 8: robust BatchNorm statistics under correlated batches

Selected method: `lame_view8_k20`, validation 1987/2112, user-reported public **0.87906** (leaderboard top 0.9567, rank 63/77; grading mean μ = 0.890748). Round 7's neighbor-count gain barely transferred to public (+86 validation images → +0.00423), so this round tries a different component.

Every view currently normalizes with its own current-batch BN statistics. On class-correlated, non-i.i.d. batches of 64, those statistics carry class bias. Two established remedies use only the supplied checkpoint and already-served data:

- **Source mix** (`mix`): `α · batch + (1 − α) · source running statistics`, the prior used by Schneider et al. (NeurIPS 2020).
- **Per-view EMA** (`ema`): one exponential moving average of batch statistics per view index. It is initialized from that view's first served batch and updated with momentum α before normalizing, in the spirit of RoTTA's robust BN (Yuan et al., CVPR 2023, cited on the HW2 slides). This uses current and previously served batches only; no batch-index schedule.

Affine parameters are never updated. Views, graph (k = 20), and output updates are unchanged.

## Predeclared runs

Official validation harness, supplied checkpoint, seed 0, diagnostics off, one commit:

| Run | BN statistics | Role |
| --- | --- | --- |
| `lame_view8_k20` | current batch (native BN) | control, expect 1987 |
| `lame8k20_mix10` | custom module, α = 1 (current batch) | equivalence and timing check only, not a candidate |
| `lame8k20_mix09` | 0.9 batch + 0.1 source | candidate |
| `lame8k20_mix07` | 0.7 batch + 0.3 source | candidate |
| `lame8k20_ema03` | per-view EMA, momentum 0.3 | candidate |
| `lame8k20_ema01` | per-view EMA, momentum 0.1 | candidate |

Synthetic checks on a random ResNet-18 confirmed: α = 1 mix matches native current-batch BN (max |Δ| 4e-6), α = 0 mix matches source BN exactly, and EMA equals current-batch BN on each view's first batch.

## Stopping rule

- A candidate needs **≥ 10 more correct images** than the control.
- The local full test is only 2.5 s under 90 s, so a candidate whose validation harness time exceeds **15.6 s** (≈ 89 s test, using the observed 5.7× ratio) gets no test pass this round; its speed would be addressed separately.
- At most **one** full test pass, for the best qualifying candidate. Any new CSV stays separate until the user reports a public score above 0.87906.

## Results

Pending.
