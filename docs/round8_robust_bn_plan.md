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

All runs used code commit `bf99b69`, seed 0, diagnostics off, on the local GTX 1080 Ti:

| Run | Correct / 2112 | Top-1 | Δ vs. control | Harness s | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lame_view8_k20` (control) | 1987 | .940814 | — | 15.3 | 848 |
| `lame8k20_mix10` (equivalence check) | 1987 | .940814 | 0 | 16.1 | 848 |
| `lame8k20_mix09` | 1995 | .944602 | +8 | 16.1 | 848 |
| `lame8k20_mix07` | 2011 | .952178 | +24 | 16.1 | 848 |
| `lame8k20_ema03` | **2019** | **.955966** | **+32** | 16.1 | 848 |
| `lame8k20_ema01` | 1962 | .928977 | −25 | 16.1 | 848 |

- The α = 1 custom module reproduced the control's accuracy, confirming the implementation; its cost is **+0.8 s** on validation (≈ +4.6 s on the full test, estimated ≈ 92 s locally).
- `ema03` and `mix07` qualify on accuracy. `ema01` (slow EMA) hurts, presumably because it lags behind domain changes.
- **No test pass was run**: all custom-BN variants exceed the predeclared 15.6 s validation-time guard.
- Speed probes on one 64-image ResNet-50 forward: native current-batch BN 54.4 ms versus custom EMA 57.4 ms. Fusing the small EMA arithmetic (`torch._foreach_lerp_`) saved only 0.3 ms, and `cudnn.benchmark` saved nothing. The overhead comes from the two-pass `var_mean` + inference-BN kernels versus cuDNN's single training-mode kernel, not from the small ops. Both probes were reverted.

The user chose to accept a modest local overrun to obtain a public score. The user's reasoning: the RTX 3090 is faster than the GTX 1080 Ti, so a slight local excess should be acceptable as long as it is not severe. `lame8k20_ema03` then received one full test pass at the same commit:

```bash
../../.venv/bin/python run.py --method lame8k20_ema03 --split test --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/final_lame8k20_ema03 --seed 0
```

It took **91.6 s** (1.6 s above 90 s on this GPU) and **848 MiB** locally. The verified upload copy is `runs/final_lame8k20_ema03/submission.csv` (SHA-256 `195be44f8e8b27d5247483d0bf001732348d9e47d15c849b44373856477e69cf`, matching its log); the reproduction snapshot is in `runs/lame8k20_ema03_candidate/`. Against k = 20 with native BN, **970/12,288 (7.9%)** test predictions differ, versus 74/2112 (3.5%) on validation. No test labels were read. Public score pending; the root `submission.csv` stays `lame_view8_k20`. RTX 3090 runtime remains unmeasured.

The user subsequently reported **public 0.89143** (rank 57/79) for this file, versus 0.87906 for k = 20 with native BN. The gain is **+0.01237**, and the score now exceeds the grading mean μ = 0.890748. `lame8k20_ema03` is now selected for the root `submission.csv`, staged COOL files, and Q4 figure (`runs/q4_lame8k20_ema03/`). The local test time of 91.6 s on the GTX 1080 Ti is a known risk; the RTX 3090 rerun is expected to be faster but is unmeasured.
