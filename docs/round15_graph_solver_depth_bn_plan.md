# Round 15: LAME solver length, matched RBF graph, and layer4 source BN

**Selected method.** `lame8k20_ema03_src02`: validation 2018/2112, user-reported public **0.89257**, local full test 92.3 s.

**Origin.** Codex proposed these candidates in `docs/next_directions_20261116.md`. Each candidate changes exactly one component of the selected method. No k, EMA, or source-share value is searched. The multi-view affinity consensus idea stays in reserve.

**Author implementation.** The [LAME author code](https://raw.githubusercontent.com/fiveai/LAME/master/src/adaptation/lame.py) was checked on 2026-11-16 and works as follows:

- Features are L2-normalized.
- Affinities are binary kNN, RBF, or linear.
- RBF is dense, uses Euclidean distance, and sets σ to the batch mean of the k-th neighbor distance.
- The solver runs at most 100 updates of `z ∝ q·exp(Wz)`, with a relative energy-change stop of 1e-8.

No author code was copied.

## Candidates (independently written)

| Run | Change | Question |
| --- | --- | --- |
| `lame8k20_ema03_src02` | none (same-commit control, expect 2018) | — |
| `lame_src02_steps100` | 100 fixed graph updates instead of 10; same graph and probabilities | Is ten steps stopping early? More steps may also hurt. |
| `lame_src02_rbf` | Graph is a dense RBF graph instead of top-20 cosine. RBF uses Euclidean distance on L2-normalized mean features, σ = batch mean of each sample's distance to its 20th nearest other sample, and a zero diagonal. One global scale per batch sets its mean row sum equal to the top-20 cosine graph's. | With average correction strength matched, does a smoother edge geometry help? |
| `lame_src02_l4src` | All ten `layer4` BNs, including the downsample BN, use checkpoint running statistics. Stem and layer1–3 are unchanged (EMA 0.3 + 20% source). | Does keeping late semantic statistics fixed help, following the label-shift sensitivity of deep layers reported for [Channel-Selective Normalization](https://arxiv.org/abs/2402.04958) (CoLLAs 2024)? |

Implementation notes:

- **`steps100`.** All 100 updates run, with no energy check, to avoid per-step GPU syncs. For diagnostics only, when `LAME_SOLVER_DIAG` is set, each batch records three label-free values: max |z₁₀ − z₁₀₀|, the number of argmax changes after step ten, and the last-step residual. Predictions are unaffected.
- **`rbf`.** It falls back to the top-20 cosine graph only if σ is non-finite or ≤ 1e-12, or if the RBF mass is ≤ 1e-12.
- **`l4src`.** The paper's source-data channel scores are not used, so this is a depth-only ablation, not Hybrid-TTN. The cut point is fixed at layer4 and is not searched afterwards.

## Pre-checks (random inputs, supplied checkpoint, no dataset access)

- **Refactor.** The refactored `adjust_outputs` and the selected method are bitwise identical to `fb2b96b` over four 64-image batches.
- **RBF graph.** It is symmetric, its diagonal is 0, its entries are non-negative, and its mean row sum equals the cosine graph's (0.49933 both).
- **layer4 BN.** Exactly 10 of 53 BNs are in source mode, all of them in layer4.
- **Steps on random inputs.** max |z₁₀ − z₁₀₀| was 0.85–0.99 per batch, with 38, 8, 0, and 0 argmax changes. The last-step residual was 0 in every batch. Random inputs are not a proxy for accuracy.
- **Time per batch.** Control 461.4 ms, `steps100` 461.9 ms, `rbf` 462.6 ms, `l4src` 457.3 ms.

## Stopping rule

- A candidate needs **≥ 10 more correct images** than the control, and validation harness time **≤ 1.05 ×** the control's.
- The largest qualifying gain gets one test pass and one public check. On a tie, the lower-cost candidate wins. Nothing else is tested.
- No bandwidth, step count, or BN cut point is chosen after seeing the results. If no candidate qualifies, this round ends, and runtime verification and the report come next.
- The selected artifact stays until a public score above 0.89257.
