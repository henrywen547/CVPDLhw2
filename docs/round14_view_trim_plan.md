# Round 14: per-image view-variation trimming (TACT-inspired)

Selected method: `lame8k20_ema03_src02`, validation 2018/2112, user-reported public **0.89257**, local full test 92.3 s. Candidate screening is in `docs/next_method_review_20261114.md`. This is a single bounded round. Nearby k, EMA, and source-share values are not searched again.

[TACT](https://papers.neurips.cc/paper_files/paper/2025/hash/09d6e2295eb50b4cdcb45d75220a2598-Abstract-Conference.html) (Liu et al., *Test-Time Adaptation by Causal Trimming*, NeurIPS 2025; [author code, MIT](https://github.com/NancyQuris/TACT)) works per sample:

1. It centers the features of several semantics-preserving augmentations of the same image.
2. It removes the leading principal directions, `z − Σ(z·u)u`, from the test feature and from the classifier rows.
3. Its full method also accumulates the trimmed prototypes.

Its trimming-only base class `CT` classifies the trimmed feature with the original classifier and keeps the bias. No author code was copied.

## Implementation (`LameViewTrim`, independent and simplified)

- **Features.** It reuses the eight existing view features per image (original and flip at 100/90/80/70% crops), so there is no extra backbone pass. BN, k = 20, EMA 0.3, source share 0.2, and ten graph updates are fixed.
- **Direction.** For each image, the view features are centered. The leading eigenvector of the 8×8 Gram matrix, mapped back through the centered matrix, gives the unit direction `u` of the within-image variation. Only **one** direction is removed.
- **Degenerate ranks.** An image is not trimmed (`u = 0`) when the top eigenvalue is at most 1e-6 × the mean squared view-feature norm.
- **`trim_prior`.** Each view feature is trimmed as `f − (f·u)u` and passed through the supplied classifier, bias kept. The view softmaxes are averaged in the baseline's order. The prior therefore keeps the baseline's mean-of-softmax structure.
- **`trim_graph`.** The graph uses the mean trimmed feature `f̄ − (f̄·u)u`.
- **Not used.** TACT's photometric augmentations and accumulated trimmed prototypes. For a single shared projection, trimming both the feature and the classifier rows equals trimming the feature alone, so that is not a separate arm.
- **Caveat.** Crops can cut away part of the object, and each view has its own EMA BN state. The removed direction is therefore view variation, which does not cleanly separate causal from non-causal factors. A negative result rejects only this low-cost integration, not TACT.

## Pre-checks (random inputs, supplied checkpoint, no dataset access)

- With `u` forced to zero, both the predictions and the graph inputs are **bitwise identical** to `lame8k20_ema03_src02` over three 64-image batches.
- `u` matches the top right singular vector of the centered view matrix (|cos| ≥ 0.99999993). Removing it takes out 38.7% of the within-image variance on random inputs, equal to that vector's share of the squared singular values.
- Cost per batch: 460.8 ms (control), 462.7 ms (`lame_trim_prior`), 463.1 ms (`lame_trim_both`). That is about +0.5%.

## Predeclared runs (official validation only)

| Run | Prior | Graph | Role |
| --- | --- | --- | --- |
| `lame8k20_ema03_src02` | baseline | baseline | same-commit control, expect 2018 |
| `lame_trim_prior` | trimmed | unchanged | changes only the classifier evidence |
| `lame_trim_both` | trimmed | trimmed | also changes the neighborhood geometry |

## Stopping rule

- A candidate needs **≥ 10 more correct images** than the control, and validation harness time **≤ 1.05 ×** the control's.
- No rank, tolerance, crop, or trimming-weight sweep after seeing the outcome. If neither candidate qualifies, the branch stops.
- At most one qualifying candidate (the larger gain; `lame_trim_prior` on a tie) gets one test pass and one public check. The selected artifact stays until a public score above 0.89257.

## Results

All runs used code commit `c8bd656`, seed 0, and diagnostics off, on the local GTX 1080 Ti:

| Run | Correct / 2112 | Top-1 | Δ vs. control | Predictions changed | Harness s | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lame8k20_ema03_src02` (control) | 2018 | .955492 | — | — | 16.2 | 848 |
| `lame_trim_prior` | 1981 | .937973 | −37 | 78 | 16.4 | 890 |
| `lame_trim_both` | 1886 | .892992 | −132 | 200 | 16.3 | 890 |

The control CSV is byte-identical to round 10's `lame8k20_ema03_src02`. Neither candidate qualifies, so **no test pass** was run and the branch stops here as predeclared. The selected method stays `lame8k20_ema03_src02`.

Both arms lose accuracy, and trimming the graph features hurts most. A plausible reading: on this stream, the main within-image variation across flips and 70–100% crops carries class evidence, such as object scale and extent, rather than nuisance. Every view also shares the same corruption, so this direction cannot isolate the corruption factor that TACT's photometric augmentations target. Each image also gets its own direction, which distorts the cosine neighborhoods the graph relies on. This rejects only this low-cost integration, not TACT itself.
