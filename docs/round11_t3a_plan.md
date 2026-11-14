# Round 11: T3A-style prototype prior on top of LAME + EMA BN

Selected method: `lame8k20_ema03`, validation 2019/2112, user-reported public **0.89143**, local full test 91.6 s. Rounds 9–10 found the graph (k) and BN-statistics settings saturated on validation.

## Motivation from the experiment record

- What worked: output-only LAME correction, a denser within-batch graph, and **cross-batch** smoothing of BN statistics (EMA).
- What failed: gradient updates (Tent, COME), and using the previous batch's refined outputs as fixed graph nodes (−58).
- LAME's own paper states that it cannot accumulate knowledge across batches. EMA BN adds cross-batch information at the normalization layer; this round adds it at the **classifier**, cheaply.

[T3A](https://papers.neurips.cc/paper_files/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf) (Iwasawa & Matsuo, NeurIPS 2021) is backpropagation-free. It keeps a per-class support set of test features filtered to the lowest-entropy members and classifies by similarity to class prototypes. Unlike the round-6 memory (one batch, as graph nodes), the support set accumulates entropy-filtered class evidence over the whole stream.

## Implementation (`LameT3A` in `starter/lame_output.py`, written independently)

- The support set starts from the classifier's weight rows; their labels are the classifier's own argmax (exactly 0–19).
- **Our combination:** after each batch's outputs are final, the batch's view-averaged features (under per-view EMA BN) join the support set with **LAME-refined labels and entropies**. The original T3A uses source-classifier argmax and entropy.
- Each class keeps its `M` lowest-entropy members. Prototype logits are `z · normalize(Σ normalized supports of class c)`, as in T3A. Their softmax replaces (or is blended with) the classifier probabilities as the LAME prior. Then the unchanged k = 20 LAME correction runs.
- Only already-served batches enter the support set; returned predictions are never revised.

## Predeclared runs

Official validation harness, supplied checkpoint, seed 0, diagnostics off, one commit:

| Run | M per class | Prior | Role |
| --- | ---: | --- | --- |
| `lame8k20_ema03` | — | source classifier | same-commit control, expect 2019 (the base class gained identity hooks) |
| `lame_t3a_m20` | 20 | T3A prototypes | candidate |
| `lame_t3a_m100` | 100 | T3A prototypes | candidate |
| `lame_t3a_mix_m20` | 20 | 0.5 source + 0.5 T3A | candidate |

## Stopping rule

- A candidate needs **≥ 10 more correct images** than the control.
- At most **one** full test pass for the best qualifying candidate, provided validation time stays close to the control's (≈ 16.1 s).
- A new CSV stays separate until the user reports a public score above 0.89143.

## Results

All runs used code commit `4457d66`, seed 0, diagnostics off, on the local GTX 1080 Ti:

| Run | Correct / 2112 | Top-1 | Δ vs. control | Harness s | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lame8k20_ema03` (control) | 2019 | .955966 | — | 16.0 | 848 |
| `lame_t3a_m20` | 1127 | .533617 | −892 | 16.1 | 852 |
| `lame_t3a_m100` | 1011 | .478693 | −1008 | 16.1 | 852 |
| `lame_t3a_mix_m20` | 2016 | .954545 | −3 | 16.1 | 852 |

The control reproduced 2019, so the new base-class hooks did not change existing methods. The pure T3A prior **collapses below the frozen source model** (1152), and the 50/50 blend is within noise of the control. No candidate qualifies, so no test pass was run.

Label-free diagnosis from the official prediction CSVs: the control predicts all 20 classes, with the largest class at 6–9% in each third of the stream. `t3a_m20` predicts only 14 classes and `t3a_m100` only 10. One class (9) takes 309 and 551 predictions, and the largest class's share in the first third is 25% and 40%. This is a first-come takeover: classes that enter the support set early dominate.

A plausible explanation, not verified: post-ReLU average-pool features are non-negative and share a large common direction. Prototypes built from real features therefore score higher than classes still represented only by classifier weight rows. Under class-correlated streams, classes seen first then absorb later samples. This matches NOTE's (NeurIPS 2022) observation that T3A degrades on temporally correlated streams. A centered-feature variant might avoid the takeover, but the blend result (−3) suggests limited upside; it was not pursued.
