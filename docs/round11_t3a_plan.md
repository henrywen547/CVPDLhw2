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

Pending.
