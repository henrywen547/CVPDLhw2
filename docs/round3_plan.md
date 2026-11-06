# Round 3: public feedback and low-cost accuracy probes

The user reported a public score of **0.72005** for the official Tent lr .003 test CSV. This is close to official validation **0.7244318 (1530/2112)** and below the assignment's 0.84000 public strong baseline. The public score is feedback on that already submitted artifact; all new method selection below uses official validation only. The local test pass took 28.3 seconds and 3470 MiB peak VRAM on a GTX 1080 Ti. The RTX 3090 limit is 90 seconds and 10 GB, but local timing does not certify it.

The previous P0/P1 diagnosis found current-batch BN statistics account for almost all the measured gain: BN-only scored 1526/2112, and Tent lr .003 scored 1530/2112. This makes the large remaining accuracy gap unlikely to be resolved by small changes to Tent's BN-affine learning rate alone. First test two one-variable alternatives under the official validation harness, each from the original source checkpoint, seed 0, with diagnostics disabled:

| Candidate | Change | Purpose |
| --- | --- | --- |
| `bn_flip` | Average class probabilities from original and horizontally flipped views with current-batch BN; no update. | Check whether a cheap second view recovers errors from the source image. |
| `tent_post` | Apply Tent lr .003 on the current served batch before returning its prediction; use a second forward pass after the update. | Test whether the existing pre-update prediction timing hides an immediate benefit. The harness permits using the current batch before returning, and no future batch is accessed. |

These are local ablations, not claims to reproduce MEMO or another paper. Stop variants that fail to improve official validation Top-1. A favorable result can motivate a limited augmentation-based method with the same stream contract. Preserve `submission.csv` and the current staged test pair until a candidate shows a meaningful, reproducible validation gain and passes the full test resource check.

## Predeclared follow-up after the two initial probes

`bn_flip` improved official validation to 1562/2112 (.7395833), 4.4 seconds, 706 MiB. `tent_post` remained at 1530/2112 (.7244318), 7.0 seconds, 3470 MiB. The next single-variable extension adds a third view: a 90% center crop resized back to 160×160 and averaged with the original and horizontal-flip probabilities (`bn_flip_zoom`). It tests whether mild scale variation adds value without optimization or extra weights. If it fails to beat `bn_flip`, stop that augmentation branch. This is still an inference ensemble, not a MEMO reproduction.

`bn_flip_zoom` reached **1604/2112 (.7594697)** in 6.2 seconds, 706 MiB. The next fixed grid adds the horizontal flip of that 90% center zoom (`bn_view4`) and then the unflipped and flipped 80% center zooms (`bn_view6`). All views are evaluated sequentially with the same source model and current-view batch BN, and probabilities are equally averaged. These candidates use only the served batch; there is no model update, additional checkpoint, memory bank, or label access. Compare both overall official validation Top-1 and the resource trend. Stop adding views if the gain ceases or the projected full test time approaches the 90-second grading limit.

`bn_view4` reached **1617/2112 (.765625)** in 8.0 seconds, 706 MiB; `bn_view6` reached **1682/2112 (.7964015)** in 11.7 seconds, 706 MiB. Because the 80% scale pair produced a large gain, make one final fixed extension to 70% (`bn_view8`) and 60% (`bn_view10`) center crops, each with horizontal flip. This caps the view search at ten. Select by official validation Top-1 subject to the full test resource limit; do not use public feedback to tune individual scales. The current lr .003 submission remains untouched while these are validation candidates.
