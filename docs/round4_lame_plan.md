# Round 4 continuation: LAME-inspired output correction

The normalization-only UnMix-TNS component scored **1413/2112 (.669034)**, below current-batch BN's **1526/2112 (.722538)**. Stop that branch. The current eight-view submission scored public **0.76822** versus ten-view paired public **0.76627**, while validation favored ten views. These observations argue for a method that changes output assignments conservatively rather than expanding crop scales.

[LAME](https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html), Boudiaf et al., CVPR 2022 ([author code](https://github.com/fiveai/LAME)), adjusts model probabilities with a graph of current-batch feature similarities without updating model parameters. The local independent variant in `starter/lame_output.py` uses the supplied ResNet-50's average-pool features, cosine similarity, a symmetric five-neighbor graph, and ten fixed iterations of the paper's output update. It does not load source samples, extra weights, or future batches. The graph choice and fixed iteration count are local settings, so results must be labeled **LAME-inspired**, not a full author-code reproduction.

Predeclare two official validation runs from the source checkpoint, seed 0, no diagnostics: `lame_bn` applies output correction to one current-batch-BN view; `lame_view6` averages six views (original and flip at 100%, 90%, and 80% center crops), then corrects the averaged probabilities with averaged features. Compare with their exact uncorrected baselines, `bn_batch` and `bn_view6`, using official overall Top-1, harness time, peak VRAM, and label-free prediction differences. Only consider a full test pass if a candidate shows a meaningful validation gain with adequate resource headroom. Preserve the current eight-view `submission.csv` and staging files until a new candidate is selected.

## Official validation result

Both methods ran at code commit `06a95fc04d8de1d80a50183438c663773a058d65`, seed 0, with diagnostics disabled on the GTX 1080 Ti:

| Method | Correct / 2112 | Top-1 | Harness seconds | Peak VRAM MiB | Different predictions from exact baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bn_batch` baseline | 1526 | .722538 | 2.6 | 706 | — |
| `lame_bn` | 1594 | .754735 | 2.7 | 706 | 133/2112 |
| `bn_view6` baseline | 1682 | .796402 | 11.7 | 706 | — |
| `lame_view6` | **1772** | **.839015** | **11.7** | **808** | 142/2112 |

The six-view graph method is 50 correct images above the current eight-view validation result and has a similar short-stream cost to six-view inference. This supports a separate official full test pass for `lame_view6` to measure the 12,288-image cost and generate a paired CSV/log. It does **not** establish public or private accuracy. The current eight-view root submission stays in place until the user's public feedback on this new candidate.
