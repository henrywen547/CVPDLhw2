# Candidate review after public 0.89257

Reviewed 2026-11-14. Research only: no new adaptation run or Kaggle submission. Baseline code: `624405e`; selected method: `lame8k20_ema03_src02`. The root CSV and staged CSV both have SHA-256 `56924a4720d801d19191a9907c610d1e759b72e3ccb1dd939a52dbbbeca2fc57`.

## What the existing evidence supports

The latest CLAUDE.md and rounds 6–13 supersede the older view-count discussion. Official validation is 2018/2112 = 0.955492; user-reported public is 0.89257. Full local test is 92.3 seconds / 848 MiB on GTX 1080 Ti; RTX 3090 remains unmeasured. The main gains came from graph correction, neighborhood size, and EMA normalization. Nearby k, EMA, and source-statistic interpolation settings have already been explored. T3A-like support accumulation, previous-batch graph memory, and simplified balanced BN did not help.

The most recent public changes, 0.89143 -> 0.89241 -> 0.89257, are small and were used for model selection. Further adaptive selection against this same public set risks overfitting it. This does not prove the current model has overfit, nor prove that other mechanisms cannot improve it. The scores are deterministic observations on that subset, not repeated random measurements. Their small differences are weak evidence of performance on unseen corruptions. Likewise, the validation/public gap alone does not prove which corruption is harder or identify the failure mechanism. Ninety-four remaining validation errors limit headroom but do not make validation useless.

## First candidate: TACT (NeurIPS 2025)

[Paper](https://papers.neurips.cc/paper_files/paper/2025/hash/09d6e2295eb50b4cdcb45d75220a2598-Abstract-Conference.html), [author repository, MIT](https://github.com/NancyQuris/TACT).

TACT estimates directions that vary across augmentations of the same image and trims their contribution to features and classifier prototypes. This changes representation geometry without backbone gradient updates. Its semantic-preservation assumption is essential: augmentation-varying directions are not automatically non-causal. This mechanism is substantially different from changing BN moments or reinforcing neighboring predictions.

The [author implementation](https://raw.githubusercontent.com/NancyQuris/TACT/main/methods/tact.py) computes a subspace separately for each sample, trims source classifier weight rows, and accumulates the trimmed prototypes. It does not require replacing the classifier with pseudo-labeled target class centroids, the mechanism implicated in our T3A collapse. This is a reason to test it, not evidence that it will beat our baseline. The repository's ImageNet-R/V2 results do not establish performance on this correlated corruption stream.

Implementation considerations from [the utilities](https://raw.githubusercontent.com/NancyQuris/TACT/main/methods/tact_utils.py): the image benchmarks use RandAugment/AutoAugment; the PCA implementation constructs feature-dimensional covariance matrices. For 2048-dimensional features, directly copying that decomposition is unsuitable for our budget. With V views, a thin SVD of the centered V-by-2048 matrix, or a V-by-V Gram eigendecomposition, recovers the nonzero right-singular subspace. Rank must be bounded by V-1 and numerically nonzero singular values.

Proposed HW2 experiment (a local TACT-inspired component, not full reproduction): reuse the eight existing per-image features, remove only the leading nonzero within-image variation direction, and recompute classifier probabilities using the trimmed feature and supplied classifier weights, retaining any classifier bias. Keep the graph and BN settings fixed. First test the representation-only component; accumulating trimmed prototypes is a separate future component, not an additional tuning dimension in this round. Avoid a cross-image PCA, which can remove actual between-class variation.

This reuse is cheap in backbone calls, but is the main scientific caveat: existing crops may remove objects, and per-view BN has different EMA states. The observed feature variation therefore mixes augmentation sensitivity with normalization effects. Call it view-variation trimming, not proven causal separation. A negative result only rejects this inexpensive integration; it does not falsify the original TACT method. A faithful augmentation comparison would need its own matched augmentation-only control and resource assessment.

## Second candidate: ROID (WACV 2024), lower priority

[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Marsden_Universal_Test-Time_Adaptation_Through_Weight_Ensembling_Diversity_Weighting_and_Prior_WACV_2024_paper.pdf), [author code](https://github.com/mariodoebler/test-time-adaptation).

ROID combines certainty/diversity weighting, source/adapted weight ensembling, and prior correction to handle diverse online settings. It is a meaningful alternative to plain Tent; the Tent/COME negatives do not rule it out. However, it requires gradient updates and a different resource allocation. The [implementation](https://raw.githubusercontent.com/mariodoebler/test-time-adaptation/master/classification/methods/roid.py) also has optional augmentation consistency. A full version cannot be assumed to fit beside our eight backbone forwards.

Do not simply multiply a class-prior correction into already sharpened LAME probabilities: this may reinforce the same class-concentration evidence twice. A proper comparison should allocate fewer views to the gradient method and report the associated accuracy/cost tradeoff. Given the deadline and existing strong output-only baseline, prioritize the TACT component before this larger branch. Source weight snapshots must originate solely from the supplied checkpoint; no external model is needed in principle.

## Other screened directions

| Method | Decision | Reason |
| --- | --- | --- |
| [TAST, ICLR 2023](https://github.com/mingukjang/TAST) | Backup, not first | Adds nearest-neighbor self-training in projection ensembles, but still initializes a support set from classifier weights and accumulates target features. Our support-set collapse makes this a higher-risk branch. This is an inference, not a result for TAST. |
| [PEA, ICLR 2026](https://github.com/TheMaXiao/PEA_TTA) | Exclude full method under current assets | Its source mean/covariance alignment requires source statistics. The [author setup](https://raw.githubusercontent.com/TheMaXiao/PEA_TTA/main/README.md) explicitly collects source statistics. BN running means/variances do not provide the missing full cross-channel covariance. |
| [ADAPT, NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/6e9a0a72da9b76c3ebc8cc33ff10ac29-Abstract-Conference.html) | Exclude full method | Gaussian alignment is interesting, but its [paper](https://arxiv.org/abs/2508.15568) uses CLIP priors. An extra CLIP checkpoint is prohibited; removing it would be a different method needing independent justification. |

## Bounded next experiment proposal

This proposal is not an already-executed experiment or a claimed gain.

1. Freeze the current method and public score. Fix k=20, EMA=0.3, source share=0.2, eight views, and ten graph updates for this round. Do not search nearby values again.
2. Before evaluation, commit an implementation and a concrete protocol with only three validation runs: unchanged baseline; rank-one trimming used only to recompute the classifier prior (graph features unchanged); and the same trimming also applied to the graph features. This separates changing classifier evidence from changing neighborhood geometry. Specify deterministic handling of degenerate ranks. No prototype memory in this first round. For a single shared orthogonal projection P, projecting both a feature and its classifier weights is algebraically equivalent to projecting the feature alone: (fP)(WP)^T = fPW^T. Do not present those as different ablations; the original method's averaging of sample-dependent trimmed prototypes is what makes its prototype update different.
3. Check the zero-trimming path reproduces baseline predictions. Run official validation only. A practical advancement gate is at least ten additional correct images relative to the same-code control, with no additional backbone forwards and at most 5% extra local harness time. This is a screening rule, not a significance test. Stop the branch if neither candidate passes; do not sweep ranks or crop strengths after seeing the outcome.
4. At most one qualifying candidate receives a separate official test pass and one public check. Public feedback is external evaluation, not a loop for selecting another rank or weight. Preserve the existing selected artifact. Its observed public advantage still would not certify private improvement.
5. Do not construct a purported fresh holdout by splitting the already-repeatedly-used validation set. No truly untouched labeled set is available here. Fixed prospective experiment limits reduce adaptive selection, but cannot remove that limitation.

The requested 0.986811 target is far above the current public score. None of the reviewed papers establishes that this target is achievable with this checkpoint and stream. Reliable runtime reproduction and report completion remain valuable regardless of whether this bounded new mechanism succeeds.
