# Remaining directions after round 14

Research review, 2026-11-16; no new adaptation or test runs. Current code `fb2b96b`, selected artifact `lame8k20_ema03_src02`, public 0.89257, validation 2018/2112, local GTX 1080 Ti full test 92.3 seconds / 848 MiB. RTX 3090 runtime remains unmeasured. This review supersedes the earlier proposal to try TACT: round 14 has completed and that branch is closed.

## Updated user target

The user reports a leaderboard mean of 0.913874396 and standard deviation 0.050095068 after excluding outliers. The gap is 0.021304396 (2.13044 percentage points), approximately 131 additional net correct predictions on a 6,144-image public half. This is about a 19.83% reduction in the current error rate. The leaderboard standard deviation describes submissions, not uncertainty of our method or a significance threshold. The mean can change; these calculations use the user's snapshot.

## Priority 1: LAME graph construction and solver audit

The local `adjust_outputs` uses top-20 cosine-weighted edges, symmetrization, and ten updates. Changing k changes both the selected neighbors and total graph influence. Therefore the earlier k sweep does not isolate graph connectivity from correction strength. Ten steps is a deliberate local setting, not evidence of convergence. Neither observation proves a bug or predicts an accuracy gain.

The [LAME author implementation](https://raw.githubusercontent.com/fiveai/LAME/master/src/adaptation/lame.py) exposes binary kNN, RBF, and linear affinities. Its optimizer allows up to 100 iterations with an energy-change stopping rule. The RBF bandwidth is estimated from within-batch neighbor distances. These differ from the local variant. The [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.pdf) provides the KL-to-model plus pairwise-agreement objective. Its claims do not automatically establish convergence guarantees for our sparse cosine graph.

Recommended bounded first step: an official validation run from the supplied checkpoint with label-free diagnostics inside the method comparing the current ten-step assignment to 100 steps using identical probabilities and affinity. Record assignment residuals, changed argmaxes, and time. A 100-step official candidate can return the latter predictions for aggregate evaluation. Retain the ten-step method as the same-code control. Do not infer accuracy from a smaller residual. If assignments already stabilize by step ten, close this subdirection immediately.

A separate graph-shape candidate can use a symmetric RBF graph with a fixed within-batch bandwidth rule, diagonal zero, and one global scale chosen so its mean row sum equals the baseline graph's mean row sum for that same batch. This data-derived scaling approximately controls average correction strength while changing edge geometry; it does not equate every node's influence. Keep k=20 as the bandwidth neighbor rank, ten iterations, and all backbone settings fixed. This is a custom LAME-inspired ablation, not the author default. Define all tie and zero-bandwidth cases before evaluation, with baseline fallback for degeneracy. Do not subsequently sweep bandwidth or graph strength against public scores.

Why this is worth considering: it tests whether fixed top-k edges miss useful structure or overweight a wrong neighborhood under harder shifts, without additional backbone forwards. It can also fail by adding cross-class edges. No evidence currently establishes which behavior dominates HW2.

## Priority 2: depth-selective BN, not another global mixing sweep

[Channel-Selective Normalization for Label-Shift Robust Test-Time Adaptation, CoLLAs 2024](https://arxiv.org/abs/2402.04958) studies sensitivity of deeper layers and individual channels to label shift. Its [full algorithm](https://arxiv.org/html/2402.04958v2) requires source data to compute class/channel scores; that full implementation is unavailable under the HW2 constraints.

A legal, independent depth-only ablation is nevertheless possible: keep current EMA/source-mixed BN in the stem and `layer1`–`layer3`, while restoring checkpoint running statistics in every BN of `layer4`, including its shortcut BN. Keep the checkpoint affine weights, views, k, and graph solver unchanged. Do not collect source examples or use validation labels to score channels. Call this a depth-selective BN ablation inspired by the paper's hypothesis, not Hybrid-TTN reproduction.

This tests a different hypothesis from rounds 8–10: preserving late semantic activations while adapting early appearance statistics. The prior rounds applied one mixing setting throughout the model. Cost should not require additional forwards; actual timing and accuracy must be measured. Fix the boundary at `layer4` before the run and do not search every possible cut point if it fails. A failure would support stopping this particular depth policy, not all possible selective normalization.

## Reserve: multi-view affinity consensus

Current code constructs one graph from the mean feature across views. A custom alternative constructs normalized cosine affinities independently for each existing view, averages those affinities, then applies the same top-20 selection and output correction. Keep mean-of-softmax classifier probabilities unchanged. This retains feature components instead of trimming them as round 14 did. It adds feature-level matrix products, not backbone forwards. It tests whether view averaging before graph construction obscures neighborhood evidence. It remains an unvalidated engineering hypothesis, not a newly identified paper method, and should not be combined with other changes in the initial screen.

## Exclusions and experiment limits

- Do not reopen the failed TACT integration or nearby k/EMA/source-share sweeps.
- ROID remains lower value under the stated deadline/resource constraints; no reversal of the prior decision to exclude that branch is proposed.
- [DART, TMLR 2025](https://openreview.net/pdf?id=c7AAHdEYz5) is relevant to label shift but trains a refinement module using source training data. It cannot be adopted as released here. The earlier PEA/ADAPT restrictions also remain.
- At most one same-code control and three initial candidates: extended solver, matched-strength RBF graph, and fixed `layer4` source BN. Each candidate changes one component. Multi-view graph consensus stays in reserve, not an automatic extra run.
- Commit exact code, settings, failure handling, and criteria before any formal run. Use only official validation access; keep the harness and stream order unchanged. No data-file inspection or historical test-CSV ensemble.
- Retain the existing practical screen: at least ten more correct validation images than the same-code control and no more than 5% added local time. These are resource/selection rules, not statistical significance guarantees. Validation gains cannot establish the required 131-public-image gain.
- At most one qualifying method receives one separate full test pass and one public check. Do not pick a new bandwidth, boundary, or solver length in response to that public result. A fresh validation split carved from this heavily reused set would not be untouched evidence.
- If no candidate passes, complete runtime verification and report preparation. Do not claim the checkpoint has a proven ceiling, but do not invent expected gains to justify further submissions. The requested mean is a goal, not evidence that any particular method can reach it.
