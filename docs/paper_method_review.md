# Paper method review for the official HW2 stream

Reviewed 2026-11-06. The local implementation is independently written and uses only the supplied checkpoint and official harness batches. The official `harness.py` is unchanged.

| Reference | Upstream HEAD checked | Code license check | HW2 use |
| --- | --- | --- | --- |
| [COME, ICLR 2025](https://github.com/BlueWhaleLab/COME), particularly [tent_come.py](https://github.com/BlueWhaleLab/COME/blob/main/tent_come.py) | `409a19b71f62c765b1a5be62347a9455524ec176` | No repository-root license surfaced and GitHub's `/license` API returned 404 on 2026-11-06. No upstream code copied. | Independently implement its opinion-entropy loss in a Tent-style one-step BN-affine update. |
| [GOLD, CVPR 2026](https://github.com/AIGNLAI/GOLD), particularly [classification/methods/gold.py](https://github.com/AIGNLAI/GOLD/blob/main/classification/methods/gold.py) | `0bd18942e804df7dbccc01609f4f59effcc85ff7` | No repository-root license surfaced and GitHub's `/license` API returned 404 on 2026-11-06. No upstream code copied. | Q1 reading candidate. Full released classification method cannot run under HW2 rules as written. |

## P0/P1 implementation choices

The official method gets only a normalized current batch and returns predictions before the next batch is served. The local `tent` and `tent_come` run exactly one forward/backward/SGD step on each batch. Their returned labels are computed from logits **before** the current update. They update only `BatchNorm2d` affine parameters; BN uses current-batch statistics with running statistics disabled. No labels enter the method. Each validation run loads the official source checkpoint anew. The local initial setting is SGD learning rate 0.001, momentum 0.9, seed 0; COME uses K=1000, as in the authors' Tent+COME code. These optimizer choices are local settings, not a claim of exact reproduction of the paper's full experiment.

The [author implementation](https://raw.githubusercontent.com/BlueWhaleLab/COME/main/tent_come.py) computes opinion entropy from `exp(logits)` and adds `1e-7` to the resulting probabilities. Our independently written loss uses a `log_softmax` over the 20 class logits plus one uncertainty logit `log(K)`, which is algebraically equivalent before the author's tiny additive epsilon and avoids exponential overflow. We retain the author's identity-valued norm operation, which changes the gradient. We will report this as a **Tent+COME loss adaptation**, not as a reproduction of their full benchmark protocol.

P2 adds `tent_gate`, an explicitly custom ablation. It skips the current update when the largest **current predicted** class share exceeds 0.4; predictions for that batch are still returned. The threshold is not part of COME or GOLD. No future batch or validation label is inspected.

## GOLD restriction

The [released classification implementation](https://raw.githubusercontent.com/AIGNLAI/GOLD/main/classification/methods/gold.py) calls a source-data loader, computes or loads class-wise **source prototypes**, performs source warm-up, and uses those prototypes in its contrastive loss. It also creates an EMA copy of the model and processes an augmented view. HW2 forbids source training data and extra pretrained weights, and has strict 90-second/10-GB test limits. The source-data-dependent parts therefore cannot be used. If pursued after P2, only a clearly identified, independently written **source-free subspace component** may be tested; that would be a modified method, not the released GOLD classifier. It must pass official validation accuracy and resource checks before any test run.

The local `gold_sf` ablation follows that limited path. For ResNet-50's 20×2048 classifier weight matrix `W`, it starts with the top eight eigenvectors of `WᵀW`. For each current batch it predicts with a trainable eight-value scaling vector in this subspace, makes one entropy-minimizing SGD update to that vector, and forms the exact classifier-gradient outer product for selected confident predictions: each top-logit feature gradient is the corresponding row of the fixed `W`. Exponential averaging of the 20 class proportions (alpha 0.1) updates the subspace through a 20×20 Gram eigendecomposition. This is algebraically equivalent to updating a `Wᵀ diag(coefficients) W` matrix while avoiding a 2048×2048 eigendecomposition. The backbone is frozen; BN uses only current-batch statistics. The source prototypes, teacher, augmented view, source warm-up, and contrastive losses in released GOLD are **omitted**. The replacement objective is ordinary prediction entropy. It therefore tests only whether a source-free online classifier-sensitive subspace helps this assignment.

## LAME (selected method's source paper)

Boudiaf et al., *Parameter-free Online Test-time Adaptation*, CVPR 2022 ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html), [author code](https://github.com/fiveai/LAME)). The paper freezes the source model and minimizes `Σ KL(z_i‖q_i) − Σ w_ij z_iᵀz_j` over the current batch's assignments. Its concave-convex update `z_ik ∝ q_ik exp(Σ_j w_ij z_jk)` is iterated to convergence; the affinity function is the only hyperparameter. It uses only the current batch and needs no source data, no extra weights, and no gradients, so it is compatible with the HW2 rules as an output-only method.

`starter/lame_output.py` is written independently from the paper's equations; no author code was copied or executed. Local differences:

1. BatchNorm uses current-view batch statistics (running statistics disabled) instead of the frozen source statistics.
2. Probabilities and average-pool features are averaged over 1, 4, 6, or 8 test-time views before correction.
3. The affinity is a symmetric cosine k-nearest-neighbor graph with negative similarities clamped to zero (k = 5 up to round 5; k = 10 in round 6; the selected `lame_view8_k20` uses k = 20).
4. Exactly ten update iterations are run instead of iterating to convergence.

All results are therefore reported as **LAME-inspired**. The selected `lame8k20_ema03` also replaces difference 1 with per-view exponential moving averages of batch BN statistics (momentum 0.3, initialized from each view's first served batch; `starter/robust_bn.py`). This is an independently written variant of the robust BN idea in RoTTA (Yuan et al., *Robust Test-Time Adaptation in Dynamic Scenarios*, CVPR 2023; [author code](https://github.com/BIT-DA/RoTTA)). RoTTA's memory bank, teacher-student training, and reweighting are not used. LAME and COME are the Q1 papers; GOLD remains documented here only for the `gold_sf` ablation.

No paper repository's data download, checkpoint loader, or benchmark runner was executed.
