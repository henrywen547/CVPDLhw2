# HW2 report source notes (Q1–Q4)

This is editable report content, not a PDF. Numbers are from the official validation harness unless marked as model-prediction diagnostics. Test accuracy is unknown.

## Q1. Two different image-classification TTA settings

Q1 pairs the paper behind the submitted method (LAME, online TTA on non-i.i.d. streams) with a gradient-based fully TTA paper that we also tested (COME). The two papers take opposite positions on whether the network itself should be adapted.

**LAME — online test-time adaptation on non-i.i.d. streams (practical / non-i.i.d. setting).** Boudiaf, Mueller, Ben Ayed, and Bertinetto, *Parameter-free Online Test-time Adaptation*, CVPR 2022 ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html), [author code](https://github.com/fiveai/LAME)).

- *Problem.* The paper targets adaptation that is unsupervised, online, robust to **non-i.i.d.** streams (e.g., temporally correlated video frames) and class-prior shift, needs no source data, and is not tailored to one model. It shows that network adaptation methods such as Tent, SHOT, PseudoLabel, and AdaBN are brittle: entropy minimization on class-correlated batches can *silently* collapse accuracy while prediction entropy keeps decreasing (paper Fig. 1). Hyperparameters tuned for one scenario can degrade accuracy by up to 66 points absolute in another (Fig. 3).
- *Method.* LAME freezes the pretrained model and corrects only its **outputs**. For a batch with source predictions `q_i` and features `φ(x_i)`, it finds simplex assignments `z_i` that minimize `Σ_i KL(z_i ‖ q_i) − Σ_{i,j} w_ij z_iᵀ z_j`, where `w_ij` is a feature affinity such as a kNN graph. The KL term keeps assignments close to the model; the Laplacian term pushes nearby samples toward the same class. A concave-convex procedure yields the closed-form, monotonically improving update `z_ik ∝ q_ik exp(Σ_j w_ij z_jk)`, iterated to convergence. The only hyperparameter is the affinity function.
- *Strengths.* No gradients and no second forward pass, so it runs about twice as fast as Tent and needs about half the memory (Fig. 6). Because parameters never change, it cannot degrade over a long stream. With one fixed hyperparameter set, it improves on the non-adaptive baseline across scenarios, ResNet-50 training procedures (MSRA, torchvision, SimCLR), and architectures (ResNet-18/50/101, EfficientNet-B4, ViT-B). The average gain is 6.7 points in non-i.i.d. scenarios and up to 15 points on ImageNet-V2.
- *Weaknesses.* It does not help in standard i.i.d., class-balanced covariate shift and can fall about 0.5 points below the baseline there. Because the feature extractor is frozen, it cannot repair poor features or accumulate knowledge across batches. It also relies on informative neighbors within a batch and on the manifold-smoothness assumption in the source feature space.
- *Relation to HW2.* The validation permutation test indicated temporally correlated predictions. One-step Tent gained only two images over current-batch BN statistics, whereas LAME-style correction added 68 images with one view and 90 with six views. Our submitted variant differs from the paper in four ways: it normalizes with a per-view moving average of batch BN statistics (a RoTTA-style robust BN) instead of the frozen source BN, averages eight test-time views, uses a cosine twenty-neighbor graph, and runs ten fixed iterations instead of iterating to convergence. It is therefore labeled **LAME-inspired**.

**COME — fully test-time adaptation.** Zhang et al., *COME: Test-time Adaption by Conservatively Minimizing Entropy*, ICLR 2025 ([paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html), [author code](https://github.com/BlueWhaleLab/COME)).

- *Problem.* Conventional test-time entropy minimization, as in Tent, can become overconfident on unreliable target samples and destabilize adaptation.
- *Method.* COME replaces softmax entropy with an opinion distribution over the classes plus explicit uncertainty. The class belief is `b_c = exp(z_c)/(Σ_j exp(z_j)+K)` and the uncertainty is `u = K/(Σ_j exp(z_j)+K)`. It minimizes entropy over `(b_1,…,b_C,u)` and modifies the gradient through a norm operation. The loss drops into a standard fully TTA update (e.g., Tent's BN-affine SGD) without changing the classifier.
- *Strengths.* It is a small code change that adds explicit uncertainty and reduces overconfident updates.
- *Weaknesses.* It adds a scale hyperparameter `K` and still depends on online gradient updates, so it inherits the optimizer sensitivity and backward-pass cost that LAME avoids. It offers no specific protection against class-correlated batches.
- *Setting.* The paper also evaluates lifelong and open-world settings; Q1 discusses its standard fully TTA protocol, which contrasts with LAME's non-i.i.d. online setting.
- *Relation to HW2.* With matched Tent settings (lr .001, momentum .9, K = 1000), Tent+COME scored 1526/2112, two images fewer than Tent, so the conservative loss did not help on this stream.

## Q2. Submitted method

The current test artifact uses the supplied 20-class ResNet-50 with per-view moving-average BatchNorm statistics, eight-view prediction, and a twenty-neighbor output graph (`lame8k20_ema03`).

- **Robust BN statistics.** Every BatchNorm layer keeps one exponential moving average of batch mean and variance per test-time view. Each average is initialized from that view's first served batch and updated with momentum 0.3 before normalizing: `μ ← 0.7 μ + 0.3 μ_batch`, and likewise for the variance. This follows the robust BN idea of RoTTA (Yuan et al., CVPR 2023). It damps the class bias of correlated 64-image batches while still tracking domain changes within a few batches. The checkpoint's affine weights are kept fixed.
- **Views.** For each official batch of 64 normalized 160×160 images, the method runs eight views in sequence: the original and horizontal flip at 100%, 90%, 80%, and 70% center crops, each resized back to 160×160.
- **Fusion.** Softmax probabilities and average-pool features are averaged across views.
- **Output correction.** Within the current batch, the averaged features form a symmetric cosine twenty-neighbor graph (negative similarities clamped to zero). Ten fixed LAME-inspired output-assignment updates, `z ← softmax(log p + W z)`, refine the averaged probabilities before argmax.
- **No loss, no updates.** There is no loss function and **no optimizer update**. The BN averages use only current and previously served batches. The method uses no future batch, batch index, image ID, labels, extra weights, source images, or replay.

This is an independent local variant, not the complete author implementation. It reached **2019/2112 (.955966)** official validation and **public 0.89143**, as reported by the user. The full local GTX 1080 Ti test took **91.6 seconds** and **848 MiB**, slightly above 90 s on this older GPU; the custom BN costs about 5% more than native BN. RTX 3090 runtime remains unverified. Full command, code commit, and checksum are in the staged `Readme.md`.

## Q3. Strategies and official validation results

| Strategy | Correct / 2112 | Top-1 | Gain vs. source (pp) | GTX 1080 Ti harness seconds | Peak VRAM MiB | Finding |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Frozen source | 1152 | .545455 | 0 | 2.7 | 706 | Baseline |
| Current-batch BN statistics only | 1526 | .722538 | +17.7083 | 2.6 | 706 | Large gain without trainable update |
| Tent, one BN-affine update, lr .001 | 1528 | .723485 | +17.8030 | 5.2 | 3470 | Prior test artifact retained |
| Tent, lr .003 | 1530 | .724432 | +17.8977 | 5.1 | 3470 | Earlier public 0.72005, reported by user |
| Tent + COME opinion entropy | 1526 | .722538 | +17.7083 | 5.3 | 3470 | Two fewer correct than Tent at matched optimizer settings |
| Tent + custom concentration gate, .4 | 1528 | .723485 | +17.8030 | 5.2 | 3470 | Gate inactive: 33/33 batches updated |
| Tent + custom concentration gate, .25 | 1528 | .723485 | +17.8030 | 5.0 | 3370 | Skipped 2/33 updates; no accuracy gain |
| Source-free GOLD-inspired subspace | 1526 | .722538 | +17.7083 | 2.7 | 754 | Rank-8 online subspace and scale updated; no accuracy gain |
| BN original and horizontal flip | 1562 | .739583 | +19.4129 | 4.4 | 706 | Two-view inference |
| BN four-view | 1617 | .765625 | +22.0170 | 8.0 | 706 | Adds 90% center zoom and flip |
| BN six-view | 1682 | .796402 | +25.0947 | 11.7 | 706 | Adds 80% center zoom and flip |
| BN eight-view | 1722 | .815341 | +26.9886 | 15.3 | 706 | Earlier public 0.76822; full local test 87.3 s |
| BN ten-view sequential | 1759 | .832860 | +28.7405 | 18.9 | 706 | Adds 60% center zoom and flip |
| BN ten-view paired | 1763 | .834754 | +28.9299 | 19.0 | 1164 | Public 0.76627; full local test 109.3 s |
| LAME-inspired output correction, one BN view | 1594 | .754735 | +20.9280 | 2.7 | 706 | Graph correction improves on one-view BN |
| LAME-inspired output correction, six BN views | 1772 | .839015 | +29.3561 | 11.7 | 808 | Public 0.81266; full local test 66.2 s |
| LAME-inspired output correction, four BN views | 1704 | .806818 | +26.1364 | 8.1 | 768 | Bounded view-count check; no test pass |
| LAME-inspired output correction, eight BN views | 1827 | .865057 | +31.9602 | 15.3 | 848 | Public 0.81835; full local test 87.5 s |
| Eight views, three-neighbor graph | 1779 | .842330 | +29.6875 | 15.3 | 848 | Sparser graph hurts |
| Eight views, ten-neighbor graph | 1901 | .900095 | +35.4640 | 15.3 | 848 | Public 0.87483; full local test 87.5 s |
| Eight views, five neighbors + previous-batch fixed nodes | 1769 | .837595 | +29.2140 | 15.4 | 848 | Cross-batch memory hurts |
| Eight views, 15-neighbor graph | 1967 | .931345 | +38.5890 | 15.4 | 848 | Round 7 |
| Eight views, 20-neighbor graph | 1987 | .940814 | +39.5360 | 15.4 | 848 | Public 0.87906; full local test 87.5 s |
| Eight views, 30-neighbor graph | 1974 | .934659 | +38.9205 | 15.4 | 848 | Validation declines past k = 20 |
| k = 20 + BN 0.9 batch / 0.1 source statistics | 1995 | .944602 | +39.9148 | 16.1 | 848 | Round 8 |
| k = 20 + BN 0.7 batch / 0.3 source statistics | 2011 | .952178 | +40.6723 | 16.1 | 848 | Source prior helps on validation |
| k = 20 + per-view EMA BN, momentum 0.3 | **2019** | **.955966** | **+41.0511** | 16.1 | 848 | **Submitted; public 0.89143**; full local test 91.6 s |
| k = 20 + per-view EMA BN, momentum 0.1 | 1962 | .928977 | +38.3523 | 16.1 | 848 | Too slow to follow domain changes |

The frozen model's adjacent **predicted** same-class rate was .203; Tent reduced it to .134. This is only a prediction-sequence diagnostic, not per-window ground-truth accuracy. The original fixed-model permutation check found the .2035 local-256-window rate above all 10,000 fixed-window-class-count shuffles (null mean .1384), motivating a correlated-batch ablation. The .25 gate's negative result does not prove the true class sequence lacks correlation. The eight-view method improved validation by 192 images over Tent lr .003. The deeper ten-view method improved validation by another 41 images, yet scored 0.00195 lower on the observed public subset. This demonstrates the limits of validation-only model selection here; private accuracy remains unknown. Timings and memory above are local GTX 1080 Ti results and do not certify the RTX 3090 rerun. The complete experiment list, including negative results for post-update Tent and alternative ten-view fusion, is in `q3_results.csv`.

The same-code LR round found .003 improved overall validation by two images over .001. The multiview gain is larger. LAME-inspired correction gave the largest single-step public gain (eight-view BN 0.76822 → six-view LAME 0.81266). Eight LAME views then added 55 validation images but only 0.00569 public accuracy. This is another sign that validation gains from extra crop scales transfer only partly to the test stream. By contrast, a denser graph at the same cost (ten instead of five neighbors) gained 74 validation images and 0.05648 public accuracy, exceeding the 0.84 strong baseline. Graph density mattered more than the number of views. Raising k to 20 added another 86 validation images but only 0.00423 public accuracy (0.87906), so the density gain is saturating on the test stream. A fixed memory of the previous batch's refined outputs hurt, so the output correction stays per-batch. Smoothing the **BN statistics** across batches instead (per-view EMA, momentum 0.3) added 32 validation images and 0.01237 public accuracy (0.89143). It addresses the non-i.i.d. batches at the normalization layer rather than at the output; a slower momentum of 0.1 hurt, so responsiveness to domain change matters. The current root official test CSV, staged COOL files, and Q4 metadata correspond to `lame8k20_ema03`.

## Q4. Eight validation examples

The original slide wording provided by the user explicitly requires eight validation images with predicted and ground-truth labels. `scripts/build_q4_figure.py` used the completed official `lame8k20_ema03` validation prediction CSV (`runs/round8_lame8k20_ema03/val_pred.csv`, SHA-256 `66ad4dae1b8ef33d66a88a386e978e836912c75898acb467f6797cc08a47f217`). Only **after** evaluation, and only for report preparation, did it read validation-only labels and the eight selected validation images. The Q4 image labels did not enter the method or test pass; method selection used official aggregate validation and user-reported public scores. The procedure and narrow interpretation of the data rule are recorded in `docs/q4_procedure.md`.

| Result | Stream position | Predicted class | Ground-truth class |
| --- | ---: | --- | --- |
| Correct | 265 | 3 golden retriever | 3 golden retriever |
| Correct | 793 | 15 acoustic guitar | 15 acoustic guitar |
| Correct | 1321 | 14 steam locomotive | 14 steam locomotive |
| Correct | 1849 | 10 sports car | 10 sports car |
| Incorrect | 332 | 13 container ship | 10 sports car |
| Incorrect | 791 | 2 red fox | 1 timber wolf |
| Incorrect | 1303 | 14 steam locomotive | 16 grand piano |
| Incorrect | 1859 | 14 steam locomotive | 17 laptop |

The local eight-panel figure is `runs/q4_lame8k20_ema03/q4_examples.png` (SHA-256 `778c2d4bbb28aa77df2bf1e840b566cbe225c0222eb1505fce0ee9a628fdedcd`). The figure should be embedded in the report PDF, not submitted as separate validation images.
