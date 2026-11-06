# HW2 report source notes (Q1–Q4)

This is editable report content, not a PDF. Numbers are from the official validation harness unless marked as model-prediction diagnostics. Test accuracy is unknown.

## Q1. Two different image-classification TTA settings

**COME — standard / fully test-time adaptation.** Zhang et al., *COME: Test-time Adaption by Conservatively Minimizing Entropy*, ICLR 2025 ([paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html), [author code](https://github.com/BlueWhaleLab/COME)). Conventional test-time entropy minimization can become overconfident on unreliable target samples and destabilize adaptation. COME replaces softmax entropy with an opinion distribution over the classes plus explicit uncertainty: class belief `b_c = exp(z_c)/(Σ_j exp(z_j)+K)` and uncertainty `u = K/(Σ_j exp(z_j)+K)`. It minimizes entropy over `(b_1,…,b_C,u)` and modifies the gradient through a norm operation. The loss can be used within a standard fully TTA update without changing the classifier. Its strengths are a small code change and explicit uncertainty; limitations include the scale `K`, continued dependence on online gradient updates, and no guarantee against correlated or continuously shifting batches. The paper also evaluates lifelong and open-world settings; Q1 discusses its standard fully TTA protocol to contrast with GOLD's continual setting.

**GOLD — continual test-time adaptation.** Lai et al., *The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation*, CVPR 2026 ([paper](https://arxiv.org/abs/2603.21928), [author code](https://github.com/AIGNLAI/GOLD)). Continual TTA faces changing target distributions and a cost-versus-generalization trade-off. GOLD identifies a low-dimensional feature subspace related to the pretrained classifier row space, updates it online with average gradient outer products, and learns a small scaling vector in that subspace. Its appeal is few trainable adapter parameters while retaining a changing adaptation direction. The released classification implementation also uses source prototypes, source warm-up, EMA teacher, augmentation, and contrastive loss. Those parts complicate the method and the source-data-dependent ones are prohibited in this assignment; our `gold_sf` experiment is explicitly a modified source-free component, **not** a reproduction of full GOLD. The paper's reported gains and resource use do not establish HW2 performance.

## Q2. Submitted method

The current test artifact uses the supplied 20-class ResNet-50 with current-view BatchNorm statistics and an eight-view prediction ensemble (`bn_view8`). For each official batch of 64 normalized 160×160 images, four center-crop scales (100%, 90%, 80%, 70%) are resized to 160×160, each with an original and horizontal-flip view. The eight views are inferred sequentially; their softmax probability vectors are summed and the highest-scoring class is returned. The model uses **no optimizer update** and no future batch, image ID, labels, extra weights, source images, or replay. The earlier Tent lr .003 artifact scored public 0.72005. The ten-view paired version reached 0.834754 validation but public 0.76627; this eight-view version reached 0.815341 validation and public **0.76822**, as reported by the user. Its full local GTX 1080 Ti test took 87.3 seconds and 706 MiB; RTX 3090 runtime remains unverified. Full command, code commit, and checksum are in the staged `Readme.md`.

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
| BN eight-view | **1722** | **.815341** | **+26.9886** | 15.3 | 706 | Current public 0.76822; full local test 87.3 s |
| BN ten-view sequential | 1759 | .832860 | +28.7405 | 18.9 | 706 | Adds 60% center zoom and flip |
| BN ten-view paired | 1763 | .834754 | +28.9299 | 19.0 | 1164 | Public 0.76627; full local test 109.3 s |
| LAME-inspired output correction, one BN view | 1594 | .754735 | +20.9280 | 2.7 | 706 | Graph correction improves on one-view BN |
| LAME-inspired output correction, six BN views | 1772 | .839015 | +29.3561 | 11.7 | 808 | Separate full test 66.2 s; public pending |

The frozen model's adjacent **predicted** same-class rate was .203; Tent reduced it to .134. This is only a prediction-sequence diagnostic, not per-window ground-truth accuracy. The original fixed-model permutation check found the .2035 local-256-window rate above all 10,000 fixed-window-class-count shuffles (null mean .1384), motivating a correlated-batch ablation. The .25 gate's negative result does not prove the true class sequence lacks correlation. The eight-view method improved validation by 192 images over Tent lr .003. The deeper ten-view method improved validation by another 41 images, yet scored 0.00195 lower on the observed public subset. This demonstrates the limits of validation-only model selection here; private accuracy remains unknown. Timings and memory above are local GTX 1080 Ti results and do not certify the RTX 3090 rerun. The complete experiment list, including negative results for post-update Tent and alternative ten-view fusion, is in `q3_results.csv`.

The same-code LR round found .003 improved overall validation by two images over .001. The multiview gain is larger. The current official test CSV and Q4 metadata correspond to `bn_view8`.

## Q4. Eight validation examples

The original slide wording provided by the user explicitly requires eight validation images with predicted and ground-truth labels. `scripts/build_q4_figure.py` used the completed official `bn_view8` validation prediction CSV, then read validation-only labels and eight selected validation images **after** evaluation for report preparation. The Q4 image labels did not enter the method or test pass; method selection used official aggregate validation and user-reported public scores. The procedure and narrow interpretation of the data rule are recorded in `docs/q4_procedure.md`.

| Result | Stream position | Predicted class | Ground-truth class |
| --- | ---: | --- | --- |
| Correct | 265 | 3 golden retriever | 3 golden retriever |
| Correct | 793 | 15 acoustic guitar | 15 acoustic guitar |
| Correct | 1321 | 14 steam locomotive | 14 steam locomotive |
| Correct | 1849 | 10 sports car | 10 sports car |
| Incorrect | 254 | 6 flamingo | 19 strawberry |
| Incorrect | 791 | 0 Siberian husky | 1 timber wolf |
| Incorrect | 1323 | 9 fire engine | 14 steam locomotive |
| Incorrect | 1848 | 18 teapot | 17 laptop |

The local eight-panel figure is `runs/q4_bn_view8/q4_examples.png` (SHA-256 `39954f4681a44e8b87151ad0d803b5640e038a107a5d14bf6218f300dd1054d3`). A report-only aggregate cross-check reproduced the official 1722/2112 validation count. The figure should be embedded in the report PDF, not submitted as separate validation images.
