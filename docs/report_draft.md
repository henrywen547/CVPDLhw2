# HW2 report source notes (Q1–Q4)

This is editable report content, not a PDF. Numbers are from the official validation harness unless marked as model-prediction diagnostics. Test accuracy is unknown.

## Q1. Two different image-classification TTA settings

**COME — standard / fully test-time adaptation.** Zhang et al., *COME: Test-time Adaption by Conservatively Minimizing Entropy*, ICLR 2025 ([paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html), [author code](https://github.com/BlueWhaleLab/COME)). Conventional test-time entropy minimization can become overconfident on unreliable target samples and destabilize adaptation. COME replaces softmax entropy with an opinion distribution over the classes plus explicit uncertainty: class belief `b_c = exp(z_c)/(Σ_j exp(z_j)+K)` and uncertainty `u = K/(Σ_j exp(z_j)+K)`. It minimizes entropy over `(b_1,…,b_C,u)` and modifies the gradient through a norm operation. The loss can be used within a standard fully TTA update without changing the classifier. Its strengths are a small code change and explicit uncertainty; limitations include the scale `K`, continued dependence on online gradient updates, and no guarantee against correlated or continuously shifting batches. The paper also evaluates lifelong and open-world settings; Q1 discusses its standard fully TTA protocol to contrast with GOLD's continual setting.

**GOLD — continual test-time adaptation.** Lai et al., *The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation*, CVPR 2026 ([paper](https://arxiv.org/abs/2603.21928), [author code](https://github.com/AIGNLAI/GOLD)). Continual TTA faces changing target distributions and a cost-versus-generalization trade-off. GOLD identifies a low-dimensional feature subspace related to the pretrained classifier row space, updates it online with average gradient outer products, and learns a small scaling vector in that subspace. Its appeal is few trainable adapter parameters while retaining a changing adaptation direction. The released classification implementation also uses source prototypes, source warm-up, EMA teacher, augmentation, and contrastive loss. Those parts complicate the method and the source-data-dependent ones are prohibited in this assignment; our `gold_sf` experiment is explicitly a modified source-free component, **not** a reproduction of full GOLD. The paper's reported gains and resource use do not establish HW2 performance.

## Q2. Submitted method

The validation-selected method is one-step Tent-style BatchNorm adaptation of the supplied 20-class ResNet-50. For each official batch of 64 normalized 160×160 images, the model uses current-batch BN statistics and makes one SGD update to only the BN affine weights and biases by minimizing the mean softmax entropy. SGD learning rate is 0.001, momentum 0.9; seed 0. The current batch's labels are predicted from logits **before** the optimizer step; the next batch sees the update. The method uses no source data, validation labels, other weights, replay, scheduled reset, image ID, or future batch. The official test was a single pass. Full command, code commit and checksum are in `Readme.md` of the local submission staging directory.

## Q3. Strategies and official validation results

| Strategy | Correct / 2112 | Top-1 | GTX 1080 Ti harness seconds | Peak VRAM MiB | Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen source | 1152 | .545455 | 2.7 | 706 | Baseline |
| Current-batch BN statistics only | 1526 | .722538 | 2.6 | 706 | Large gain without trainable update |
| Tent, one BN-affine update | **1528** | **.723485** | 5.2 | 3470 | Selected; two more correct than BN-only |
| Tent + COME opinion entropy | 1526 | .722538 | 5.3 | 3470 | Two fewer correct than Tent at matched optimizer settings |
| Tent + custom concentration gate, .4 | 1528 | .723485 | 5.2 | 3470 | Gate inactive: 33/33 batches updated |
| Tent + custom concentration gate, .25 | 1528 | .723485 | 5.0 | 3370 | Skipped 2/33 updates; no accuracy gain |
| Source-free GOLD-inspired subspace | 1526 | .722538 | 2.7 | 754 | Rank-8 online subspace and scale updated; no accuracy gain |

The frozen model's adjacent **predicted** same-class rate was .203; Tent reduced it to .134. This is only a prediction-sequence diagnostic, not per-window ground-truth accuracy. The original fixed-model permutation check found the .2035 local-256-window rate above all 10,000 fixed-window-class-count shuffles (null mean .1384), motivating a correlated-batch ablation. The .25 gate's negative result does not prove the true class sequence lacks correlation. Timings and memory above are local GTX 1080 Ti results; they do not certify the RTX 3090 rerun. The single official test pass produced the required CSV in 28.2 seconds and 3470 MiB on this local GPU, without test accuracy access.

## Q4. Eight validation examples

The original slide wording provided by the user explicitly requires eight validation images with predicted and ground-truth labels. `scripts/build_q4_figure.py` used the completed official Tent validation prediction CSV, then read validation-only labels and eight selected validation images **after** evaluation for report preparation. None of that information entered the adaptation method, validation model selection, or test pass. The procedure and narrow interpretation of the data rule are recorded in `docs/q4_procedure.md`.

| Result | Stream position | Predicted class | Ground-truth class |
| --- | ---: | --- | --- |
| Correct | 265 | 3 golden retriever | 3 golden retriever |
| Correct | 793 | 15 acoustic guitar | 15 acoustic guitar |
| Correct | 1320 | 16 grand piano | 16 grand piano |
| Correct | 1849 | 10 sports car | 10 sports car |
| Incorrect | 254 | 17 laptop | 19 strawberry |
| Incorrect | 791 | 2 red fox | 1 timber wolf |
| Incorrect | 1321 | 13 container ship | 14 steam locomotive |
| Incorrect | 1848 | 8 school bus | 17 laptop |

The local eight-panel figure is `runs/q4_tent/q4_examples.png` (SHA-256 `92af36734b91e742147547a66f8d9cea4efd6815bb53f77f3057c763dafcefc3`). A report-only aggregate cross-check reproduced the official 1528/2112 validation count. The figure should be embedded in the report PDF, not submitted as separate validation images.
