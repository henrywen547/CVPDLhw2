"""Render the three-page HW2 report from recorded results and the Q4 figure."""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image


INK = "#19344d"
MUTED = "#4e6376"
ACCENT = "#087e83"
PALE = "#e9f2f3"


def page(number, title, subtitle):
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0, 0.885), 1, 0.115, facecolor=INK, edgecolor="none"))
    axis.text(0.045, 0.952, title, color="white", fontsize=18, weight="bold", va="center")
    axis.text(0.045, 0.905, subtitle, color="#c7e5e5", fontsize=9.5, va="center")
    axis.text(0.955, 0.035, f"{number} / 3", ha="right", color=MUTED, fontsize=9)
    return figure, axis


def wrapped(axis, x, y, text, width, size=10, line_gap=0.031, color=INK):
    for line in textwrap.wrap(text, width=width, break_long_words=False,
                              break_on_hyphens=False):
        axis.text(x, y, line, fontsize=size, color=color, va="top")
        y -= line_gap
    return y


def section(axis, x, y, heading, body, width=63):
    axis.text(x, y, heading, fontsize=10.6, weight="bold", color=ACCENT, va="top")
    return wrapped(axis, x, y - 0.039, body, width, size=9.6, line_gap=0.029) - 0.025


def q1(pdf):
    figure, ax = page(1, "CVPDL 2026 HW2  |  Practical Test-Time Adaptation",
                      "R14725031  ·  Q1: two image-classification papers with different TTA settings")
    ax.text(0.045, 0.853, "COME  ·  Fully / standard TTA", color=INK, fontsize=14, weight="bold")
    ax.text(0.525, 0.853, "GOLD  ·  Continual TTA", color=INK, fontsize=14, weight="bold")
    ax.plot([0.5, 0.5], [0.14, 0.84], color="#d8e4e8", linewidth=1.2)

    y = 0.805
    y = section(ax, 0.045, y, "Problem",
                "Ordinary test-time entropy minimization can become overconfident on unreliable target samples, destabilizing updates and potentially collapsing predictions.", 64)
    y = section(ax, 0.045, y, "Method",
                "COME replaces softmax entropy with opinion entropy over C class beliefs and one uncertainty mass. With logits z and scale K, b_c = exp(z_c)/(sum_j exp(z_j)+K), and u = K/(sum_j exp(z_j)+K). A norm operation changes the gradient while preserving logit values.", 64)
    y = section(ax, 0.045, y, "Strengths",
                "A drop-in loss for standard fully TTA updates; explicit uncertainty modeling and little architectural change.", 64)
    y = section(ax, 0.045, y, "Limitations and HW2 finding",
                "K and optimizer settings matter. It still relies on online gradients and does not directly solve correlated batches. On HW2, Tent+COME scored 1526/2112, two fewer correct than matched Tent.", 64)

    y = 0.805
    y = section(ax, 0.525, y, "Problem",
                "In a changing unlabeled stream, continual TTA must retain generalization while keeping per-batch computation and trainable parameters small.", 64)
    y = section(ax, 0.525, y, "Method",
                "GOLD adapts features in a compact classifier-sensitive subspace. An online average-gradient outer product updates that subspace; a small scaling vector controls its residual feature transform.", 64)
    y = section(ax, 0.525, y, "Strengths",
                "Low-dimensional adaptation offers a path to efficient updates as domains evolve; the subspace itself can change with incoming target data.", 64)
    y = section(ax, 0.525, y, "Limitations and HW2 finding",
                "The released classifier also uses source prototypes, source warm-up, EMA teacher, augmentation, and contrastive loss. Source-data components are disallowed in HW2. Our clearly modified source-free subspace ablation scored 1526/2112; it is not full GOLD.", 64)

    ax.add_patch(Rectangle((0.04, 0.069), 0.92, 0.062, facecolor=PALE, edgecolor="none"))
    ax.text(0.055, 0.103, "References: COME — Zhang et al., ICLR 2025, arxiv.org/abs/2410.10894;", fontsize=8.4,
            color=INK, va="center")
    ax.text(0.055, 0.079, "GOLD — Lai et al., CVPR 2026, arxiv.org/abs/2603.21928. Author code: BlueWhaleLab/COME; AIGNLAI/GOLD.",
            fontsize=8.4, color=INK, va="center")
    pdf.savefig(figure)
    plt.close(figure)


def q2_q3(pdf):
    figure, ax = page(2, "Method and validation evidence",
                      "Q2: submitted adaptation  ·  Q3: official validation comparisons")
    ax.text(0.045, 0.847, "Q2  |  One-step Tent on the supplied ResNet-50", fontsize=13, weight="bold", color=INK)
    lines = [
        "Input: official online batch of 64 normalized 160×160 images; 20 output classes.",
        "State: source checkpoint only; current-batch BN statistics; BN affine weight and bias are trainable.",
        "Loss: mean softmax entropy. Optimizer: SGD, learning rate 0.001, momentum 0.9; seed 0.",
        "Timing: predict the current batch from pre-update logits, then update once; no labels, replay, future data, or reset schedule.",
    ]
    y = 0.807
    for line in lines:
        ax.text(0.058, y, "• " + line, fontsize=9.7, color=INK, va="center")
        y -= 0.043

    ax.text(0.045, 0.61, "Q3  |  Official validation stream (2,112 images; GTX 1080 Ti)",
            fontsize=13, weight="bold", color=INK)
    columns = [(0.06, "Strategy"), (0.49, "Correct"), (0.60, "Top-1"),
               (0.72, "Seconds"), (0.84, "Peak MiB")]
    ax.add_patch(Rectangle((0.045, 0.55), 0.91, 0.044, facecolor=INK, edgecolor="none"))
    for x, title in columns:
        ax.text(x, 0.572, title, color="white", fontsize=9.5, weight="bold", va="center")
    rows = [
        ("Frozen source", "1152", ".545455", "2.7", "706"),
        ("Batch BN statistics only", "1526", ".722538", "2.6", "706"),
        ("Tent  ·  selected", "1528", ".723485", "5.2", "3470"),
        ("Tent + COME opinion entropy", "1526", ".722538", "5.3", "3470"),
        ("Tent + gate at .4 (inactive)", "1528", ".723485", "5.2", "3470"),
        ("Tent + gate at .25 (2 skips)", "1528", ".723485", "5.0", "3370"),
        ("Source-free GOLD subspace*", "1526", ".722538", "2.7", "754"),
    ]
    for index, row in enumerate(rows):
        y = 0.528 - index * 0.054
        ax.add_patch(Rectangle((0.045, y - 0.023), 0.91, 0.052,
                               facecolor=PALE if index == 2 else ("#f6f9fa" if index % 2 else "white"),
                               edgecolor="none"))
        for (x, _), value in zip(columns, row):
            ax.text(x, y, value, fontsize=9.2, color=INK, va="center",
                    weight="bold" if index == 2 else "normal")

    ax.text(0.05, 0.13, "Interpretation", fontsize=10.5, weight="bold", color=ACCENT)
    wrapped(ax, 0.05, 0.105,
            "Tent improved by 376 correct images over frozen source, but only two over BN-only. "
            "COME, the concentration gate, and the modified GOLD component gave no additional overall gain. "
            "The test pass produced 12,288 predictions in 28.2 s and 3470 MiB on this local GPU; test accuracy is unknown.",
            170, size=8.8, line_gap=0.024)
    ax.text(0.05, 0.043, "* Source-free ablation omits source prototypes, teacher, and contrastive loss; not full GOLD. Runtime excludes model startup.",
            fontsize=7.8, color=MUTED)
    pdf.savefig(figure)
    plt.close(figure)


def q4(pdf, image_path):
    figure, ax = page(3, "Q4  |  Eight validation examples",
                      "Top row: four correct  ·  Bottom row: four incorrect  ·  Predicted and ground-truth labels shown under each image")
    image = Image.open(image_path).convert("RGB")
    image_axis = figure.add_axes([0.045, 0.152, 0.91, 0.708])
    image_axis.imshow(image)
    image_axis.axis("off")
    ax.text(0.045, 0.102,
            "Examples come from the committed Tent method's official validation predictions. Validation truth and these eight images",
            fontsize=9.0, color=INK)
    ax.text(0.045, 0.075,
            "were accessed only after evaluation to prepare this report; they never entered adaptation or test prediction.",
            fontsize=9.0, color=INK)
    pdf.savefig(figure)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q4-image", type=Path, required=True)
    parser.add_argument("--q4-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.q4_metadata.read_text())
    selected = metadata["selected"]
    if len(selected) != 8 or sum(item["correct"] for item in selected) != 4:
        parser.error("Q4 requires four correct and four incorrect examples")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out, metadata={"Title": "CVPDL 2026 HW2 Report",
                                    "Author": "R14725031", "Subject": "Practical Test-Time Adaptation"}) as pdf:
        q1(pdf)
        q2_q3(pdf)
        q4(pdf, args.q4_image)
    print(args.out)


if __name__ == "__main__":
    main()
