"""Build Q4 examples under the original slide's report-only val/ requirement.

This script is deliberately separate from starter/run.py and adaptation methods.
It reads the already-produced official val prediction CSV, validation-only truth,
and exactly eight selected validation images. It never reads test data.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def read_rows(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"Expected id,label header in {path}")
        rows = [(row["id"], int(row["label"])) for row in reader]
    if len({image_id for image_id, _ in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def read_classes(path):
    classes = {}
    for line in path.read_text().splitlines():
        index, _, name = line.split(maxsplit=2)
        classes[int(index)] = name.replace("_", " ")
    return classes


def choose_examples(predictions, truths):
    n = len(predictions)
    if n < 8:
        raise ValueError("At least eight validation predictions are required")
    chosen = []
    used_indices = set()
    for correct in (True, False):
        used_true_classes = set()
        for quarter in range(4):
            midpoint = (quarter + 0.5) * n / 4
            candidates = [i for i, (image_id, predicted) in enumerate(predictions)
                          if (predicted == truths[image_id]) == correct and i not in used_indices]
            if not candidates:
                raise ValueError("Need at least four correct and four incorrect predictions")
            # Prefer broad stream coverage and distinct true classes; use fixed
            # stream order as a deterministic tie breaker, never for adaptation.
            candidates.sort(key=lambda i: (
                not (quarter * n / 4 <= i < (quarter + 1) * n / 4),
                truths[predictions[i][0]] in used_true_classes,
                abs(i - midpoint),
                i,
            ))
            index = candidates[0]
            image_id, predicted = predictions[index]
            true = truths[image_id]
            chosen.append(dict(position=index + 1, id=image_id, prediction=predicted,
                               truth=true, correct=correct))
            used_indices.add(index)
            used_true_classes.add(true)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-val-files-for-q4", action="store_true",
                        help="Acknowledge report-only val labels/images access under Q4")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_val_files_for_q4:
        parser.error("Explicitly acknowledge report-only Q4 val/ access")
    if args.val_root.name != "val":
        parser.error("--val-root must point to a validation directory named val")
    predictions = read_rows(args.pred)
    truths = dict(read_rows(args.val_root / "labels.csv"))
    if any(image_id not in truths for image_id, _ in predictions):
        raise ValueError("Official predictions contain IDs absent from validation labels")
    classes = read_classes(args.classes)
    selected = choose_examples(predictions, truths)
    figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for axis, item in zip(axes.flat, selected):
        image_path = args.val_root / "images" / f"{item['id']}.png"
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(("Correct" if item["correct"] else "Incorrect") +
                       f" · position {item['position']}", fontsize=10)
        axis.set_xlabel(f"Pred: {item['prediction']} {classes[item['prediction']]}\n"
                        f"True: {item['truth']} {classes[item['truth']]}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    metadata = dict(prediction_sha256=hashlib.sha256(args.pred.read_bytes()).hexdigest(),
                    source_prediction_csv=str(args.pred), selected=selected)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
