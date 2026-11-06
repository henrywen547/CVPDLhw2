"""Summarize official-harness validation outputs; never reads data/."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


def entropy_bits(counts):
    n = sum(counts.values())
    return -sum((v / n) * math.log2(v / n) for v in counts.values() if v)


def summary(rows):
    classes = [r["prediction"] for r in rows]
    counts = Counter(classes)
    n = len(rows)
    same = sum(a == b for a, b in zip(classes, classes[1:]))
    return dict(
        n=n,
        confidence=mean(r["confidence"] for r in rows),
        entropy=mean(r["entropy"] for r in rows),
        intensity=mean(r["intensity"] for r in rows),
        contrast=mean(r["contrast"] for r in rows),
        horizontal_difference=mean(r["horizontal_difference"] for r in rows),
        distinct_classes=len(counts),
        max_class_share=max(counts.values()) / n,
        class_entropy_bits=entropy_bits(counts),
        adjacent_same=same / (n - 1) if n > 1 else 0,
        iid_same=sum((v / n) ** 2 for v in counts.values()),
        class_counts=dict(sorted(counts.items())),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--baseline-pred", type=Path, required=True)
    parser.add_argument("--diagnostic-pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.diagnostics.read_text().splitlines()]
    assert [r["batch"] for r in records] == list(range(len(records)))
    keys = ("prediction", "confidence", "entropy", "intensity", "contrast", "horizontal_difference")
    rows = []
    for batch in records:
        assert all(len(batch[key]) == batch["n"] for key in keys)
        rows.extend(dict(zip(keys, values)) for values in zip(*(batch[key] for key in keys)))

    def read_predictions(path):
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == ["id", "label"]
            return [(row["id"], int(row["label"])) for row in reader]

    baseline = read_predictions(args.baseline_pred)
    diagnostic = read_predictions(args.diagnostic_pred)
    assert baseline == diagnostic, "Diagnostic mode changed official predictions"
    assert [label for _, label in diagnostic] == [r["prediction"] for r in rows]

    output = dict(
        n=len(rows),
        batches=[batch["n"] for batch in records],
        predictions_identical=True,
        overall=summary(rows),
        batch=[dict(start=i * 64 + 1, end=i * 64 + batch["n"], **summary(rows[i * 64:i * 64 + batch["n"]])) for i, batch in enumerate(records)],
        windows={str(size): [dict(start=start + 1, end=min(start + size, len(rows)), **summary(rows[start:start + size])) for start in range(0, len(rows), size)] for size in (128, 256)},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ("batch", "windows")}, indent=2))
    for size, windows in output["windows"].items():
        print(f"\nWindow {size}")
        print("start-end n confidence entropy intensity contrast horizontal_difference distinct max_share class_entropy adjacent_same iid_same")
        for w in windows:
            print(f"{w['start']}-{w['end']} {w['n']} {w['confidence']:.3f} {w['entropy']:.3f} {w['intensity']:.3f} {w['contrast']:.3f} {w['horizontal_difference']:.3f} {w['distinct_classes']} {w['max_class_share']:.3f} {w['class_entropy_bits']:.3f} {w['adjacent_same']:.3f} {w['iid_same']:.3f}")


if __name__ == "__main__":
    main()
