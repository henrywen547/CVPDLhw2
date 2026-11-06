"""Shuffle saved validation predictions within windows to test local adjacency.

Reads only source_diag output produced by the official harness, never data/.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def same_count(labels):
    return int(np.count_nonzero(labels[1:] == labels[:-1]))


def exact_null_mean(labels):
    """Expected adjacent-same fraction for a uniform permutation of fixed counts."""
    counts = np.unique(labels, return_counts=True)[1]
    n = len(labels)
    return float(np.sum(counts * (counts - 1)) / (n * (n - 1)))


def distribution(observed, null):
    return dict(
        observed=float(observed),
        null_mean=float(null.mean()),
        null_std=float(null.std(ddof=1)),
        null_q025=float(np.quantile(null, 0.025)),
        null_median=float(np.median(null)),
        null_q975=float(np.quantile(null, 0.975)),
        null_max=float(null.max()),
        observed_percentile=float(np.mean(null < observed)),
        one_sided_p=float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
    )


def analyze(labels, size, reps, seed):
    rng = np.random.default_rng(seed)
    windows = [labels[i:i + size] for i in range(0, len(labels), size)]
    observed_counts = np.array([same_count(w) for w in windows])
    pairs = np.array([len(w) - 1 for w in windows])
    null_counts = np.empty((reps, len(windows)), dtype=np.int16)
    for replicate in range(reps):
        for i, window in enumerate(windows):
            null_counts[replicate, i] = same_count(rng.permutation(window))
    window_results = []
    for i, window in enumerate(windows):
        result = distribution(observed_counts[i] / pairs[i], null_counts[:, i] / pairs[i])
        result.update(start=i * size + 1, end=i * size + len(window), n=len(window),
                      exact_null_mean=exact_null_mean(window))
        window_results.append(result)
    pooled_null = null_counts.sum(axis=1) / pairs.sum()
    pooled = distribution(observed_counts.sum() / pairs.sum(), pooled_null)
    pooled.update(
        pair_count=int(pairs.sum()),
        exact_null_mean=float(sum(exact_null_mean(w) * (len(w) - 1) for w in windows) / pairs.sum()),
        windows_above_q975=sum(w["observed"] > w["null_q975"] for w in window_results),
    )
    return dict(window_size=size, pooled=pooled, windows=window_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    if args.reps < 1 or any(size < 2 for size in args.window_sizes):
        parser.error("reps must be positive and window sizes must be at least two")
    records = [json.loads(line) for line in args.diagnostics.read_text().splitlines()]
    assert [r["batch"] for r in records] == list(range(len(records)))
    labels = np.array([prediction for r in records for prediction in r["prediction"]], dtype=np.int16)
    assert len(labels) == sum(r["n"] for r in records)
    result = dict(n=len(labels), repetitions=args.reps, seed=args.seed,
                  null="uniform permutation within each window, fixed predicted-class counts; window-boundary pairs excluded",
                  sizes=[analyze(labels, size, args.reps, args.seed) for size in args.window_sizes])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for group in result["sizes"]:
        print(f"Window size {group['window_size']}: pooled {json.dumps(group['pooled'])}")
        print("range n observed null_mean null_95_interval percentile one_sided_p")
        for w in group["windows"]:
            print(f"{w['start']}-{w['end']} {w['n']} {w['observed']:.3f} {w['null_mean']:.3f} "
                  f"[{w['null_q025']:.3f}, {w['null_q975']:.3f}] {w['observed_percentile']:.3f} {w['one_sided_p']:.5f}")


if __name__ == "__main__":
    main()
