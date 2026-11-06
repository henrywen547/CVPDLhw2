"""Compare prediction proxies and official metrics without reading data/."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def read_predictions(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["id", "label"]
        return [(row["id"], int(row["label"])) for row in reader]


def proxy(labels):
    counts = Counter(labels)
    return dict(n=len(labels), adjacent_same=sum(a == b for a, b in zip(labels, labels[1:])) / (len(labels) - 1),
                max_class_share=max(counts.values()) / len(labels), distinct_classes=len(counts))


def parse_named(values):
    return dict(value.split("=", 1) for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="name=run_directory")
    parser.add_argument("--diagnostic", action="append", default=[], help="name=jsonl_path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = parse_named(args.run)
    diagnostics = parse_named(args.diagnostic)
    output = {}
    common_ids = None
    for name, location in runs.items():
        folder = Path(location)
        pairs = read_predictions(folder / "val_pred.csv")
        ids, labels = zip(*pairs)
        if common_ids is None:
            common_ids = ids
        assert ids == common_ids, "Methods must use the same official stream order"
        log = json.loads((folder / "val_run_log.json").read_text())
        batch = [proxy(labels[i:i + 64]) for i in range(0, len(labels), 64)]
        windows = [dict(start=i + 1, end=min(i + 256, len(labels)), **proxy(labels[i:i + 256]))
                   for i in range(0, len(labels), 256)]
        result = dict(acc=log["acc"], seconds=log["seconds"], peak_vram_mib=log["peak_vram_mib"],
                      config=log["config"], overall=proxy(labels),
                      mean_batch_max_class_share=mean(item["max_class_share"] for item in batch),
                      max_batch_max_class_share=max(item["max_class_share"] for item in batch),
                      windows_256=windows)
        if name in diagnostics:
            records = [json.loads(line) for line in Path(diagnostics[name]).read_text().splitlines()]
            assert [r["batch"] for r in records] == list(range(len(records)))
            assert [p for record in records for p in record["prediction"]] == list(labels)
            confidence = [v for record in records for v in record["confidence"]]
            entropy = [v for record in records for v in record["entropy"]]
            assert len(confidence) == len(labels)
            result["mean_confidence"] = mean(confidence)
            result["mean_entropy"] = mean(entropy)
            if all("updated" in r for r in records):
                result["updated_batches"] = sum(r["updated"] for r in records)
        output[name] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print("method acc seconds peak_mib adj_same batch_max_share mean_confidence mean_entropy updated_batches")
    for name, item in output.items():
        print(name, f"{item['acc']:.5f}", item["seconds"], item["peak_vram_mib"],
              f"{item['overall']['adjacent_same']:.3f}", f"{item['mean_batch_max_class_share']:.3f}",
              f"{item.get('mean_confidence', float('nan')):.3f}",
              f"{item.get('mean_entropy', float('nan')):.3f}", item.get("updated_batches", "-"))


if __name__ == "__main__":
    main()
