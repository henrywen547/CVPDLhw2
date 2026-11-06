"""Copy a verified official test prediction CSV to Kaggle's submission.csv.

This reads only the official run outputs. It does not rerun or inspect test data.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path,
                        default=REPO / "runs/final_lame_view6/test_pred.csv")
    parser.add_argument("--run-log", type=Path,
                        help="Defaults to test_run_log.json beside --pred")
    parser.add_argument("--out", type=Path, default=REPO / "submission.csv")
    args = parser.parse_args()
    run_log = args.run_log or args.pred.with_name("test_run_log.json")
    log = json.loads(run_log.read_text())
    if log.get("split") != "test" or log.get("n") != 12288:
        parser.error("Run log must describe the official 12,288-image test split")

    content = args.pred.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != log.get("pred_sha256"):
        parser.error("Prediction CSV does not match the paired official run log")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    if reader.fieldnames != ["id", "label"]:
        parser.error("Expected exactly the id,label CSV header")
    rows = list(reader)
    if len(rows) != 12288 or len({row["id"] for row in rows}) != 12288:
        parser.error("Expected 12,288 rows with unique IDs")
    if any(not row["id"] or not row["label"].isdigit() or
           not 0 <= int(row["label"]) <= 19 or
           str(int(row["label"])) != row["label"] for row in rows):
        parser.error("Every label must be an integer in 0..19")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(content)
    print(f"Created {args.out} with {len(rows)} rows; SHA-256 {digest}")


if __name__ == "__main__":
    main()
