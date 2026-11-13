"""Collect editable report notes, tables, and eight validation-only Q4 images."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classes_from_file(path):
    classes = {}
    for line in path.read_text().splitlines():
        index, _, name = line.split(maxsplit=2)
        classes[int(index)] = name.replace("_", " ")
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-val-images-for-q4", action="store_true")
    parser.add_argument("--q4-figure", type=Path, required=True)
    parser.add_argument("--q4-metadata", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_val_images_for_q4 or args.val_images.parent.name != "val":
        parser.error("Explicitly acknowledge Q4-only access to validation images")
    metadata = json.loads(args.q4_metadata.read_text())
    selected = metadata["selected"]
    if len(selected) != 8 or sum(item["correct"] for item in selected) != 4:
        parser.error("Q4 metadata must contain four correct and four incorrect examples")

    repo = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "q4_individual_images"
    image_dir.mkdir(exist_ok=True)
    for previous in image_dir.glob("*.png"):
        previous.unlink()
    for relative in ("docs/report_draft.md", "docs/q3_results.csv",
                     "docs/paper_method_review.md", "docs/data_exploration.md",
                     "docs/q4_procedure.md", "docs/experiments.md",
                     "docs/tent_diagnostic_round.md", "docs/round3_plan.md",
                     "docs/round4_unmix_plan.md", "docs/round4_lame_plan.md",
                     "docs/round5_view_sensitivity.md"):
        shutil.copy2(repo / relative, args.out / Path(relative).name)
    shutil.copy2(args.q4_figure, args.out / "q4_examples.png")
    shutil.copy2(args.q4_metadata, args.out / "q4_examples.json")

    classes = classes_from_file(args.classes)
    table = ["| Image file | Result | Stream position | Predicted | Ground truth |",
             "| --- | --- | ---: | --- | --- |"]
    for number, item in enumerate(selected, start=1):
        filename = f"{number:02d}_{'correct' if item['correct'] else 'incorrect'}_p{item['position']:04d}.png"
        source = args.val_images / f"{item['id']}.png"
        with Image.open(source) as image:
            if image.size != (160, 160):
                raise ValueError(f"Unexpected validation image size: {source}")
        shutil.copy2(source, image_dir / filename)
        table.append(f"| [{filename}](q4_individual_images/{filename}) | "
                     f"{'Correct' if item['correct'] else 'Incorrect'} | {item['position']} | "
                     f"{item['prediction']} {classes[item['prediction']]} | "
                     f"{item['truth']} {classes[item['truth']]} |")

    paper_sources = (
        ("LAME_CVPR2022.pdf", "https://openaccess.thecvf.com/content/CVPR2022/papers/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.pdf"),
        ("COME_ICLR2025.pdf", "https://proceedings.iclr.cc/paper_files/paper/2025/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf"),
        ("GOLD_CVPR2026.pdf", "https://arxiv.org/pdf/2603.21928"),
    )
    papers = []
    for filename, url in paper_sources:
        paper = args.out / "papers" / filename
        if paper.exists():
            papers.append(f"- [`papers/{filename}`](papers/{filename}) — [original source]({url}); SHA-256 `{sha256(paper)}`.")

    readme = "\n".join([
        "# CVPDL HW2 report materials",
        "",
        "These are source materials for the user's report. No PDF was generated here.",
        "Q4 images and ground truth were used only after official validation predictions",
        "for the original slide's report-only visualization requirement.",
        "They never entered model adaptation or the official test pass.",
        "",
        "- `report_draft.md`: editable Q1–Q4 text and source links.",
        "- `paper_method_review.md`: paper code, upstream revisions, and HW2 restrictions.",
        "- `q3_results.csv`: official validation comparison table for charting.",
        "- `experiments.md`: official validation and test run record.",
        "- `tent_diagnostic_round.md`: same-code P0/P1 diagnosis and LR grid.",
        "- `round3_plan.md`: public feedback, bounded multiview experiments, and resource checks.",
        "- `round4_unmix_plan.md`: UnMix-TNS normalization-only test and negative result.",
        "- `round4_lame_plan.md`: LAME-inspired graph output correction and validation result.",
        "- `round5_view_sensitivity.md`: bounded four/eight-view comparison and selected eight-view method.",
        "- `data_exploration.md`: validation diagnostics and permutation results.",
        "- `q4_procedure.md`: exact validation-only selection procedure.",
        "- `q4_examples.png`: ready-to-insert eight-panel Q4 figure.",
        "- `q4_examples.json`: selected IDs, positions, predicted and true indices.",
        "- `q4_individual_images/`: the same eight original validation PNGs for custom layout.",
        "- `papers/`: original LAME and COME PDFs (Q1 papers) plus GOLD (`gold_sf` ablation only), when downloaded.",
        "",
        "## Q1 papers",
        "",
        *papers,
        "",
        f"Q4 figure SHA-256: `{sha256(args.q4_figure)}`.",
        f"Official validation prediction CSV used for Q4 SHA-256: `{metadata['prediction_sha256']}`.",
        "",
        "## Q4 images",
        "",
        *table,
        "",
        "Do not include the validation dataset or these eight source PNGs separately in the COOL ZIP;",
        "only place the chosen figure or images inside the final report PDF.",
        "Do not include the paper PDFs as separate COOL ZIP files.",
        "",
    ])
    (args.out / "README.md").write_text(readme)
    print(args.out)
    print("collected_files", len(list(args.out.rglob("*"))))


if __name__ == "__main__":
    main()
