"""Audit whether local fastMRI files can be matched to fastMRI+ annotations.

The tool creates a provenance summary only. A filename overlap is not proof
that the local image reconstruction, coil acquisition and orientation are the
same as the data used for the published annotations. Do not use its output to
train a model until that identity check is documented and a protocol is locked.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def read_reviewed_ids(path: Path) -> set[str]:
    """Read the public one-column file list, with or without a header."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]
    if values and values[0].lower() in {"file", "filename"}:
        values = values[1:]
    return set(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fastMRI+ image/annotation filename overlap.")
    parser.add_argument("--image-root", type=Path, required=True, help="Folder containing local .h5 fastMRI files.")
    parser.add_argument("--annotations-csv", type=Path, required=True, help="fastMRI+ knee.csv annotation file.")
    parser.add_argument("--reviewed-files-csv", type=Path, help="Optional fastMRI+ knee_file_list.csv file.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON audit report.")
    args = parser.parse_args()

    if not args.image_root.is_dir():
        raise SystemExit(f"Image root does not exist: {args.image_root}")
    if not args.annotations_csv.is_file():
        raise SystemExit(f"Annotation CSV does not exist: {args.annotations_csv}")

    local_files = {path.stem for path in args.image_root.rglob("*.h5")}
    rows = read_csv(args.annotations_csv)
    required = {"file", "slice", "study_level", "x", "y", "width", "height", "label"}
    available = set(rows[0]) if rows else set()
    missing = required - available
    if missing:
        raise SystemExit(f"Annotation CSV is missing required columns: {', '.join(sorted(missing))}")

    annotations_by_file: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        annotations_by_file[row["file"]].append(row)
    annotated_files = set(annotations_by_file)
    matched_annotated = local_files & annotated_files

    reviewed_files: set[str] | None = None
    if args.reviewed_files_csv:
        reviewed_files = read_reviewed_ids(args.reviewed_files_csv)

    boxes_per_label = Counter(row["label"] for row in rows if row["file"] in matched_annotated)
    studies_per_label = {
        label: len({row["file"] for row in rows if row["file"] in matched_annotated and row["label"] == label})
        for label in sorted(boxes_per_label)
    }
    matched_reviewed = local_files & reviewed_files if reviewed_files is not None else set()
    reviewed_without_annotations = matched_reviewed - annotated_files if reviewed_files is not None else set()

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fastMRI+ filename and label-coverage audit; not a training approval",
        "image_root": str(args.image_root),
        "annotations_csv": str(args.annotations_csv),
        "local_h5_files": len(local_files),
        "annotation_rows": len(rows),
        "annotated_files_in_source": len(annotated_files),
        "matched_annotated_files": len(matched_annotated),
        "matched_annotation_boxes": sum(boxes_per_label.values()),
        "matched_reviewed_files": len(matched_reviewed) if reviewed_files is not None else None,
        "matched_reviewed_files_without_annotations": len(reviewed_without_annotations) if reviewed_files is not None else None,
        "boxes_per_label_on_matched_files": dict(sorted(boxes_per_label.items())),
        "studies_per_label_on_matched_files": dict(sorted(studies_per_label.items())),
        "mandatory_gate_before_training": [
            "Verify the local files are the same fastMRI cohort and compatible reconstruction/orientation as the annotation source.",
            "Document how box labels become study-level labels; a box label is not automatically a complete clinical reference standard.",
            "Create a patient-level split and reserve an untouched evaluation set.",
            "Choose targets with enough positive and negative studies, then lock the protocol before training.",
            "Do not claim clinical utility or junior-doctor benefit without an independent reader/workflow study.",
        ],
        "training_ready": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
