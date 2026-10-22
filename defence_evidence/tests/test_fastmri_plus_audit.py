import csv
import json
import subprocess
import sys
from pathlib import Path


def test_fastmri_plus_audit_reports_overlap_and_keeps_training_gate(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "file1000000.h5").write_bytes(b"fixture")
    (image_root / "file9999999.h5").write_bytes(b"fixture")

    labels_path = tmp_path / "knee.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "slice", "study_level", "x", "y", "width", "height", "label"])
        writer.writeheader()
        writer.writerow({"file": "file1000000", "slice": "2", "study_level": "No", "x": "1", "y": "2", "width": "3", "height": "4", "label": "Meniscus Tear"})

    report_path = tmp_path / "audit.json"
    reviewed_path = tmp_path / "knee_file_list.csv"
    reviewed_path.write_text("file1000000\nfile9999999\n", encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "audit_fastmri_plus_alignment.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--image-root", str(image_root), "--annotations-csv", str(labels_path), "--reviewed-files-csv", str(reviewed_path), "--output", str(report_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert report["local_h5_files"] == 2
    assert report["matched_annotated_files"] == 1
    assert report["matched_annotation_boxes"] == 1
    assert report["matched_reviewed_files"] == 2
    assert report["matched_reviewed_files_without_annotations"] == 1
    assert report["studies_per_label_on_matched_files"] == {"Meniscus Tear": 1}
    assert report["training_ready"] is False
