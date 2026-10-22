"""Validate the non-identifying CSV metadata for a future clinical study.

This tool deliberately validates metadata only. It does not read images, remove
DICOM identifiers, establish reference labels or authorise clinical use.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable


TEMPLATE_FILES = {
    "study_manifest.csv": {
        "study_id",
        "source_file_relpath",
        "file_format",
        "anatomy",
        "sequence_plane",
        "sequence_description_generic",
        "num_slices",
        "rows",
        "columns",
        "site_code",
        "scanner_vendor_group",
        "field_strength_t",
        "ingestion_status",
        "reference_standard_status",
        "notes",
    },
    "labels.csv": {
        "study_id",
        "general_abnormality",
        "acl_tear",
        "meniscal_tear",
        "label_status",
        "reference_standard",
        "reader_1_code",
        "reader_2_code",
        "adjudicator_code",
        "label_version",
        "label_notes",
    },
    "quality_control.csv": {
        "study_id",
        "deidentification_verified",
        "knee_anatomy_confirmed",
        "supported_file_format",
        "series_complete",
        "corruption_check_passed",
        "duplicate_check_passed",
        "label_linkage_checked",
        "qc_reviewer_code",
        "qc_status",
        "qc_notes",
    },
    "splits.csv": {
        "study_id",
        "subject_key",
        "split",
        "split_method",
        "leakage_check_status",
        "protocol_lock_version",
        "notes",
    },
}

IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{2,63}$")
PII_TERMS = {
    "patientname", "patient_name", "firstname", "first_name", "lastname",
    "last_name", "mrn", "medicalrecord", "medical_record", "accession",
    "dateofbirth", "date_of_birth", "dob", "address", "phone", "email",
    "governmentid", "nationalid", "dicomuid", "dicom_uid", "originaluid",
    "original_uid", "reporttext", "report_text", "patientid", "patient_id",
}
LABELS = ("general_abnormality", "acl_tear", "meniscal_tear")


def read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return rows, fields


def contains_pii_field(fields: Iterable[str]) -> list[str]:
    findings: list[str] = []
    for field in fields:
        normalised = re.sub(r"[^a-z0-9_]", "", field.lower())
        if any(term in normalised for term in PII_TERMS):
            findings.append(field)
    return findings


def is_safe_relative_path(value: str) -> bool:
    if not value:
        return False
    path = PurePosixPath(value.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts and not re.match(r"^[A-Za-z]:", value)


def unique_values(rows: list[dict[str, str]], field: str) -> tuple[set[str], list[str]]:
    values: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        value = row.get(field, "")
        if not value:
            continue
        if value in values:
            duplicates.append(value)
        values.add(value)
    return values, sorted(set(duplicates))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate KneeAssist clinical dataset manifest metadata.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Private directory containing renamed template CSV files.")
    parser.add_argument("--strict", action="store_true", help="Treat missing rows, pending data and incomplete checks as errors.")
    args = parser.parse_args()

    messages: list[tuple[str, str]] = []
    loaded: dict[str, list[dict[str, str]]] = {}
    for name, expected_fields in TEMPLATE_FILES.items():
        path = args.dataset_dir / name
        if not path.exists():
            messages.append(("ERROR", f"Missing required file: {name}"))
            continue
        rows, fields = read_csv(path)
        loaded[name] = rows
        missing = sorted(expected_fields - fields)
        if missing:
            messages.append(("ERROR", f"{name}: missing columns: {', '.join(missing)}"))
        pii = contains_pii_field(fields)
        if pii:
            messages.append(("ERROR", f"{name}: prohibited identifying-looking columns: {', '.join(pii)}"))
        if not rows:
            messages.append(("ERROR" if args.strict else "WARNING", f"{name}: no data rows yet"))

    manifests = loaded.get("study_manifest.csv", [])
    labels = loaded.get("labels.csv", [])
    quality = loaded.get("quality_control.csv", [])
    splits = loaded.get("splits.csv", [])

    study_ids, duplicate_studies = unique_values(manifests, "study_id")
    if duplicate_studies:
        messages.append(("ERROR", f"study_manifest.csv: duplicate study_id values: {', '.join(duplicate_studies[:10])}"))
    for index, row in enumerate(manifests, start=2):
        study_id = row.get("study_id", "")
        if not IDENTIFIER_RE.match(study_id):
            messages.append(("ERROR", f"study_manifest.csv row {index}: unsafe or missing study_id"))
        if row.get("anatomy", "").lower() not in {"knee", ""}:
            messages.append(("ERROR", f"study_manifest.csv row {index}: anatomy must be knee"))
        if row.get("sequence_plane", "").lower() not in {"axial", "coronal", "sagittal", "multiplane", "unknown", ""}:
            messages.append(("ERROR", f"study_manifest.csv row {index}: unsupported sequence_plane"))
        if not is_safe_relative_path(row.get("source_file_relpath", "")):
            messages.append(("ERROR", f"study_manifest.csv row {index}: source_file_relpath must be a safe relative path"))
        if args.strict and row.get("ingestion_status") != "accepted":
            messages.append(("ERROR", f"study_manifest.csv row {index}: strict mode requires ingestion_status=accepted"))

    for filename, rows in (("labels.csv", labels), ("quality_control.csv", quality), ("splits.csv", splits)):
        ids, duplicates = unique_values(rows, "study_id")
        if duplicates:
            messages.append(("ERROR", f"{filename}: duplicate study_id values: {', '.join(duplicates[:10])}"))
        unknown = sorted(ids - study_ids)
        if unknown:
            messages.append(("ERROR", f"{filename}: IDs not found in study_manifest.csv: {', '.join(unknown[:10])}"))

    for index, row in enumerate(labels, start=2):
        for label in LABELS:
            value = row.get(label, "")
            if value not in {"", "0", "1"}:
                messages.append(("ERROR", f"labels.csv row {index}: {label} must be blank, 0 or 1"))
        status = row.get("label_status", "")
        if status not in {"pending", "single_reader", "consensus", "adjudicated", ""}:
            messages.append(("ERROR", f"labels.csv row {index}: invalid label_status"))
        if args.strict and status not in {"consensus", "adjudicated"}:
            messages.append(("ERROR", f"labels.csv row {index}: strict mode requires consensus or adjudicated labels"))
        if status in {"consensus", "adjudicated"} and not row.get("reference_standard"):
            messages.append(("ERROR", f"labels.csv row {index}: missing reference_standard for completed label"))

    for index, row in enumerate(quality, start=2):
        for field in (
            "deidentification_verified", "knee_anatomy_confirmed", "supported_file_format",
            "series_complete", "corruption_check_passed", "duplicate_check_passed", "label_linkage_checked",
        ):
            value = row.get(field, "").lower()
            if value not in {"yes", "no", "pending", ""}:
                messages.append(("ERROR", f"quality_control.csv row {index}: {field} must be yes, no or pending"))
        if args.strict and row.get("qc_status") != "pass":
            messages.append(("ERROR", f"quality_control.csv row {index}: strict mode requires qc_status=pass"))

    subject_splits: dict[str, str] = {}
    for index, row in enumerate(splits, start=2):
        subject_key, split = row.get("subject_key", ""), row.get("split", "")
        if not IDENTIFIER_RE.match(subject_key):
            messages.append(("ERROR", f"splits.csv row {index}: unsafe or missing subject_key"))
        if split not in {"development", "validation", "test", "external_test", ""}:
            messages.append(("ERROR", f"splits.csv row {index}: invalid split"))
        prior = subject_splits.get(subject_key)
        if subject_key and prior and prior != split:
            messages.append(("ERROR", f"splits.csv row {index}: subject_key appears in more than one split"))
        subject_splits[subject_key] = split
        if args.strict and row.get("leakage_check_status") != "pass":
            messages.append(("ERROR", f"splits.csv row {index}: strict mode requires leakage_check_status=pass"))

    for severity, message in messages:
        print(f"{severity}: {message}")
    errors = sum(severity == "ERROR" for severity, _ in messages)
    warnings = sum(severity == "WARNING" for severity, _ in messages)
    print(f"\nValidation complete: {errors} error(s), {warnings} warning(s).")
    if not messages:
        print("Metadata structure is valid. This does not establish clinical validation or data approval.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
