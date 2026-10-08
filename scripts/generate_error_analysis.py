"""Generate a transparent development error-analysis queue from saved predictions."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ROOT

RESULTS = ROOT / "runs/mri_finetune_02/results/final"
OUTPUT = RESULTS / "error_analysis"
TARGETS = [
    ("abnormal", "General abnormality"),
    ("acl", "ACL tear"),
    ("meniscus", "Meniscal tear"),
]


def category(label: int, probability: float, threshold: float) -> str:
    predicted = probability >= threshold
    if label == 1 and predicted:
        return "true_positive"
    if label == 0 and not predicted:
        return "true_negative"
    if label == 0:
        return "false_positive"
    return "false_negative"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    with (RESULTS / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))

    report = {
        "dataset": "MRNet-v1.0",
        "partition": "official_valid",
        "purpose": "Development error analysis. This cohort was previously used in development and is not an independent clinical test.",
        "selection_rule": "For each target and outcome category, select up to two studies closest to the decision threshold. No cases were selected for visual appearance or apparent model success.",
        "targets": {},
        "limitations": [
            "Reference labels are study-level labels; they do not identify a lesion slice or location.",
            "Patient linkage is unavailable in this MRNet copy.",
            "Grad-CAM is a classifier explanation and not ground-truth lesion localisation.",
        ],
    }
    csv_rows = []
    for key, display in TARGETS:
        threshold = float(metrics[key]["threshold"])
        grouped: dict[str, list[dict]] = {name: [] for name in ("true_positive", "true_negative", "false_positive", "false_negative")}
        for row in predictions:
            label = int(row[f"{key}_label"])
            probability = float(row[f"{key}_probability"])
            outcome = category(label, probability, threshold)
            grouped[outcome].append({
                "study_id": row["study_id"],
                "reference_label": label,
                "probability": probability,
                "threshold": threshold,
                "outcome": outcome,
                "distance_from_threshold": abs(probability - threshold),
            })
        selected = {}
        for outcome, items in grouped.items():
            chosen = sorted(items, key=lambda item: (item["distance_from_threshold"], item["study_id"]))[:2]
            selected[outcome] = chosen
            for item in chosen:
                csv_rows.append({"target": key, "finding": display, **item})
        report["targets"][key] = {
            "display": display,
            "threshold": threshold,
            "validation_confusion_matrix": metrics[key]["confusion_matrix"],
            "selected_cases": selected,
        }
    (OUTPUT / "error_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["target", "finding", "outcome", "study_id", "reference_label", "probability", "threshold", "distance_from_threshold"]
    with (OUTPUT / "review_queue.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Development error analysis",
        "",
        "This report is generated from saved MRNet official-validation predictions. It is development evidence, not an independent clinical test.",
        "",
        "Cases were selected automatically by closeness to the locked decision threshold. They were not selected for visual appearance, diagnosis confidence, or a desired result.",
        "",
        "## Review method",
        "",
        "For each target, review the selected true positives, true negatives, false positives and false negatives alongside the original MRI study, the study-level reference label, the model probability and the Grad-CAM explanation. Do not describe a heatmap as a confirmed lesion.",
        "",
        "## Selected review cases",
        "",
        "| Finding | Outcome | Study ID | Reference label | Probability | Threshold |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        lines.append(f"| {row['finding']} | {row['outcome'].replace('_', ' ')} | {row['study_id']} | {row['reference_label']} | {row['probability']:.3f} | {row['threshold']:.3f} |")
    lines += [
        "",
        "## Limits of interpretation",
        "",
        "- The MRNet labels are study-level reference labels, so they do not prove where an abnormality appears on a displayed slice.",
        "- A false positive or false negative must be discussed as a model-output disagreement with the reference label, not as a clinical diagnosis error without radiologist review.",
        "- The official validation cohort has been used previously in development; results must not be presented as a pristine independent test.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_cases": len(csv_rows), "output": str(OUTPUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
