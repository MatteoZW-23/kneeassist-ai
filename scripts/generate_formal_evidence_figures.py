"""Generate dissertation figures from retained two-model MRNet evidence.

The figures are derived from saved JSON/CSV records only; this script does not
retrain a model, replace a checkpoint or alter source metrics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "mrnet_repro_20261013"
OUT = ROOT / "docs" / "academic" / "audit_20261015" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = (("abnormal", "General abnormality"), ("acl", "ACL tear"), ("meniscus", "Meniscal tear"))


def main() -> None:
    comparison = json.loads((RUN / "results" / "model_comparison.json").read_text(encoding="utf-8"))
    final = json.loads((RUN / "results" / "final" / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    with (RUN / "results" / "final" / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    names = []
    values = []
    for item in comparison["models"]:
        names.append("ResNet-18" if item["architecture"] == "resnet18" else "EfficientNet-B0")
        values.append(item["metrics"]["macro"]["auroc"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colours = ["#4c78a8", "#1f9d8b"]
    bars = ax.bar(names, values, color=colours, width=0.58)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Internal validation macro AUROC")
    ax.set_title("Formal two-model comparison on the locked MRNet tuning partition")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")
    fig.text(0.5, 0.01, "EfficientNet-B0 was selected by the prespecified tuning macro-AUROC criterion. One run; no superiority test.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "formal_two_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    labels = {key: np.asarray([int(row[f"{key}_label"]) for row in rows]) for key, _ in TARGETS}
    scores = {key: np.asarray([float(row[f"{key}_probability"]) for row in rows]) for key, _ in TARGETS}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    for key, label in TARGETS:
        RocCurveDisplay.from_predictions(labels[key], scores[key], name=label, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(labels[key], scores[key], name=label, ax=axes[1])
    axes[0].set_title("ROC curves: selected EfficientNet-B0")
    axes[1].set_title("Precision-recall curves: selected EfficientNet-B0")
    fig.suptitle("Post-selection MRNet development evaluation (120 studies; not independent external validation)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "formal_evaluation_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8))
    for ax, (key, label) in zip(axes, TARGETS):
        threshold = final[key]["threshold"]
        matrix = confusion_matrix(labels[key], scores[key] >= threshold, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(matrix[r, c]), ha="center", va="center", color="white" if matrix[r, c] > matrix.max() / 2 else "#1b1b1b", fontweight="bold")
        ax.set_xticks([0, 1], ["Below threshold", "At/above threshold"], rotation=20, ha="right")
        ax.set_yticks([0, 1], ["Reference 0", "Reference 1"])
        ax.set_title(f"{label}\nthreshold {threshold:.3f}")
    fig.suptitle("Confusion matrices: selected EfficientNet-B0 on MRNet development evaluation", fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.76, bottom=0.22, wspace=0.40)
    fig.savefig(OUT / "formal_confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "source_run": str(RUN.relative_to(ROOT)),
        "source_comparison": "results/model_comparison.json",
        "source_predictions": "results/final/predictions.csv",
        "source_metrics": "results/final/metrics.json",
        "purpose": "Figures for the formal ResNet-18 versus EfficientNet-B0 dissertation evidence.",
        "limits": "Development evidence only; one run; the 120-study cohort had prior development exposure.",
    }
    (OUT / "formal_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
