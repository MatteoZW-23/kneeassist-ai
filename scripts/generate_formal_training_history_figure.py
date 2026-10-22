"""Create a transparent history figure from the retained formal two-model run.

This script performs no training.  It reads the saved candidate histories and
model-comparison record used by the dissertation, then writes one figure for
the evidence-aligned report.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "mrnet_repro_20261013" / "results"
OUT = ROOT / "docs" / "academic" / "audit_20261015" / "figures" / "formal_training_history.png"


def load_history(name: str) -> list[dict]:
    with (RUN / name / "history.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_comparison() -> dict:
    with (RUN / "model_comparison.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_history(axis, history: list[dict], title: str) -> None:
    epochs = [record["epoch"] for record in history]
    axis.plot(epochs, [record["training_loss"] for record in history], marker="o", label="Fitting loss")
    axis.plot(epochs, [record["validation_loss"] for record in history], marker="o", label="Internal-tuning loss")
    axis.set_title(title)
    axis.set_xlabel("Retained epoch")
    axis.set_ylabel("Weighted BCE loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def main() -> None:
    resnet = load_history("resnet18")
    efficientnet = load_history("efficientnet_b0")
    comparison = load_comparison()

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    plot_history(axes[0, 0], resnet, "ResNet-18 retained stage")
    plot_history(axes[0, 1], efficientnet, "EfficientNet-B0 retained stage")

    for history, label, colour in [
        (resnet, "ResNet-18", "#1f77b4"),
        (efficientnet, "EfficientNet-B0", "#ff7f0e"),
    ]:
        axes[1, 0].plot(
            [record["epoch"] for record in history],
            [record["validation_auroc"] for record in history],
            marker="o",
            label=label,
            color=colour,
        )
    axes[1, 0].set_title("Internal-tuning macro AUROC by retained epoch")
    axes[1, 0].set_xlabel("Retained epoch")
    axes[1, 0].set_ylabel("Macro AUROC")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    candidates = comparison["models"]
    labels = [candidate["architecture"].replace("_", "-").replace("b0", "B0") for candidate in candidates]
    scores = [candidate["metrics"]["macro"]["auroc"] for candidate in candidates]
    colours = ["#1f77b4", "#ff7f0e"]
    bars = axes[1, 1].bar(labels, scores, color=colours)
    axes[1, 1].set_title("Saved selection metric")
    axes[1, 1].set_ylabel("Internal-tuning macro AUROC")
    axes[1, 1].set_ylim(min(scores) - 0.01, max(scores) + 0.01)
    axes[1, 1].grid(axis="y", alpha=0.25)
    for bar, score in zip(bars, scores):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, score + 0.00025, f"{score:.6f}", ha="center", va="bottom", fontsize=8)

    figure.suptitle(
        "Formal ResNet-18 and EfficientNet-B0 histories from saved MRNet training artifacts\n"
        "Retained stages differ; the figure is descriptive and does not establish architecture-only causality.",
        fontsize=12,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(OUT)


if __name__ == "__main__":
    main()
