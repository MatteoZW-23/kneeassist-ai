"""Run a bounded Grad-CAM parameter-sensitivity check on saved MRNet data.

This is a research transparency check, not lesion-localisation validation.  It
compares a Grad-CAM map from the saved EfficientNet-B0 checkpoint with a map
after only the final classifier layer is randomly reinitialised in memory. No
checkpoint, model registry, dataset or application file is changed.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import Predictor


RUN = ROOT / "runs" / "mrnet_repro_20261013"
OUT = ROOT / "docs" / "academic" / "audit_20261015" / "gradcam_sanity"
STUDY = "1130"
TARGET = "abnormal"
PLANE = "sagittal"


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    rank_a = np.argsort(np.argsort(first))
    rank_b = np.argsort(np.argsort(second))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = RUN / "models" / "best_model.pth"
    volumes = {
        plane: np.load(ROOT / "data" / "MRNet-v1.0" / "valid" / plane / f"{STUDY}.npy")
        for plane in ("axial", "coronal", "sagittal")
    }
    predictor = Predictor(path=checkpoint)
    reference = predictor.explain(volumes, TARGET, PLANE)
    index = int(reference["suggested_slice"])
    before = np.asarray(reference["heatmaps"][index], dtype=float)

    torch.manual_seed(20261015)
    final_layer = predictor.model.head[-1]
    original = copy.deepcopy(final_layer.state_dict())
    with torch.no_grad():
        torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.05)
        torch.nn.init.zeros_(final_layer.bias)
    randomized = predictor.explain(volumes, TARGET, PLANE)
    after = np.asarray(randomized["heatmaps"][index], dtype=float)
    final_layer.load_state_dict(original)
    predictor.model.eval()

    result = {
        "purpose": "Grad-CAM parameter-sensitivity check; not lesion-localisation validation.",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "study": STUDY,
        "target": TARGET,
        "plane": PLANE,
        "slice_index_zero_based": int(reference["indices"][index]),
        "slice_position_in_sampled_stack": index,
        "parameter_change": "Final linear classifier layer randomly reinitialised in memory with seed 20261015; original state restored before exit.",
        "reference_has_positive_attention": bool(reference["has_positive_attention"]),
        "randomized_has_positive_attention": bool(randomized["has_positive_attention"]),
        "mean_absolute_difference": float(np.mean(np.abs(before - after))),
        "rank_correlation": rank_correlation(before, after),
        "interpretation": "A changed map after parameter randomisation is a limited sensitivity observation. It does not prove that any attention region is anatomically or clinically correct.",
    }
    (OUT / "gradcam_parameter_sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    original_image = reference["original"][index]
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original MRI slice")
    axes[1].imshow(original_image, cmap="gray")
    axes[1].imshow(before, cmap="magma", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title("Saved-model Grad-CAM")
    axes[2].imshow(original_image, cmap="gray")
    axes[2].imshow(after, cmap="magma", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title("After classifier-head randomisation")
    for axis in axes:
        axis.axis("off")
    fig.suptitle("Parameter-sensitivity check: Grad-CAM is classifier attention, not a lesion boundary", fontsize=11)
    fig.text(0.5, 0.02, f"Mean absolute map difference: {result['mean_absolute_difference']:.3f}; rank correlation: {result['rank_correlation'] if result['rank_correlation'] is not None else 'not defined'}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(OUT / "gradcam_parameter_sanity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
