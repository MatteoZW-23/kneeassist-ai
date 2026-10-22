> **Historical-run note (8 October 2026):** this index preserves baseline experiment folders. The live dashboard is governed by `../model_registry/active_models.json` and the target-routed model card.

# Released experiment evidence

This repository keeps the small, reviewable evidence files needed to understand the selected checkpoint:

- `mri_finetune_02/READINESS_REPORT.md`, `validation_summary.json`, `protocol.json`, fixed split and retained `results/` document the formal EfficientNet-B0 research checkpoint.
- `model_family_v2/` documents the later five-architecture benchmark and the decision not to replace the recorded baseline checkpoint.

Raw predictions, plots, training logs, historical checkpoints, and data-derived outputs are intentionally excluded. They can contain large artifacts or depend on datasets that are not redistributed here.

## Plane-specific selection benchmark

`plane_specific_dms_v1/` is a completed supplementary MRNet experiment. It trains five architectures separately for each axial, coronal and sagittal plane (15 candidates), with target-specific selection after fitting. Its final evaluation records macro AUROC 0.838, F1 0.772 and Brier 0.172. EfficientNet-B0 fixed plane fusion recorded macro AUROC 0.837, F1 0.793 and Brier 0.169.

The result does not demonstrate robust improvement over the fixed baseline. The selector artifacts are therefore `pending_verification` and are not used by the live dashboard. See `plane_specific_dms_v1/final_evaluation.json` for the full saved metrics.


