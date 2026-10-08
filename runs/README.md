> **Historical-run note (8 October 2026):** this index preserves baseline experiment folders. The live dashboard is governed by `../model_registry/active_models.json` and the target-routed model card.

# Released experiment evidence

This repository keeps the small, reviewable evidence files needed to understand the selected checkpoint:

- `mri_finetune_02/READINESS_REPORT.md`, `validation_summary.json`, `protocol.json`, and `test_results.txt` document the active EfficientNet-B0 research checkpoint.
- `model_family_v2/` documents the later five-architecture benchmark and the decision not to replace the recorded baseline checkpoint.

Raw predictions, plots, training logs, historical checkpoints, and data-derived outputs are intentionally excluded. They can contain large artifacts or depend on datasets that are not redistributed here.


