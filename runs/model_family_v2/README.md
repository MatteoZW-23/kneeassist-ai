> **Historical benchmark note (8 October 2026):** the macro-AUROC non-promotion decision below describes the baseline release. The current dashboard uses a separate fixed target-routing registry, not an overall model promotion or weighted ensemble. See `../../models/MODEL_CARD.md`.

# Model family benchmark V2

This completed run compares ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin-T using the existing MRNet study split in `runs/mri_finetune_02/splits.json`.

It is isolated from the deployed V1 checkpoint. Checkpoints, feature caches, logs and metrics written here must not replace the active EfficientNet-B0 model until the completed locked-comparison results have been reviewed.

Selection criterion: internal tuning macro AUROC. Training accuracy and external-reference scores must not select a model. The run reports per-target AUROC, PR-AUC, sensitivity, specificity, F1, precision and accuracy through the existing evaluation pipeline.

Command used:

```powershell
.venv\Scripts\python.exe -m src.training.train --config configs\model_family_v2.yaml
```

The log is `logs/model_family_v2_training_02.log`. The final comparison and decision are in `results/model_comparison.json`, `results/REPORT.md` and `FINAL_DECISION.md`.

EfficientNet-B0 won among the five new candidates at 0.8472 internal-tuning macro AUROC, but did not beat the historical calibrated baseline (0.8478). It was not promoted and the historical V1 checkpoint remains unchanged.


