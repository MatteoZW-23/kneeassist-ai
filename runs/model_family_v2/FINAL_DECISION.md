# Model-family benchmark decision

Completed 30 September 2026 using MRNet only, one locked study-level split and one random seed. This run compared frozen-encoder ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin Transformer candidates. It did not overwrite the active checkpoint.

## Selection result

EfficientNet-B0 was the strongest new candidate on the prespecified internal-tuning macro AUROC criterion: **0.8472**. The other candidates were ResNet-18 0.8393, ResNet-50 0.8275, DenseNet-121 0.8354 and Swin-T 0.8395.

The active calibrated EfficientNet-B0 baseline scored **0.8478** on the same internal-tuning metric. The new candidate therefore did not improve on the active baseline and was **not promoted**. The active checkpoint remains `runs/mri_finetune_02/models/best_model.pth`.

The 120-exam MRNet official-validation report for the new candidate is retained in `results/final/`. Its macro AUROC was 0.8335. This cohort was evaluated earlier in development, so it is development-validation evidence only. It was not used to select the candidate.

## Interpretation and limits

The new candidate used 804 fitting, 226 tuning, 100 reserved-calibration and 120 official-validation studies. The active model's history differs, so this is a development comparison rather than a clinical performance claim. Patient linkage is unavailable, only one seed was run, and no independent clinical cohort was used.

The new benchmark checkpoint is uncalibrated because it lost the locked internal comparison. Its internal-F1 thresholds must not be used as clinical operating thresholds. The retained active model uses the existing reserved-MRNet calibration procedure, which is still not external or clinical calibration.

This project remains a research and decision-support prototype. Grad-CAM maps model attention, not a confirmed lesion, and clinician review remains required.
