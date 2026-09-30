# Ensemble, detection and segmentation gate

KneeAssist AI now contains an ensemble engine and registry gates for detection and segmentation. They are intentionally disabled in the active research deployment because no qualifying trained models are registered.

## What the software can do now

- Define a target-specific weighted ensemble only when every member, target weight and validation result is recorded in `model_registry/active_models.json`.
- Aggregate calibrated member probabilities and show score disagreement as a standard deviation.
- Reject unsafe ensemble specifications: missing members, guessed weights, missing checkpoints and unsupported planes do not become inference paths.
- Show the exact reason detection boxes and segmentation masks are unavailable instead of drawing misleading regions.

The active registry has one calibrated research classifier. Its ensemble list and localisation-model list are empty. The dashboard therefore uses the single active model and clearly identifies the fallback.

## Evidence needed to enable an ensemble

1. Train each candidate only on the locked fitting partition.
2. Calibrate each member on a reserved calibration partition that none of the members used for fitting.
3. Choose members and weights without looking at the locked evaluation set.
4. Demonstrate that the ensemble improves the prespecified target metric and does not create unacceptable sensitivity, specificity or calibration trade-offs on the locked evaluation.
5. Record checkpoints, preprocessing, weights, per-target metrics, calibration diagnostics and the decision in the registry.

The current five-model benchmark does not meet this gate for deployment: its best new candidate did not beat the active model on the locked internal comparison. No ensemble has been enabled from those checkpoints.

## Evidence needed to enable localisation

Detection requires MRI studies with verified abnormality bounding boxes. Segmentation requires MRI studies with verified pixel or voxel masks. For either task, the project must also have a fixed train/validation/evaluation protocol, a saved checkpoint, quantitative localisation metrics and qualified review of visual outputs.

Grad-CAM is retained as classifier attention. It does not substitute for a lesion box or segmentation mask.
