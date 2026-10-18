# Dynamic model-selection protocol

## Purpose

The active KneeAssist XAI workflow selects one of five trained study classifiers for each complete uploaded MRI study. It is an offline-trained model-selection component. It does not diagnose an image, access the unknown reference label during upload, or train again when a study is submitted.

## Fixed experiment inputs

The recorded `dynamic_selection_v1` experiment used MRNet v1.0 with five ImageNet-pretrained candidates: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin Transformer. Every candidate used the same saved study roles, three label targets, preprocessing and study-level aggregation.

| Role | Studies | Use |
|---|---:|---|
| Fitting | 704 | Base-classifier fitting |
| Tuning | 100 | Base-classifier early stopping |
| Calibration | 100 | Probability mapping and research thresholds |
| Selector training | 226 | Competence-estimator fitting |
| Recorded final evaluation | 120 | Evaluation only after selector fitting |

The split has zero listed study and exact-series overlaps. The local MRNet copy has no patient identifiers, and the 120-study role had previous project exposure. It is therefore recorded development evidence, not an independent patient-level, external or clinical test.

## Upload-time selection

1. The intake pipeline validates MRI file content and loads axial, coronal and sagittal sequences.
2. All five registered checkpoints process the same prepared study.
3. The selector constructs a 90-feature vector from candidate probabilities, entropy, inter-model disagreement and coarse study descriptors.
4. A fitted Gradient Boosting Regressor estimates the mean Brier loss for each candidate.
5. The candidate with the lowest estimated loss supplies all three reported scores and the selected-model Grad-CAM view.

The system does not select the model with the highest abnormality probability. It does not average model outputs as an ensemble. Its estimated loss is not a probability that the selected model is correct.

## Compatibility and integrity checks

The active selector runs only when all three planes are available. A missing-plane study is rejected rather than being passed through a selector trained and evaluated on complete studies. Before inference, the application verifies the selector-artifact hash, all five checkpoint hashes, target order, plane order, estimator family and the exact 90-feature schema.

The dashboard shows all five candidate score vectors, estimated losses, the selected architecture, processing time and the feature importances of the selected competence estimator. These importances explain the model-selection estimate only; they do not identify anatomy, a tear or a lesion boundary.

## Recorded evaluation and limitations

The source file is [`runs/dynamic_selection_v1/final_evaluation.json`](../runs/dynamic_selection_v1/final_evaluation.json). Dynamic selection varied across all five candidates on the recorded 120 studies. Its recorded macro AUROC was 0.835 and it did not exceed every fixed candidate's macro AUROC. The saved evidence therefore demonstrates an implemented case-specific selection mechanism, not superior clinical performance.

The active artifact declares Gradient Boosting with the extended 90-feature schema. An older archived protocol text still names RidgeCV. The dashboard displays this as a provenance warning; runtime follows the fitted artifact metadata and the historical evidence file is not rewritten. A clean replication with an independent patient-linked cohort is needed before making a generalisation or clinical-use claim.

Grad-CAM remains coarse classifier attention. No validated detector, bounding-box localiser or segmentation model is part of the active system.
