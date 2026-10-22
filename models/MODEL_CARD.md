# KneeAssist XAI model card

## Intended research use

KneeAssist XAI is a local research prototype for study-level knee MRI classification. It reports research scores for general abnormality, ACL tear and meniscal tear. It is not a diagnostic device and does not recommend treatment.

## Active model family and selection

The active registry contains ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin Transformer. Each valid upload runs every registered candidate after the same saved MRI preprocessing. A Gradient Boosting competence selector estimates mean Brier loss for each candidate using the current candidate probability vector, prediction entropy, model disagreement and available-plane descriptors. The lowest estimated loss selects one architecture for all three outputs and the Grad-CAM view.

The selection is performed after upload but before the final result is displayed. It cannot access the study's unknown MRI label. It is not a probability ensemble, it does not retrain online, and it is not a guarantee that the selected model is correct.

## Data and preprocessing

The supervised source is MRNet v1 only. Inputs are MRI slice stacks. The active pipeline uses p1–p99 normalisation, 224 × 224 resizing, twelve evenly spaced slices per available plane and study-level slice aggregation. Supported file formats are NPY/NPZ, NIfTI, DICOM slices and supported DICOM ZIP studies. JPEG/PNG inputs are rejected.

## Recorded dynamic-system evidence

Source: `runs/dynamic_selection_v1/final_evaluation.json`.

| Macro measure | Result |
|---|---:|
| AUROC | 0.847 |
| Average precision | 0.840 |
| Accuracy | 0.744 |
| Sensitivity | 0.848 |
| Specificity | 0.550 |
| Precision | 0.697 |
| F1-score | 0.764 |
| Brier score | 0.160 |

These results come from a 120-study MRNet development cohort excluded from selector fitting but previously exposed in project development. The local MRNet copy contains no patient identifiers. They are not independent patient-level, external or clinical validation. The saved comparison shows a higher macro AUROC than the fixed candidates on this cohort, but does not establish independent or clinical superiority.

## Known limits

- Only three MRNet reference-label targets are reported.
- No reader study, prospective evaluation, external calibration or multi-site validation has been completed.
- Grad-CAM is coarse positive classifier attention, not a tear location, detection box or segmentation mask.
- The model has no validated prediction for MCL/PCL injury, fracture, effusion, cartilage damage or osteoarthritis.
- A low or below-threshold score does not rule out injury.

Clinical review remains required.
