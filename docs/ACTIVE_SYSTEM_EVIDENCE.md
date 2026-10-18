# Active system evidence and limitations

## What is active

The active application configuration is `dynamic_selection_v1`. The registry lists five validated local checkpoints: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin Transformer. Each complete three-plane MRI upload is processed by all five. A saved gradient-boosting competence selector estimates a mean Brier loss for each candidate using current candidate probabilities, entropy, disagreement and study descriptors. It selects the candidate with the lowest estimate for all three reported outputs and the Grad-CAM view.

The selector is an offline-trained component. It does not train during upload and cannot use the uploaded study's unknown reference label. The active registry, selector feature schema and checkpoint hashes are checked before inference. The system requires axial, coronal and sagittal sequences from the same study because the selector was evaluated on complete three-plane studies; it rejects a missing-plane upload rather than treating the result as validated.

## Recorded final evaluation

Source: `runs/dynamic_selection_v1/final_evaluation.json`.

| Measure | Dynamic system macro result |
|---|---:|
| AUROC | 0.835 |
| Average precision | 0.833 |
| Accuracy | 0.764 |
| Sensitivity | 0.782 |
| Specificity | 0.681 |
| Precision | 0.736 |
| F1-score | 0.757 |
| Brier score | 0.162 |
| ECE (10 bins) | 0.096 |

The evaluation contains 120 MRNet studies excluded from dynamic-selector fitting. It had prior project exposure and local MRNet records do not include patient IDs. Therefore, it is not an independent patient-level, external or clinical test. In this saved comparison the dynamic system did not have the highest macro AUROC of every fixed candidate, and small Brier-score differences do not establish superiority.

## What the evidence supports

- A working local MRI-study classification pipeline for three supplied MRNet labels.
- Fairly recorded execution of five candidate architectures on the same prepared upload.
- A traceable case-specific selection mechanism that varies the selected architecture by study.
- Study-level research score display, model identification, candidate comparison and selected-model Grad-CAM.

## What it does not support

- A diagnosis, lesion location, treatment decision or claim of benefit for junior doctors.
- Performance for conditions without reliable training labels, including MCL/PCL injury, fracture, effusion, cartilage damage and osteoarthritis.
- Independent patient-level, external, multi-site or clinical validation.
- A claim that Grad-CAM is a tear boundary, detection box or segmentation mask.
