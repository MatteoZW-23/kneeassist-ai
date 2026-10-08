# KneeAssist XAI active research model card

## Purpose

KneeAssist XAI is a local explainable-AI research prototype for **study-level classification** of knee MRI examinations. It produces research scores for general abnormality, ACL tear and meniscal tear. It is not a diagnostic device and does not recommend treatment.

## Active target routing

The dashboard selects compatible models from a fixed local registry before analysing the upload. It does **not** select the model that gives the highest score for an individual case.

| Finding | Registered model | Selection evidence |
|---|---|---|
| General abnormality | DenseNet-121 | Highest locked tuning score for this finding |
| ACL tear | ResNet-18 | Highest locked tuning score for this finding |
| Meniscal tear | Swin Transformer | Highest locked tuning score for this finding |

The selection score weights AUROC (0.45), sensitivity (0.30), average precision (AP; a precision-recall-curve summary) (0.15) and F1-score (0.10). Each candidate was calibrated with a reserved MRNet development subset. See `../runs/model_portfolio_v1/portfolio_report.json` and `../runs/model_portfolio_v1/final/metrics.json`.

## Development evaluation

The target-routed system was evaluated on 120 MRNet official validation studies that had previously been used during development. It is therefore development evidence, not an independent clinical test.

| Finding | AUROC | F1 | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|
| General abnormality | 0.861 | 0.929 | 0.968 | 0.560 | 0.883 |
| ACL tear | 0.870 | 0.763 | 0.926 | 0.591 | 0.742 |
| Meniscal tear | 0.808 | 0.677 | 0.846 | 0.500 | 0.650 |
| Macro average | 0.846 | 0.790 | 0.914 | 0.550 | 0.758 |

## Training data and inputs

All active classifiers were trained and compared using MRNet v1.0 only. The local working project contains the three registered route checkpoints. The Git repository retains only the historical EfficientNet-B0 baseline checkpoint; it does not redistribute the additional route weights. The registry supports one to three compatible axial, coronal and sagittal MRI sequences from the same study. NumPy, NIfTI, DICOM and supported study ZIP inputs are software-supported. NIfTI/DICOM convenience support has not been validated on a clinical cohort. JPEG/PNG pictures are rejected.

## Important limits

- MRNet patient linkage is unavailable in this local copy; study separation does not prove patient independence.
- No external clinical validation, prospective study, radiologist reader study or treatment evaluation has been completed.
- Grad-CAM is coarse positive classifier attention, not a tear location, bounding box or segmentation mask.
- A below-threshold score does not rule out an injury.

The system is suitable for academic demonstration and research testing only. Clinical review remains required.


