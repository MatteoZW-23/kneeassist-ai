# KneeAssist AI Fourth Year Project Proposal

## Proposed title

**KneeAssist AI Development of a Deep Learning System for Study Level Classification of Normal and Abnormal Knee MRI Studies**

This title is narrow enough for a fourth-year data-science project. It states the method, the imaging modality, the unit of analysis and the binary classification task. KneeAssist AI is the system name; the academic contribution is the deep-learning study-level classification system.

## Problem statement

Knee MRI examinations contain many related slices across different planes. A model that classifies a single screenshot may miss information available elsewhere in the examination. There is a need for a reproducible research prototype that combines selected slices from a complete knee MRI study and classifies the study as normal or abnormal. The proposed system is for research and learning; it is not a diagnostic replacement for a radiologist.

## Main objective

To develop a deep-learning system for study-level classification of normal and abnormal knee MRI studies.

## Specific objectives

1. To prepare and organise publicly available knee MRI studies for normal and abnormal classification.
2. To preprocess knee MRI studies using resizing, within-study normalisation and training-only augmentation.
3. To develop and compare deep-learning models for classifying knee MRI studies as normal or abnormal.
4. To evaluate the selected model using AUROC, accuracy, precision, recall, F1-score, specificity and a confusion matrix.
5. To develop a simple KneeAssist AI prototype for uploading a supported MRI study and displaying the classification result with a confidence score.

## Scope

| Item | Fourth-year scope |
|---|---|
| Input | One knee MRI study with available axial, coronal and sagittal stacks |
| Main output | Normal or abnormal study classification |
| Dataset | MRNet public knee MRI studies |
| Models | Transfer-learning ResNet-18 and EfficientNet-B0 |
| Study representation | Sampled slices combined at study level |
| Evaluation | AUROC, accuracy, precision, recall, F1-score, specificity and confusion matrix |
| Prototype | Local Streamlit research interface |
| Boundary | Research and decision-support prototype; not a hospital diagnostic system |

## Corrected data description

The project uses **1,250 MRNet studies**, not 1,370 patients. The available project records support study-level grouping but do not provide reliable patient linkage. The current split is 804 fitting studies, 226 internal tuning studies, 100 calibration studies and 120 official validation studies. It is therefore incorrect to describe the dataset as a simple 70/15/15 patient split.

The model receives three-dimensional MRI stacks shaped as slices by height by width. It does not train from folders of unrelated JPEG or PNG screenshots. A complete study is the prediction unit. The local interface also accepts a guarded DICOM ZIP structure, but DICOM support does not establish clinical validation.

## System flow

Knee MRI study → file and structure checks → within-study normalisation and resizing → pretrained image encoder → slice and plane aggregation → normal/abnormal score → research interface result and Grad-CAM attention view.

## Optional extension

The existing system can additionally display ACL tear and meniscal tear research scores because MRNet has those labels. They should be described as secondary outputs. They are not required for the narrowed fourth-year title and should not be presented as coverage of all knee abnormalities.

## Expected deliverables

- Reproducible data-preparation code and study-level split record.
- ResNet-18 and EfficientNet-B0 comparison.
- Saved best research checkpoint.
- Evaluation results, curves and confusion matrix.
- Local Streamlit prototype with an MRI viewer, classification score, export and Grad-CAM attention map.
- Academic report, notebooks, evidence index and viva preparation guide.

## Limitation statement for the proposal

The project will demonstrate a deep-learning classification workflow on public data. It will not claim to diagnose patients, replace radiologists, prove effectiveness in Zimbabwean hospitals or provide clinical approval. Independent clinician-reviewed, multi-site data would be required before those claims could be considered.
