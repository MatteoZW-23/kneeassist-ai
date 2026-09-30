# Dataset acquisition review — 27 September 2026

## Actual outcome
Downloaded five public SKM-TEA reference files: train.json, val.json, test.json, DATASET.md and LICENSE. The JSON files describe 155 scans and 476 detection annotations across 16 categories. These are annotations and metadata, NOT MRI images. They were later moved outside the active project with other unused material; no training configuration or checkpoint changed.

## What is already available
MRNet is the current training source. KneeMRI is already imported and previously evaluated, so it is not a fresh untouched test cohort. fastMRI import_status.json reports 199 completed and reviewed volumes with archive integrity verified. fastMRI+ annotation CSVs are already present. Missing annotation rows are not confirmed negative diagnoses. Do not download duplicate copies or move previously evaluated cohorts into training without defining a new evaluation plan.

## Sources checked and decisions
| Resource | Decision and remaining requirement |
|---|---|
| [SKM-TEA](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md) | Reference annotations downloaded. Full images require portal login; documentation says 900 GB compressed and 1.6 TB expanded. About 73.7 GB free on C: during this review. Stanford source and qDESS sequences; not an automatic independent hospital test or drop-in MRNet replacement. Keep original splits and subject grouping. Repository MIT license does not replace image access terms. |
| [MeniOmni](https://github.com/ShuruiXu/MeniOmni) | Repository says code/data will be released after conference presentation. No downloadable image release verified; do not claim acquired. |
| [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) | Potential multi-site resource, but existing processed RSNA archives must be reconciled before acquisition. No local kaggle.json credential file found. Authenticated file inventory, terms, original study identifiers and label coverage must be verified before selecting downloads. Reports are not automatically expert-confirmed binary labels. |
| [Osteoarthritis Initiative](https://nda.nih.gov/oai/) | Candidate longitudinal OA cohort. Portal access and matching clinical MRI assessment tables need verification. No images downloaded. OA-focused population does not automatically cover acute ACL/meniscus injury requirements. |
| [ACL-PCL Mendeley](https://data.mendeley.com/datasets/fjns2xzhrr/1) | Not selected: author describes augmented/resized images, not verified original full studies with patient grouping. Not adequate evidence of an independent study-level test cohort. |
| [fastMRI+](https://github.com/microsoft/fastmri-plus) | Already acquired annotations and 199 matching imported images. Additional authorized image access could expand coverage, but reference-label completeness and mapping still require review. |

## Highest priority
Acquire original full MRI studies from another institution with expert-confirmed ACL, meniscus and general-abnormality labels, patient grouping and permission for the intended research. Define a development cohort and preserve an untouched external test cohort before training. More downloaded slices alone do not solve external performance or label quality.

For practical next access steps, use your Kaggle account to open the RSNA competition and review its access conditions; only after a file inventory and duplicate check should original studies be selected. For SKM-TEA, use the linked Stanford portal and arrange substantially more storage first. Never paste passwords or private download tokens into reports.

Dashboard DICOM/NIfTI import, quality checks and PDF export are software work, not additional datasets. This acquisition does not implement those features or improve measured model performance. The project remains a research prototype.
