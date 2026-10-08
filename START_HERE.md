# KneeAssist XAI: research testing

## Start the dashboard

Double-click **Launch KneeAssist AI.cmd** to launch KneeAssist XAI. The dashboard accepts one knee MRI study at a time. It supports MRI NumPy stacks, NIfTI volumes, DICOM slices and supported study ZIPs. JPEG/PNG pictures are intentionally rejected.

For a clear demonstration, select **Load research example** or **Load abnormal research example**. Each example includes axial, coronal and sagittal sequences from one MRNet validation study.

## What happens after upload

1. The system checks the file type and asks for a plane when a processed array has no orientation metadata.
2. It checks which prevalidated research models are compatible with the uploaded planes.
3. It runs the fixed registered route: DenseNet-121 for general abnormality, ResNet-18 for ACL and Swin Transformer for meniscus.
4. It displays the three study-level scores, the selected models, a slice viewer, coarse Grad-CAM attention and an exportable summary.

The results screen first lists **Models used for this study**, then repeats **Model used** under each score card. Open **Validated model selection and study compatibility** to view the architecture and saved validation evidence for each route.

The system does **not** choose whichever model gives the largest score for the uploaded case. Its selection was made from fixed MRNet development evidence before upload.

**Portability note:** this local working project contains all three registered route checkpoints. A fresh Git checkout contains the historical EfficientNet-B0 baseline checkpoint but not the local routing weights; it must reproduce or otherwise obtain the approved local route checkpoints before the routed dashboard can analyse a study.

## Evidence and limits

The routed development evaluation is in `runs/model_portfolio_v1/final/metrics.json`: macro AUROC 0.846 on 120 MRNet official validation studies that were previously used during development. Training used MRNet only. The prototype is ready for academic demonstration and research testing, but it is not clinically validated, does not diagnose disease, locate a tear precisely or recommend treatment. Clinical review is required.

Read `models/MODEL_CARD.md` and `docs/CLINICAL_VALIDATION_GAPS.md` before claiming results beyond the academic prototype.


