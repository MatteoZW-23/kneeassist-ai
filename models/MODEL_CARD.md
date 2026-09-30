# Active model card

The active deployment is specified by `../config.yaml`. The released checkpoint is `best_model.pth`, a calibrated EfficientNet-B0 from the frozen warm-up stage. Historical checkpoints and cached pretrained encoder weights are not distributed in this repository.

Training source: MRNet only. Targets: general abnormality, ACL tear, meniscal tear. Fitting/tuning/calibration/official-validation counts: 804/226/100/120 studies. No patient-linkage verification is available.

Internal AUROC: 0.8532. Official-validation macro AUROC: 0.8267. External KneeMRI ACL AUROC: 0.6030. Exploratory fastMRI meniscus AUROC: 0.6561. These cohorts were previously evaluated or have limited reference annotations. Fine-tuning did not win selection; no overall improvement or clinical readiness is claimed.

On 30 September 2026, a separate frozen-encoder benchmark compared ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin-T. Its best new candidate (EfficientNet-B0, internal-tuning macro AUROC 0.8472) did not beat this active baseline (0.8478), so it was not promoted. See `../runs/model_family_v2/FINAL_DECISION.md`.

Supported input: MRI NumPy slice stacks, NIfTI `.nii`/`.nii.gz` volumes, direct DICOM slices or a supported study ZIP. Metadata-free NPY input requires manual plane confirmation. NIfTI and DICOM conversion are software-supported but have not been clinically validated against the MRNet-trained model. JPEG/PNG uploads are unsupported. Grad-CAM is attention, not lesion ground truth. Read `../runs/mri_finetune_02/READINESS_REPORT.md` for measured results and limitations.
