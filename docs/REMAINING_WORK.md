# Release status and remaining research limitations

All current software completion gates passed: training, calibration, evaluation, 29 tests, nine notebooks and live upload/prediction/Grad-CAM/export/clear verification. Picture uploads were removed. The input router also covers unnamed NPY confirmation, NIfTI and guarded DICOM routes. See ../runs/mri_finetune_02/READINESS_REPORT.md.

The selected checkpoint is calibrated EfficientNet-B0 from the frozen warm-up stage. Internal selection AUROC improved slightly; official-validation and external AUROCs declined. It is not an overall performance improvement.

Remaining research limitations: unavailable patient linkage; reused validation cohorts; one seed; incomplete fastMRI reference annotations; no anatomy/out-of-distribution or mixed-patient detection; no clinical validation of DICOM/NIfTI conversion; no prospective clinical validation. These cannot be resolved by cosmetic touch-ups. The release is for research testing, not clinical diagnosis.
