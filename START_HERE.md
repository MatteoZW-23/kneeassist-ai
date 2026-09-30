# KneeAssist AI: research testing

Double-click **Launch KneeAssist AI.cmd**. Upload an MRI `.npy` stack, `.nii`/`.nii.gz` NIfTI volume, `.dcm` DICOM slices, or a supported study ZIP. If a processed NPY file such as `0000.npy` has no reliable orientation metadata, confirm its plane in the intake panel. The local example is `sample_cases/mrnet_1130.zip`. JPEG/PNG uploads have been removed.

The active model's training, evaluation, nine notebooks and live browser workflow were previously verified. A later five-model benchmark is also complete: its EfficientNet-B0 candidate did not beat the active calibrated baseline, so the active checkpoint was retained. After the benchmark, dashboard and guarded-ensemble additions, the full software suite passed 33 tests; see `runs/model_family_v2/FINAL_DECISION.md`.

**Performance remains limited.** Internal selection AUROC rose slightly, but MRNet official-validation AUROC and both external AUROCs declined. The selected checkpoint is calibrated EfficientNet-B0 with a frozen encoder, not the fine-tuned version.

Read [the full readiness report](runs/mri_finetune_02/READINESS_REPORT.md) before testing. Training used MRNet only. KneeMRI was previously evaluated; fastMRI reference annotations are incomplete. Patient linkage is unavailable. DICOM and NIfTI input convenience does not establish model validity on clinical DICOM or NIfTI cohorts. This is research software, not a clinically validated diagnostic system.
