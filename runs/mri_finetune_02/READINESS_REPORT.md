# Historical EfficientNet-B0 baseline: ready for research testing

> **Superseded deployment note (8 October 2026):** this report preserves the original single-model EfficientNet-B0 evidence. The current dashboard uses the reviewed target-routed registry described in `../../models/MODEL_CARD.md` and evaluated in `../model_portfolio_v1/final/metrics.json`. The routed result is still MRNet development evidence, not clinical validation.

Verified 28 September 2026, with a 30 September follow-up. The application, selected checkpoint, real MRI upload, prediction, Grad-CAM, JSON download, case clearing and readable empty-input error were verified. The guarded DICOM ZIP importer has dedicated input-safety tests. After the guarded ensemble and localisation additions, the 30 September full suite passed 33 tests; nine notebooks were previously executed and verified. JPEG/PNG preview has been removed.

The 30 September five-model benchmark also completed. Its best new candidate, frozen-encoder EfficientNet-B0 (internal-tuning macro AUROC 0.8472), did not beat the historical calibrated baseline (0.8478). It was not promoted. See `../model_family_v2/FINAL_DECISION.md` and `../model_family_v2/FINAL_VERIFICATION.md`.

## Honest performance conclusion

The internal selection AUROC increased from 0.8478 to 0.8532, but this is NOT a demonstrated overall improvement. Official MRNet macro AUROC decreased from 0.8499 to 0.8267. External KneeMRI ACL AUROC decreased from 0.6346 to 0.6030; exploratory fastMRI meniscus AUROC decreased from 0.6740 to 0.6561. No external result was used to tune the model or reverse the prespecified selection decision.

The selected model is EfficientNet-B0 from the **frozen-encoder warm-up stage**, followed by calibration. Fine-tuning was attempted for both architectures but did not beat this checkpoint on internal selection AUROC. Do not describe the deployed checkpoint as an improved fine-tuned encoder.

## MRNet development validation: 120 studies

| Finding | AUROC | F1 | Sensitivity | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| General abnormality | 0.901 | 0.927 | 0.937 | 0.680 | 0.918 | 0.883 |
| ACL tear | 0.825 | 0.746 | 0.815 | 0.697 | 0.688 | 0.750 |
| Meniscal tear | 0.754 | 0.631 | 0.673 | 0.647 | 0.593 | 0.658 |
| Macro | 0.827 | 0.768 | 0.808 | 0.675 | 0.733 | 0.764 |

General-abnormality specificity improved from 20% to 68%, with sensitivity decreasing from 100% to 93.7%. Threshold tradeoffs do not establish clinical reliability.

## External reference evaluation

- KneeMRI ACL: 909 exams, AUROC 0.603, sensitivity 54.9%, specificity 58.6%; 102/226 injury-reference exams missed and 283/683 healthy-ACL-reference exams flagged.
- fastMRI meniscus: 199 files, AUROC 0.656, sensitivity 70.9%, specificity 55.2%. References are non-exhaustive annotations, not definitive clinical ground truth.

## Provenance and limits

Training: MRNet only; 804 fitting, 226 internal tuning, 100 reserved calibration and 120 official validation studies. The previous baseline used 904 fitting studies and remains preserved. Source datasets and private download links were not redistributed.

Patient linkage is unavailable. Official MRNet validation and KneeMRI have been evaluated previously, so they are not new untouched tests. One seed was used. Calibration uses a small development cohort; clinical and external calibration are not established. Study-bootstrap intervals in results/final/uncertainty.json omit patient clustering, selection bias and training-seed variability.

The system supports MRI NumPy slice stacks, NIfTI `.nii`/`.nii.gz` volumes, direct DICOM slices and supported study ZIPs. A guarded DICOM ZIP accepts one readable, uncompressed `.dcm` series per plane; the importer reads in memory and checks available geometry or metadata. Metadata-free NPY requires manual plane confirmation. These import routes do not verify anatomy, patient identity, mixed-patient uploads, scanner suitability, DICOM de-identification or clinical validity, and have not been clinically validated against the MRNet-trained model. Photographs, screenshots and X-rays are rejected. Grad-CAM is model attention, not a confirmed lesion. It predicts only three findings. Research testing only; no clinical-readiness claim.

## Artifacts

- models/best_model.pth: selected calibrated checkpoint
- results/model_comparison.json: internal selection evidence
- validation_summary.json: full MRNet and external metrics
- results/final: curves, confusion matrices, examples and reliability estimates
- test_results.txt and ../../results/notebook_execution.json: verification evidence
- browser_export_verified.json and browser_attention_verified.png: live workflow evidence

Checkpoint SHA256: `b60293f6c7130fd646c81b6d758584faf3e17b2689503b5278ad711156d546da`

