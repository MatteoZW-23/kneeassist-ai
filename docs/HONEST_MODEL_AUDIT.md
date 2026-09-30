# Honest model audit - 26 September 2026

## Verdict
The application is a working research baseline, not a clinically reliable knee diagnostic system. Calling the model complete without that qualification overstates its readiness. The current code passed 21 automated tests during this audit; these test software behaviour, not clinical safety.

## Verified performance gaps
- MRNet general abnormality: specificity 20%; 20 of 25 reference-normal exams flagged.
- External KneeMRI ACL: AUROC 0.635, sensitivity 33.2%; 151 of 226 injury-reference exams missed.
- External fastMRI meniscus: AUROC 0.674, specificity 33.3%; 64 of 96 annotation-reference-negative files flagged. Non-exhaustive annotations make these limited reference results.

## Missing capabilities and evidence
1. MRI encoder fine-tuning: src/models/study_model.py freezes the ImageNet encoder; the optimizer in src/training/train.py trains only model.head. Twelve slices per plane and mean/max pooling form a baseline; full-volume spatial reasoning is not implemented. Improvement is not guaranteed by changing architecture.
2. Combined-dataset training: only MRNet supplies training labels. No masked partial-label/multi-dataset training exists. KneeMRI has ACL-only labels; fastMRI sprain labels cannot silently become MRNet ACL tears.
3. Patient-independent validation: patient_mapping is null. Study and exact-series checks exist, but these do not establish patient independence or comprehensive cross-source duplication checks. MRNet official validation and KneeMRI were evaluated previously.
4. Reliable operating thresholds/calibration: F1-selected thresholds have poor specificity in important settings. There is no fitted probability calibration or clinically validated confidence measure. No cross-validation or multiple-seed stability assessment is implemented for the selected run.
5. Input suitability: the current intake inspector reads shape/format facts, DICOM geometry and cautious NIfTI orientation evidence, and requires manual confirmation for metadata-free NPY input. It still cannot recognise non-knee images, mixed-patient uploads or out-of-distribution studies. JPEG/PNG receives no prediction. DICOM/NIfTI import has not been validated on clinical cohorts.
6. Expanded diagnoses: the model predicts general abnormality, ACL and meniscus only. It does not identify all possible causes behind an abnormal score, grade tears or provide validated lesion segmentation. Grad-CAM is not lesion ground truth.
7. Documentation/provenance: README contains historical paths and states; config.external_fastmri.label_mapping_verified remains false although an exploratory meniscus mapping was evaluated. A reviewed mapping specification and one coherent active-run document are needed. Old artifacts remain inactive because cleanup was blocked.
8. Clinical-use evidence: no prospective clinician workflow study, deployment qualification or intended-hospital validation is available. Software tests and AUROC alone do not establish suitability for patient-care decisions.

## Recommended work order
Freeze a new versioned evaluation protocol and clarify label/reference limitations first. Audit grouping and input suitability, implement calibration/operating-point evaluation on fitting/tuning data only, then compare staged encoder fine-tuning and more appropriate slice aggregation against the baseline. If incorporating external sources for training, define separate held-out groups and disclose prior evaluations; do not tune on all previously evaluated external data and call it independent testing. Reconcile docs and run paths. Keep photograph diagnosis disabled until a separate suitable training/evaluation task exists.

Evidence: config.yaml; src/models/study_model.py; src/training/train.py; src/inference/predictor.py; runs/mrnet_fresh_01/results/{final,external_kneemri,external_fastmri}/metrics.json; tests/.
