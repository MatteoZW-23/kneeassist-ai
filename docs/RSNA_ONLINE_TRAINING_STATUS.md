# RSNA online training status

27 September 2026: Kaggle competition access verified and private notebook created:
https://www.kaggle.com/code/mathewmabira/kneeassist-ai-rsna-online-training

Online audit: 4,407 unique study IDs; only 58 studies have explicit nonmissing labels for each of the 12 conditions. Two T4 GPUs detected. The remaining reports are not treated as binary ground truth.

Version 1 (Verified-label pilot - float32 fix) was submitted using Save & Run All. Interactive session was intentionally stopped to avoid duplicate GPU usage. Completion verified: saved Version 1, run 353402281, succeeded in 269 seconds. Output tab contains 24 files including both encoder weights, ten fold heads, metrics, histories, OOF predictions, audit, configuration, folds and completion status.

Pipeline: original mounted DICOM series, one fluid-sensitive preferred series per plane, up to 12 slices per plane, per-series percentile normalization, float32 inputs, frozen ImageNet ResNet-18 and EfficientNet-B0 encoders with mean/max slice pooling and trained weighted-BCE linear heads. Five grouped folds; learning-rate scheduler and early stopping. Saves encoder weights, fold heads, normalization parameters, folds, metrics and predictions to /kaggle/working/kneeassist_rsna_pilot.

Patient IDs and exact sampled pixel fingerprints are grouped, but this does not prove clinical patient independence or detect all near duplicates. Validation folds also control early stopping, making reported OOF results exploratory and potentially optimistic. No independent test results or clinical-readiness claims. No automatic replacement of the local MRNet model. No MRI dataset downloaded to the laptop.

Known fixed error: numpy percentile normalization promoted input to float64; explicit float32 normalization and model input conversion added before committing Version 1.

## Measured exploratory results
ResNet-18 macro AUROC: 0.6582568421428666.
EfficientNet-B0 macro AUROC: 0.6582742309539501.
These are effectively tied; no meaningful superiority demonstrated. Early stopping reused validation folds, and there are only 58 labelled studies. Results are not an independent test and do not justify replacing the MRNet model. All large MRI data remained on Kaggle. Outputs remain in the private saved notebook.

## V2 review gate

The future V2 path is prepared but intentionally not trained. `docs/RSNA_LABEL_REVIEW_PROTOCOL.md` describes how to create a de-identified review manifest in Kaggle, collect two independent clinical reviews, queue disagreements for adjudication and produce training-eligible labels. `configs/kaggle_v2_review.yaml` prevents report text from being treated as ground truth. No dashboard model changes occur until reviewed labels and a locked independent evaluation plan exist.

