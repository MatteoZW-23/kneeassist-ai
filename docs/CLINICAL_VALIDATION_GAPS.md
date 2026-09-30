# KneeAssist AI: remaining validation work

## Status

The software and research checkpoint are complete for local research testing. This is **not** evidence that the model is clinically ready. The active model was trained on MRNet only. Its official MRNet validation macro AUROC is 0.827, while previously evaluated external reference results are 0.603 for KneeMRI ACL and 0.656 for fastMRI meniscus. Those results are not sufficient to claim reliable use on a new hospital, scanner, patient population, or DICOM workflow.

The RSNA online pilot is deliberately isolated from the deployed model. It used only 58 explicitly labelled studies and had exploratory grouped-fold macro AUROC near 0.658. Unverified reports were not turned into diagnostic labels.

## What has been completed

- Reproducible MRNet training, held-out development evaluation, checkpoints and test logs.
- Research dashboard with MRI NumPy stacks, a guarded plane-labelled DICOM ZIP route, Grad-CAM, exports and readable input errors.
- Rejection of JPEG/PNG screenshots, photographs and X-rays, which the model was not trained to interpret.
- Clear three-finding intended research output: general abnormality, ACL tear and meniscal tear.
- Internal and limited external reference evaluation documented without using external results to tune the selected checkpoint.

## What must happen before any clinical use claim

1. **Define an intended use and failure policy.** Specify the clinical users, care setting, MRI sequences/scanners, supported findings, the action a user may take from a result, and how the system handles unsuitable or incomplete studies.
2. **Build an independent clinical dataset.** Obtain governance approval and a multi-site, patient-level separated cohort that represents the intended setting. Preserve acquisition metadata, exclude overlap with development data, and have qualified readers establish a reference standard.
3. **Validate prospectively or with a locked retrospective protocol.** Lock preprocessing, thresholds and the checkpoint before testing. Report uncertainty intervals, subgroup results, calibration, false negatives, false positives and abstentions on an untouched cohort.
4. **Run reader and workflow studies.** Measure how clinicians use the output and whether it changes accuracy, time, safety or follow-up. The system must remain decision support; Grad-CAM must not be interpreted as lesion confirmation.
5. **Establish lifecycle controls.** Add privacy/data-governance review, cybersecurity assessment, auditability, version control, monitoring, incident handling, revalidation triggers and a controlled retraining plan.

## Evidence needed to close each gap

| Gap | Minimum evidence | Current state |
|---|---|---|
| Reference labels | Blinded, qualified-reader labels with adjudication rules | Not available for an independent deployment cohort |
| Generalization | Locked evaluation across independent sites/scanners | Limited, previously evaluated external reference sets only |
| Calibration | Calibration and decision analysis in target setting | Small internal development calibration only |
| Safety | Error analysis, subgroup analysis and defined stop/abstain behavior | Input guardrails only; no clinical safety study |
| Human factors | Reader/workflow study with intended users | Not performed |
| Governance | Local ethics, privacy, security and regulatory review appropriate to deployment | Not performed |

## References

1. U.S. Food and Drug Administration. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*. 2025. https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
2. U.S. Food and Drug Administration, Health Canada, and Medicines and Healthcare products Regulatory Agency. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*. 2021. https://www.fda.gov/media/153486/download
3. EQUATOR Network. *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods*. https://www.equator-network.org/reporting-guidelines/tripod-statement/

These sources guide development and reporting. They do not certify KneeAssist AI or replace local legal, ethics, clinical, or regulatory advice.
