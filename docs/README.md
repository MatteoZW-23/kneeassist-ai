# KneeAssist XAI documentation guide

## What this project is

KneeAssist XAI is a local research prototype for study-level classification of supported knee MRI studies. Its primary academic task is normal-versus-abnormal classification. ACL and meniscal scores are secondary research outputs. It is not a diagnostic device, treatment recommender, lesion detector or segmentation system.

## Start here

- [Project README](../README.md) — installation, inputs, data provenance, training and limitations.
- [Quick-start guide](../START_HERE.md) — how to run the dashboard and present a three-plane MRI study.
- [Dashboard user guide](USER_GUIDE.md) — what each screen result means and how model routing is displayed.
- [Active model card](../models/MODEL_CARD.md) — registered models, performance, calibration and limits.

## Academic submission

- [Final XAI report (Word)](academic/submission/Mabira_Mathew_R234371Y_XAI.docx)
- [Final XAI report (PDF)](academic/submission/Mabira_Mathew_R234371Y_XAI.pdf)
- [XAI defence sheet (Word)](academic/submission/KneeAssist_XAI_Defence_Cheat_Sheet.docx)
- [XAI defence sheet (PDF)](academic/submission/KneeAssist_XAI_Defence_Cheat_Sheet.pdf)
- [Fourth-year proposal scope](academic/FOURTH_YEAR_PROPOSAL.md)
- [Viva preparation notes](academic/VIVA_PREPARATION.md)

The XAI-named copies are the current submission documents. Earlier `KneeAssist_AI` and non-XAI files are retained as historical records and should not be submitted in their place.

## Model-selection documentation

The dashboard has three active target routes, fixed before a study is uploaded:

| Score | Registered model |
|---|---|
| General abnormality | DenseNet-121 |
| ACL tear | ResNet-18 |
| Meniscal tear | Swin Transformer (Swin-T) |

The formal undergraduate baseline compared ResNet-18 and EfficientNet-B0 using validation AUROC. The live dashboard uses the registered target routes only when the uploaded MRI is compatible. It never picks the model with the highest score for the uploaded case, and no weighted ensemble is active.

Read [the model card](../models/MODEL_CARD.md) and `../model_registry/active_models.json` for the evidence and policy.

## Evidence and limitations

- Training evidence is MRNet-only.
- The routed performance figures are development evidence from MRNet official-validation material previously used in development, not independent clinical validation.
- KneeMRI ACL and fastMRI meniscus material are limited external-reference checks, not comprehensive clinical ground truth.
- Grad-CAM is coarse positive model attention. It does not confirm or precisely locate a lesion.
- JPEG/PNG screenshots, internet pictures, knee photographs and X-rays are rejected because the models were not trained or validated for them.

For the full evidence boundaries, read [clinical validation gaps](CLINICAL_VALIDATION_GAPS.md), [honest model audit](HONEST_MODEL_AUDIT.md), and [reproducibility notes](REPRODUCIBILITY.md).
