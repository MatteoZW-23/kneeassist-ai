# Academic documentation status

## Current submission documents

Use the XAI-named submission copies below. They contain the final title, five objectives, current model-routing explanation, MRNet-only evidence, evaluation limits, Grad-CAM explanation and academic-integrity declaration.

- `submission/Mabira_Mathew_R234371Y_XAI.docx`
- `submission/Mabira_Mathew_R234371Y_XAI.pdf`
- `submission/KneeAssist_XAI_Defence_Cheat_Sheet.docx`
- `submission/KneeAssist_XAI_Defence_Cheat_Sheet.pdf`

The final academic title is:

> **KneeAssist XAI: Development of an Explainable Deep Learning System for Study-Level Classification of Normal and Abnormal Knee MRI Studies**

The report distinguishes the formal ResNet-18 versus EfficientNet-B0 baseline comparison from the live fixed target routing: DenseNet-121 for general abnormality, ResNet-18 for ACL and Swin-T for meniscus. Routing is fixed from development-validation evidence before upload; it is not patient-specific model selection or an ensemble.

## Supporting academic material

- `FOURTH_YEAR_PROPOSAL.md` — narrowed fourth-year scope and objectives.
- `VIVA_PREPARATION.md` — defensible explanations and demonstration wording.
- `PROJECT_EVIDENCE_INDEX.md` — evidence locations.
- `report_content.md` and `build_report.py` — maintained report source material.

## Historical files

Earlier documents named `KneeAssist_AI` or without `_XAI` are retained for audit history. They are not the current submission documents.

## Before submission

1. Complete only your own name, registration number, project level, campus, acknowledgements and signatures.
2. Read the declaration and disclose assistance according to institutional rules.
3. Update the Word table of contents after any final edits.
4. Submit the verified XAI-named Word or PDF copy required by your department.
5. Do not claim independent clinical validation, autonomous diagnosis, precise lesion localisation or treatment recommendation.
## Current submission build

The current submission files are the two `*_XAI.*` documents in `submission/`. The base report generator is followed by `update_xai_documents.py`, which applies the current XAI title, fixed target-routing description and research-use limits. This second step is required because the historical baseline generator is retained for provenance.
