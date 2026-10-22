# Data is intentionally not included

The source repository does not distribute MRI images, source archives, external evaluation data, or private download links. These files are large and subject to their original dataset agreements. The local MRNet copy needed to reproduce the recorded training evidence remains in this final project folder. The previously downloaded external evaluation copies were moved recoverably to the Desktop research archive listed in `../CLEANUP_MANIFEST.md`; they are not needed to run the final dashboard.

To reproduce the training workflow, obtain MRNet and any optional external datasets directly from their official sources, review their terms, preserve their original study split rules, and place them under the paths described in `config.yaml` or a copied local configuration. Do not commit downloaded studies, patient information, or access tokens.

The released checkpoints were trained using MRNet only. See `../models/MODEL_CARD.md`, `../docs/FINAL_READINESS_REPORT.md` and `../CLEANUP_MANIFEST.md` for provenance, verification and archive locations.
# Data areas

## Active research data

The active training and development data for KneeAssist XAI are documented in
`DATA_USAGE.md`. The currently registered dashboard models are MRNet-trained;
do not describe other local archives as active training data without checking
their provenance, labels, licence and patient overlap.

## Future independent clinical validation data

`clinical_validation_template/` is a **template-only** collection pack. It
contains no patient MRI studies and no real labels. It is designed to help an
approved institution prepare an independent dataset for a later locked
evaluation.

Do not store real patient MRI studies, original identifiers, DICOM headers,
or patient-to-pseudonym mapping files in this Git repository. Keep them in an
approved private location such as a hospital-controlled storage area. The
template validator can check manifests after approved de-identification; it
does not de-identify DICOM files and does not replace institutional review.

See [clinical_validation_template/README.md](clinical_validation_template/README.md).
