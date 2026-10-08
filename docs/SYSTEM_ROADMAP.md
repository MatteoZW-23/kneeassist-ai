# KneeAssist XAI System Roadmap

## Product direction

KneeAssist XAI is intended to become a knee MRI clinical decision-support platform. Its future workflow is:

`MRI study -> input understanding -> validated model selection -> finding scores -> attention or localisation view -> clinician review`

The clinician remains responsible for the final interpretation. A score, attention map, box or mask is never treated as a diagnosis by itself.

## Deployed research version

| Layer | Available now | Evidence and boundary |
|---|---|---|
| Supported study inputs | MRI NumPy arrays, NIfTI `.nii`/`.nii.gz`, direct DICOM slices, and supported study ZIPs | NPY inputs without metadata require manual plane confirmation. DICOM geometry and NIfTI affine evidence are inspected in memory. Multi-series DICOM ambiguity is rejected. |
| Study understanding | File size, numeric volume, shape, dimensions, datatype, intensity range, finite values, DICOM geometry and cautious NIfTI orientation checks | Checks do not prove anatomy, acquisition quality, patient identity or that one upload contains one patient. |
| Study-level prediction | General abnormality, ACL tear and meniscal tear scores | The active registry routes MRNet-only DenseNet-121, ResNet-18 and Swin-T checkpoints by target. The fourth-year primary task is normal versus abnormal study classification. |
| Model comparison and routing | ResNet-18 and EfficientNet-B0 were compared for the formal objective. A versioned registry then applies fixed target-specific development routes. | Active routes are DenseNet-121 for general abnormality, ResNet-18 for ACL and Swin-T for meniscus. There is no weighted ensemble. |
| Explainability | Slice-specific Grad-CAM attention with a multi-plane view | Attention shows factors that influenced the score; it is not lesion localisation, a box, or a segmentation mask. |
| Review interface | Local Streamlit upload, viewer, scores, attention, export and case clearing | It is a research interface and has no hospital integration, access controls, audit trail or prospective clinical validation. |

The measured results and limitations are in `models/MODEL_CARD.md` and `runs/mri_finetune_02/READINESS_REPORT.md`.

## Development stages

### Stage 1 — fourth-year research scope: complete

Keep the academic claim focused on normal versus abnormal classification of a complete knee MRI study. Retain ACL and meniscal scores as secondary outputs. The required evidence is reproducible splitting, ResNet-18 versus EfficientNet-B0 comparison, calibration, study-level metrics, Grad-CAM, and a working local interface.

### Stage 2 — validated input adapters

Validate the new NIfTI and direct-DICOM adapters with representative knee-MRI studies and a fixed conversion test set. Add a DICOM-series selector only after testing multi-series studies from each intended scanner environment. Each adapter must preserve orientation, identify the selected sequence, record excluded series, and reject ambiguous studies with a readable explanation.

Acceptance evidence:

- Unit tests covering accepted, ambiguous and corrupt NIfTI/DICOM studies.
- Orientation and slice-order checks against known reference studies.
- No change in model output from an equivalent validated NPY reference after conversion.
- An explicit unsupported-input response rather than a guessed result.

### Stage 3 — per-finding model selection: research implementation complete

The existing versioned model registry now records the three fixed research routes. Every candidate must state its target finding, required planes/sequences, training source, preprocessing version, checkpoint hash, calibration set, locked evaluation cohort, metrics and intended-use limit. The router may select a candidate only when the uploaded study satisfies those requirements.

Candidates can include ResNet, DenseNet, EfficientNet and a transformer, but a name alone is not enough. A model becomes selectable only when it has been trained and evaluated under the same pre-specified protocol. The registry must choose by target-specific locked validation evidence, not by an unverified score in the user interface.

Acceptance evidence:

- Patient or study disjoint training, tuning, calibration and final-test cohorts.
- AUROC, average precision (AP), sensitivity, specificity, precision, F1, calibration and confidence intervals for each finding.
- Subgroup and missing-plane analyses.
- A held-out external evaluation never used for routing, thresholds or model selection.

### Stage 4 — localisation and segmentation

Grad-CAM remains an explanation method. To display a suspicious-region box, train and validate a detector using clinician-reviewed bounding boxes. To display a region mask, train and validate a segmentation model using clinician-reviewed masks. Classification labels alone cannot justify either output.

Required data:

| Capability | Required reference standard |
|---|---|
| ACL or meniscal detection | MRI studies with lesion bounding boxes reviewed by qualified readers |
| Cartilage, bone, effusion or ligament segmentation | MRI studies with region masks and a documented annotation protocol |
| New pathology score | Sufficient study-level labels with the label definition, prevalence and reader agreement recorded |

### Stage 5 — clinical translation

Before clinical deployment, perform prospective multi-site evaluation, workflow testing with radiologists, input-quality and out-of-distribution evaluation, security and privacy review, access control and audit logging, local regulatory assessment, and a risk-management process. These are clinical and governance requirements, not software features that can be inferred from public benchmark results.

## Current model-routing rule

The current defensible research-routing rule is:

`supported MRI study -> prevalidated target-specific routing (DenseNet-121 general abnormality, ResNet-18 ACL, Swin-T meniscus) -> three research scores`

The application must not claim that DenseNet, a transformer, an ensemble, a detection model or a segmentation model was selected unless its versioned, validated checkpoint and evidence are present.

## Data needed before expanding findings

- Independent knee MRI studies from different scanners and sites.
- Clinician-reviewed study labels for each new target pathology.
- Patient or reliable study identifiers for leakage checks.
- Sequence metadata and quality labels.
- Bounding boxes or masks for any localisation claim.
- A separate final test set retained before model development.

## Next practical implementation

The next safe software increment is a read-only input-inspection report and model-registry schema. It should display what the system knows about a study and why a specific validated checkpoint is eligible. It should not add a new clinical finding until the matching labelled data and locked evaluation evidence exist.


