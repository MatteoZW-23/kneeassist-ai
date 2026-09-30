# Fresh training plan

Status: waiting for the pending fastMRI images. Training and dashboard predictions are disabled in config.yaml. No new training has started. Requested deletion was blocked by automatic approval review; existing checkpoint files, feature caches, results, logs, release archives and notebook outputs remain on disk. They must not be mistaken for a new run.

## Organisation
- src/: reusable data, model, training, evaluation and inference modules.
- notebooks/: six workflow notebooks; old outputs are pending removal.
- data/MRNet-v1.0/: original MRNet studies; preserve.
- data/external/: external metadata, annotations and access records; preserve. Never publish private access links.
- data/external/KneeMRI/original_download/: downloaded KneeMRI volumes; preserve.
- models/: trained checkpoints pending deletion; pretrained/ contains public ImageNet initialisation weights, not locally fitted classifiers.
- data/features/: derived encoder features pending deletion.
- results/, logs/, release/: previous experiment outputs pending deletion.
- sample_cases/: demonstration inputs; preserve source attribution and licence restrictions.

## Before a fresh run
1. Finish downloading fastMRI validation images and verify archive integrity.
2. Match actual image filenames to fastMRI+ annotation records and inspect orientation, sequence and label semantics. Missing annotation does not establish a negative diagnosis.
3. Audit repeated studies and image duplicates; use patient-level grouping where verified patient linkage exists. Record where patient independence cannot be established.
4. Freeze a target-specific training, tuning and evaluation plan. MRNet provides three targets; KneeMRI only ACL. Do not invent missing labels or train all targets with absent labels set to zero.
5. Remove old fitted checkpoints, feature caches and saved experiment outputs after the deletion block is resolved. Rebuild manifests/splits under a versioned run plan rather than silently mixing cohorts.
6. Start both architecture baselines from ImageNet initialisation, not the old fitted checkpoints. Compare by internal validation AUROC; choose thresholds/calibration using tuning data only.
7. Evaluate the locked model on the predefined evaluation cohorts. Report sensitivity, specificity, precision, F1, accuracy, AUROC, confusion matrices and confidence intervals. Test inference, Grad-CAM and the interface before releasing a new package.

## Evaluation history must remain disclosed
MRNet official validation and the full KneeMRI cohort have already been evaluated. Deleting weights or outputs does not make these cohorts untouched again. Prior KneeMRI evaluation used 917 volumes / 909 exam IDs with sagittal-only inputs and showed limited generalisation. Future results on the same cohort must be described with this development history. Keep genuinely new fastMRI cases reserved if using them for external evaluation; do not also train on them.

The existing trainer currently fits MRNet only. Multi-dataset training and safe partial-label handling must be implemented and tested before enabling such a run. File organisation alone does not add those capabilities.
