# Release status and remaining research limitations

The current local research release uses fixed target-specific routing: DenseNet-121 for general abnormality, ResNet-18 for ACL and Swin-T for meniscus. The routing policy uses fixed MRNet tuning evidence and is never selected from an individual patient score. A weighted ensemble, detection boxes and segmentation masks are not active.

The formal fourth-year comparison remains ResNet-18 versus EfficientNet-B0, where EfficientNet-B0 is the recorded baseline. Its older performance and external-reference results are historical baseline evidence only; they must not be presented as validation of the current three-route dashboard.

Remaining research limitations: unavailable patient linkage; reused development-validation cohorts; one training seed; incomplete fastMRI reference annotations; no anatomy, out-of-distribution or mixed-patient detection; no clinical validation of DICOM/NIfTI conversion; no prospective clinical validation, reader study, lesion annotations, detection boxes or segmentation masks. These are research and clinical-validation gaps, not issues that can be solved by cosmetic dashboard changes. The release is for academic demonstration and controlled research testing, not clinical diagnosis.
