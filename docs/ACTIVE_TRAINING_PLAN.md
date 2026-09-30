# Active dataset and training plan

MRNet alone supplies fitting labels for general abnormality, ACL tear and meniscal tear. Keep the existing 904 fitting / 226 internal tuning / 120 official validation split. Preserve previous evaluation history: the official validation set is development validation, not a new external test.

KneeMRI is reserved for ACL regression evaluation. It was already evaluated previously. fastMRI is reserved for a future external meniscus evaluation after complete integrity, annotation, orientation and identity checks; do not use this cohort to select the training model. ACL sprain is not an automatic MRNet-tear equivalent. Missing annotations require the reviewed-file list and remain limited reference labels.

No X-ray, unidentified archive or alternate RSNA package is included in this run. Those files were moved outside the active project into `C:\Users\HP\Desktop\Other-Datasets-Not-Used`; inclusion would require a separate justified task.

Fresh run configuration: configs/mrnet_fresh_01.yaml. Outputs: runs/mrnet_fresh_01. Both ImageNet-pretrained frozen encoders and freshly initialised classification heads are trained. No prior locally fitted checkpoints or cached features are loaded. This is a reproducible fresh baseline, not a claim of improved architecture or clinical readiness.

Selection uses internal-validation macro AUROC; thresholds use internal-validation F1 as in the baseline. The known false-positive trade-off must be reported. Do not describe a new run as an improvement without comparing actual metrics. The main dashboard stays disabled until a selected checkpoint and inference checks are reviewed. Historical artifacts remain because deletion was blocked.
