# No active source archives

The active project stores only data needed for its current workflow:

- `../MRNet-v1.0`: the supervised MRNet training and validation studies used by the active checkpoint.
- `../external/KneeMRI`: retained external ACL evaluation cohort.
- `../external/fastMRI`: retained exploratory meniscus evaluation inputs.

On 2026-10-03, duplicate raw source copies of those datasets were moved, recoverably, to `C:\Users\HP\Desktop\KneeAssist-AI_Recovery_Archive_20261001\source_data_duplicates_20261003`. They are not read by training, evaluation, inference, or the dashboard.

The active model remains MRNet-trained only. External datasets must not be added to its training set without a new label audit and a new untouched evaluation protocol.
