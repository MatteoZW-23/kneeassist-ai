# fastMRI validation archive received

Source supplied: C:/Users/HP/Desktop/knee_singlecoil_val.tar
Active imported evaluation inputs: data/external/fastMRI/images. The raw source archive was moved recoverably outside the active project on 2026-10-03.
The header identifies XZ compression, not an uncompressed TAR. The Desktop source was locked by another application, so it was copied rather than moved. It remains on the Desktop.

The importer writes only regular HDF5 archive members into data/external/fastMRI/images, validates finite three-dimensional reconstructed images, and matches file IDs to the existing fastMRI+ annotation and reviewed-file lists. It writes progress to import_status.json and an inventory to image_manifest.csv. Full XZ stream integrity is checked before status becomes complete. The archive filename alone does not establish successful import.

Use reconstruction_rss images where available. The observed acquisitions are coronal; do not label these sagittal. Bounding boxes require the documented vertical flip used by fastMRI+ when creating annotation images. No bounding-box localisation is currently implemented by this importer.

ACL sprain labels are not automatically equivalent to MRNet ACL tears. Reviewed files with no annotation rows mean no reported findings under that annotation protocol, not exhaustive clinical proof of a normal knee. Keep unreviewed files distinct. Patient independence needs further assessment.

Training and predictions remain disabled during dataset validation. Old checkpoint deletion is still pending from the earlier blocked cleanup. New external data should have its evaluation protocol frozen before inspecting model performance; do not both train on the full cohort and present it as untouched external validation.
