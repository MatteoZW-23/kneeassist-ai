# RSNA V2 label review protocol

## Purpose

This protocol prepares the Kaggle RSNA data for a future research training run. It does not create labels automatically. Clinical labels must be supplied by qualified reviewers under the appropriate governance arrangements.

## Step 1 Prepare a blinded manifest in Kaggle

Attach the RSNA competition data to the private Kaggle notebook. Set a private temporary salt and run:

```bash
export RSNA_REVIEW_SALT="a-private-random-value"
python scripts/build_rsna_review_manifest.py \
  --dicom-root /kaggle/input/your-rsna-dicom-folder \
  --output /kaggle/working/rsna_review_manifest.csv
```

The resulting manifest uses a pseudonymous case ID. Do not publish the salt, patient identifiers, raw images or private links.

## Step 2 Independent review

Give each reviewer the same cases and the blank `data/external/RSNA_review/label_template.csv` schema. Each reviewer should complete one row per case and use only:

- `0` for absent,
- `1` for present,
- `uncertain` if the finding cannot be determined,
- `acceptable`, `unacceptable`, or `uncertain` for image quality.

The reviewed targets are general abnormality, ACL tear and meniscal tear. The reviewers should record exclusions and uncertain findings rather than guessing.

## Step 3 Agreement and adjudication

Combine both review files into one CSV, then run:

```bash
python scripts/validate_rsna_review_labels.py \
  --manifest rsna_review_manifest.csv \
  --labels combined_reviews.csv \
  --approved-output approved_labels.csv \
  --queue-output adjudication_queue.csv
```

Only cases with two distinct reviewers, acceptable image quality and agreement on all three targets enter `approved_labels.csv`. All other cases go to the adjudication queue. The script never decides a label.

## Step 4 Lock V2 before training

Before training, record the manifest version, reviewer agreement, adjudication rules, class counts, data split, seed, checkpoint selection rule and final test cohort. Use `configs/kaggle_v2_review.yaml` as the minimum gate. The final external test cohort must not influence data cleaning, threshold selection or model choice.

## Current boundary

There are no reviewed RSNA labels in this project yet. Therefore no RSNA V2 model has been trained or added to the dashboard. This protects the existing MRNet research model from being replaced by an unverified experiment.
