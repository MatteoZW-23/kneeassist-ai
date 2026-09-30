## Picture preview
Select Picture preview to view a genuine JPEG/PNG up to 10 MB and 20 megapixels. No diagnosis, model score or Grad-CAM is generated for pictures. Damaged or unsupported files show a readable error. File validation does not verify knee anatomy. Notebook names and their index are updated in notebooks/README.md.

# CURRENT STATUS: ready for research testing

The selected fresh model is enabled. See [START_HERE.md](START_HERE.md) and [the readiness report](runs/mrnet_fresh_01/READINESS_REPORT.md) for current results, testing instructions and limitations. Eight notebooks and 19 automated checks passed. MRNet-only training; KneeMRI and fastMRI evaluation.

## Previous implementation notes (historical)

# CURRENT STATUS: waiting for datasets

Training and dashboard predictions are disabled. Requested removal of old trained artifacts was blocked; old files remain and are not a fresh run. Original datasets are preserved. See [the retraining plan](docs/RETRAINING_PLAN.md) for the folder map, pending work and evaluation safeguards.

The previous implementation documentation follows; its completed-run claims describe the previous run only.

# KneeAssist AI V1

A local knee MRI research and clinical decision-support prototype. It predicts general abnormality, ACL tear, and meniscal tear at study level. It does not provide definitive diagnoses, replace radiologists, or confirm a lesion from an attention map.

## Open the application

On this laptop, double-click **Launch KneeAssist AI.cmd**. The installed environment is in `.venv`; the application opens at `http://127.0.0.1:8501`. If installing on a new machine, install Python 3.14 and then double-click **Install.cmd** once (internet required). The CUDA package is intended for a compatible NVIDIA driver. CPU inference is supported automatically when CUDA is unavailable.

Equivalent launch command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep the machine awake during training. Uploads are processed locally in memory. They are not written to a case database or sent to an external inference service. The app binds to localhost; it is not configured as a shared clinical server.

## Accepted studies

Upload one or more named stacks: `axial.npy`, `coronal.npy`, `sagittal.npy`; alternatively upload a ZIP containing those files. Arrays must contain numeric finite values with shape **slices Ãƒâ€” height Ãƒâ€” width**, with at least 16 pixels per image dimension. Object arrays, nonfinite pixels, blank studies, corrupt files, ambiguous plane names, duplicate planes, oversized arrays and unsafe ZIP paths are rejected with readable messages.

This V1 uses **MRNet-style NumPy MRI stacks**. It does not directly import DICOM or NIfTI or identify the plane automatically. Converting clinical studies requires preserving acquisition orientation and sequence meaning; renaming unrelated images does not make them compatible. The model was developed on knee MRI and cannot verify body part from an upload. All three planes are preferred. Incomplete studies are accepted with a warning, but their clinical performance has not been separately validated.

The included `sample_cases/mrnet_1130.zip` is a real MRNet validation study for local testing. Enter a pseudonymous case reference, select **Analyse study**, inspect the slice viewer and finding-specific attention, and export a text or JSON summary. **Clear current case** clears the session's inputs and results. Changing inputs invalidates old predictions.

## Data and provenance

Both V1 checkpoints are trained exclusively on the already selected **MRNet-v1.0** data: 1,130 original training studies and 120 original validation studies, each with axial, coronal and sagittal stacks. Labels are abnormality, ACL tear and meniscus tear. The existing split is retained: 904 fitting studies, 226 internal validation studies, and 120 official validation studies. No RSNA or X-ray data are mixed into these models.

Study IDs and exact SHA-256 series content are checked across partitions. Image normalization is computed within each sequence; feature normalization, class weights, and optimization use fitting studies only. Architecture/checkpoint and decision-threshold selection use internal validation only. Official validation is used for the final selected model's evaluation. The official cohort was already examined in the earlier prototype; it is a development-validation cohort, not a new external test set.

**Patient linkage is not provided in this local MRNet copy.** Study separation does not prove patient independence if multiple examinations belong to the same person. A `study_id,patient_id` CSV can be set through `dataset.patient_mapping`; supplied patient groups are checked across every partition and training stops if leakage is found. No patient identity is inferred from an exam number. `results/leakage_audit.json` records what could and could not be verified.

Dataset source: https://aimi.stanford.edu/datasets/mrnet-knee-mris . Labels were originally extracted from clinical reports. Preserve the applicable MRNet access agreement and attribution. Commercial/clinical deployment rights have not been established by this project. Do not redistribute dataset contents without checking the source terms.

## Architecture and preprocessing

Two models are compared: **ResNet-18** and **EfficientNet-B0**, both initialized with torchvision ImageNet pretrained weights. On this 4 GB NVIDIA T500, V1 uses a frozen image encoder and a trained multilabel study classifier. It is a transfer-learning baseline, not an end-to-end MRI-fine-tuned encoder or a reproduction of the published MRNet architecture.

Each available sequence is independently clipped and scaled using its 1st/99th intensity percentiles. Up to 12 uniformly spaced slices are resized to 224Ãƒâ€”224, repeated into three channels, and normalized with ImageNet statistics. Mean and maximum slice features are concatenated for each plane. Available-plane indicators accompany the features, and a 128-unit dropout classifier produces one logit per target. Training occasionally drops a plane to support incomplete inputs; this does not substitute for missing-plane validation. Small focal lesions may be missed by slice sampling.

Class imbalance is handled using fitting-set positive weights with `BCEWithLogitsLoss`. The frozen encoder uses FP32 because the local T500/cuDNN combination produced nonfinite FP16 convolution outputs in a real-data trial. The trainable classifier uses CUDA automatic mixed precision and gradient scaling, AdamW, gradient clipping, ReduceLROnPlateau, early stopping, deterministic seeds, best checkpoints, and full resume checkpoints containing optimizer, scheduler, scaler and random-generator states. Exact reproducibility is intended on the same environment/hardware; changes in device, CUDA or library versions can change results. Feature caches are keyed by architecture, preprocessing and input content hashes.

## Training and resume

Double-click **Train or Resume.cmd**, or run:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --resume
```

Important settings are in `config.yaml`: data paths, target labels, image size, slice count, batch size, epochs, learning rate, dropout, model architectures, workers, thresholds, device and checkpoint path. Frozen-encoder extraction currently processes one study at a time without worker processes; `num_workers: 0` documents that operating mode. The configured dataset manifest supplies concrete stack paths. Retain the original fit/tune/validation split. To train a new experiment with changed configuration, archive previous models/results first; resume intentionally rejects mismatched configurations.

Progress is written to `logs/status.json` and training output to the run's log. Each architecture saves `models/<architecture>_best.pth` and `<architecture>_last.pth`. Best architecture selection uses **internal validation macro AUROC**, never training accuracy. Thresholds maximize per-target F1 on internal validation; these are research operating points, not clinical thresholds. Probabilities from weighted training are not calibrated estimates of real-world prevalence or patient risk.

`models/best_model.pth` is the selected, self-contained checkpoint (encoder, head, normalization, target labels, thresholds, configuration and dataset provenance). Load only trusted application-owned checkpoints; Python checkpoint loading is not an upload feature.

## Evaluation

`results/REPORT.md` is generated after successful training/evaluation. It reports per-target and macro AUROC, F1, sensitivity, specificity, precision and accuracy, along with model comparison. Detailed outputs include:

- `results/<architecture>/history.json`, `training_validation_curves.png`, ROC/PR curves and internal-validation confusion matrices.
- `results/model_comparison.json` with the selection criterion and both model results.
- `results/final/metrics.json`, `predictions.csv`, ROC/PR curves and confusion matrices for the selected model.
- `results/final/example_*.png` and `.txt` containing the first prespecified validation examples rather than cherry-picked correct predictions.
- `results/leakage_audit.json` for partition/content checks and their limitations.

AUROC is a ranking measure, not percentage accuracy. Small samples, class balance and single-center acquisition affect generalization. No external hospital or prospective clinical validation has been performed.

## Independent inference and explainability

No training notebook is needed:

```powershell
.\.venv\Scripts\python.exe -m src.inference.predictor sample_cases/mrnet_1130.zip --case RESEARCH-001
```

Add `--json` for structured output. Inference loads the saved model and its preprocessing/threshold configuration. The app's score-separation indicator summarizes distance from model thresholds; it is explicitly **not clinical confidence** or a correctness probability. A below-threshold finding does not rule out injury.

Grad-CAM differentiates a target's study-level logit through feature standardization, classifier and mean/max slice pooling to the final encoder spatial maps. Other available planes remain in the study context. Original sampled slices and positive attention overlays can be displayed for any target/available plane. No heatmap is presented as a confirmed lesion or segmentation. A zero positive signal produces an explanatory message.

## Structure

```text
KneeAssist-AI/
  app.py, launch.py, config.yaml, requirements.txt
  Launch KneeAssist AI.cmd, Install.cmd, Train or Resume.cmd
  data/                 MRI data, manifest, splits and cached features
  models/               checkpoints and pretrained image weights
  results/              comparisons, metrics, plots, examples and test evidence
  logs/                 progress and application logs
  sample_cases/         one local validation study for testing
  notebooks/            five executable, output-bearing workflow notebooks
  src/data/             dataset checks, secure input handling, preprocessing
  src/models/           ResNet/EfficientNet encoders and study aggregation
  src/training/         loss, scheduler, training/resume workflow
  src/evaluation/       metrics, curves, Grad-CAM and final evaluation
  src/inference/        standalone model predictor
  tests/                input, metric, model and Streamlit interaction checks
```

## Validation and limitations

Run `.\.venv\Scripts\python.exe -m pytest tests -q` for automated checks. Tests include real MRI upload through Streamlit's test framework, prediction rendering, attention generation, export controls, stale-result prevention, and clear-case behavior when the checkpoint is present. The browser/server must also be exercised before delivery; test evidence is saved under `results`.

Additional labels can be configured by adding reliable columns to the manifest and targets to configuration, then retraining the model. Existing checkpoints cannot gain new diagnostic capabilities simply by editing the UI. Future work includes controlled MRI encoder fine-tuning, improved slice coverage, calibration, verified patient/site-independent testing, reviewed RSNA labels, and a separate X-ray OA model.

**Research and clinical decision-support prototype. Clinical review required. Not an autonomous diagnostic system.**


## Release archive

`release/KneeAssist-AI-V1.zip` packages the application, both best architecture checkpoints, the selected checkpoint, results, tests and one local research example. It excludes the full MRNet corpus, feature caches and `.venv` to keep the package manageable. The Desktop installation retains the full prepared data and environment. On another machine, run Install.cmd before launching; obtain MRNet through its authorized source if retraining is required. Check dataset and pretrained-weight terms before sharing the archive, especially its included example study.


## Executable notebooks

The notebooks folder contains five notebooks, executed against the actual V1 data and checkpoints: environment/data checks; ResNet/EfficientNet training and resume; evaluation/model comparison; real-study inference and Grad-CAM; and application/reliability tests. Open them in a notebook editor using **KneeAssist AI (project .venv)**. They retain real outputs and call the production modules. The training notebook preserves an already completed run; it can start or resume training when the dataset is available. See `notebooks/README.md` for the ordered index and `results/notebook_execution.json` for execution evidence.

## External KneeMRI cohort
The locally downloaded cohort at `data/external/KneeMRI/original_download` is configured under `external_kneemri` in config.yaml. It contains 917 volumes across 909 exam IDs. It is reserved for external evaluation, not fitted or used for threshold selection.

Run/resume: `.venv\Scripts\python.exe -m src.evaluation.external_kneemri`.
Results: `results/external_kneemri/`. The protocol records the frozen checkpoint hash before inference. Repeated series are averaged per exam; the highest reference severity defines its reference class. Primary positive: partial or complete ACL injury. Secondary: complete rupture versus healthy ACL, excluding partial injury. No general-abnormality or meniscus reference labels are available. Patient independence is not proven, and sagittal-only input introduces missing-plane as well as institutional shift.

Notebook: `notebooks/05_Evaluate_ACL_on_External_KneeMRI.ipynb`.
Dashboard: upload `sample_cases/kneemri_external_example.zip`, which contains a converted full sagittal volume. The source filename and attribution are recorded alongside it. This converted example is for local research; do not redistribute it in the release package. Original dataset license: CC BY-NC-ND 4.0, source https://zenodo.org/records/14789903 .

Pickle files are imported offline with a restricted NumPy allowlist; the web uploader does not accept arbitrary pickle objects. No source files are deleted or altered. fastMRI integration remains pending the requested download and label matching.
