# KneeAssist XAI

## Development of an Explainable Deep Learning System for Study-Level Classification of Normal and Abnormal Knee MRI Studies

Local knee MRI research software whose primary academic task is study-level normal/abnormal classification. ACL tear and meniscal tear are retained as secondary research scores. It is not an autonomous diagnostic system, is not clinically validated, and does not replace a radiologist or recommend treatment.

## Documentation

Read [the documentation guide](docs/README.md) for the user guide, academic submission files, model-policy explanation and evidence boundaries.

## Current status

Ready for **research testing**, with important performance limitations. The dashboard routes each finding to a compatible model selected before upload from fixed MRNet development evidence: DenseNet-121 for general abnormality, ResNet-18 for ACL tear and Swin Transformer for meniscal tear. It never selects a model because that model gives the highest score for an individual case.

The five-architecture MRNet benchmark compared ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin-T. A later target-level portfolio used a separate calibration subset and a locked tuning partition to choose the three registered routes. The target-routed system reached macro AUROC 0.846 on the previously used 120-study MRNet official development-validation cohort. This is not independent external or clinical validation. See [the active model card](models/MODEL_CARD.md), [portfolio report](runs/model_portfolio_v1/portfolio_report.json) and [final routed metrics](runs/model_portfolio_v1/final/metrics.json).

The research workflow includes automated tests for input errors, a real MRI upload, predictions, Grad-CAM payload structure, export and case clearing. Active deployment and routing are recorded in `model_registry/active_models.json`.

## GitHub release contents

This repository includes source code, configuration, tests, notebooks, the model card, experiment evidence, academic report source and the historical EfficientNet-B0 baseline checkpoint at `models/best_model.pth`. The active three-route dashboard also requires local DenseNet-121, ResNet-18 and Swin-T checkpoints recorded in `model_registry/active_models.json`; those local route weights are excluded from Git because experiment weights and MRI materials are not redistributed. MRI datasets, source archives, example studies, cached encoder weights, logs and private links are excluded. Obtain data only from original providers and follow their terms.

For the work that still requires independent clinical data and qualified review, see [the validation gaps and evidence plan](docs/CLINICAL_VALIDATION_GAPS.md). Software completion does not close those gaps.

The RSNA Kaggle pilot is separate from the active model. Its future review and training gate is documented in [the RSNA label-review protocol](docs/RSNA_LABEL_REVIEW_PROTOCOL.md).

## Launch and use

Double-click **Launch KneeAssist AI.cmd** to launch KneeAssist XAI. On a new installation, use **Install.cmd** first with a compatible Python installation. The installed environment on this machine is `.venv`. The interface runs locally at http://127.0.0.1:8501.

Equivalent launch command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload a NumPy stack, NIfTI volume, DICOM slices, or a study ZIP. Each processed array must be a finite numeric MRI volume shaped slices x height x width. The intake panel reports file type, shape, dimensions, slice count, datatype, intensity range and orientation evidence before inference. It prioritises DICOM geometry and NIfTI affine evidence. A processed NPY array with no anatomical metadata, such as `0000.npy`, requires a manual plane confirmation; the filename is not treated as proof of orientation. The app supports one- or two-plane studies and warns when planes are missing. Enter a pseudonymous case reference and select Analyse study. View the scores, slice viewer and Grad-CAM; export the summary as text or JSON. Clear current case resets the session. No example MRI study is included in the GitHub release.

For each Grad-CAM view, the dashboard states the selected finding, MRI sequence and slice number. It describes the strongest heatmap concentration only in displayed-image coordinates, such as “upper-right area of the displayed image.” Warm colours show regions that contributed positively to that selected model score. The map is generated at low feature-map resolution and enlarged for display, so it is coarse. It does not identify a named knee structure, confirm a tear, define a lesion boundary, estimate lesion size or exclude pathology elsewhere in the study. The dashboard intentionally does not average attention across planes or let a user artificially intensify the overlay.

JPEG/PNG picture uploads, internet scan screenshots, knee photographs and X-rays do not receive predictions. File checks cannot establish anatomy, patient identity, mixed-patient archives, acquisition suitability, or clinical validity. NIfTI and direct DICOM support are technically inspected and converted in memory, but have not been clinically validated against the MRNet-trained model.

## Smart input and model selection

`src/routing/smart_input_router.py` provides the read-only intake report. `model_registry/active_models.json` records the registered models, compatible planes, targets, metrics and preprocessing reference. `src/selection/model_selector.py` chooses only from existing eligible entries using validation AUROC, sensitivity, average precision (AP) and F1 weighting. The current research registry routes general abnormality to DenseNet-121, ACL tear to ResNet-18 and meniscal tear to Swin Transformer. The selection was locked before upload; individual prediction scores never select a model. No weighted ensemble is active because it has not been shown to improve results on a held-out protocol.

The project also includes guarded ensemble and localisation infrastructure. An ensemble can run only after validated member models, calibrated probabilities and non-guessed weights are registered. Detection boxes and segmentation masks can run only after trained, validated localisers with verified annotation provenance are registered. The active deployment has three fixed target-specific classifiers, no weighted ensemble and no localiser; it uses Grad-CAM attention only. See [the ensemble and localisation gate](docs/ENSEMBLE_AND_LOCALIZATION_GATE.md).

## Datasets and provenance

| Dataset | Role | Limitation |
|---|---|---|
| MRNet-v1.0, 1250 studies | Only supervised training source; abnormal/ACL/meniscus labels | Patient linkage unavailable; official validation previously evaluated |
| KneeMRI, 917 series / 909 exams | External ACL reference evaluation | ACL-only labels; previously evaluated; not a new untouched test |
| fastMRI, 199 reconstructed H5 volumes | Exploratory external meniscus evaluation using fastMRI+ annotations | Non-exhaustive annotations; missing annotation is not definitive clinical absence |

X-ray and unverified alternate RSNA archives are not training inputs. Source archives remain preserved. Private download links are not for redistribution. Confirm original data agreements before sharing images or packaging datasets.

The frozen baseline uses 904 fitting, 226 tuning and 120 official-validation studies. The improvement experiment uses **804 fitting, 226 tuning, 100 calibration and 120 official-validation studies**. Calibration cases were removed from the old fitting partition. They must never calibrate the incumbent model, which already trained on them. Exact-series SHA256 and study overlap checks run before training. No patient-independence claim is made without reliable patient identifiers.

## Architecture and preprocessing

Both ResNet-18 and EfficientNet-B0 start from ImageNet weights. Each study uses up to 12 evenly spaced slices per available plane, within-volume percentile normalization, 224 x 224 resizing, and ImageNet channel normalization. Slice features undergo mean/max pooling, then plane concatenation and a missing-plane mask feed a multilabel head. Targets are configured in YAML.

The baseline freezes the encoder. The improvement experiment fits a fresh head and then fine-tunes the final encoder blocks with a smaller learning rate. Earlier blocks and BatchNorm statistics remain frozen. It uses modest per-sequence rotation/scale augmentation, modality dropout, gradient accumulation, weighted BCE, AdamW, a validation scheduler, early stopping and resumable checkpoints. FP32 encoder operations avoid known T500 convolution instability; CUDA mixed precision is used for the head with guarded loss scaling. CPU fallback exists, but training is substantially slower.

The locked protocol is `runs/mri_finetune_02/protocol.json`. Model selection uses tuning macro AUROC. External results never select checkpoints or tune hyperparameters. If a candidate wins, positive-slope probability calibration and maximum-Youden-J thresholds use only the reserved calibration cohort. That threshold is a research operating point, not a clinically validated threshold. If no candidate wins, the existing baseline remains selected. One seed was used; no multi-seed stability claim is made.

## Training, resume and evaluation

The improvement experiment is complete. The command below recognizes completed artifacts; do not start additional training without a new experiment protocol. The pipeline uses an operating-system file lock to reject duplicate processes.

```powershell
.\.venv\Scripts\python.exe -m scripts.complete_improvement
```

The pipeline resumes completed stages and last saved epochs, compares models, calibrates eligible candidates, evaluates the selected model, verifies inference/Grad-CAM, runs tests and executes notebooks. It keeps the previous deployment until candidate inference succeeds, and restores the previous deployment config if software verification fails. A browser workflow check follows automated checks before the final readiness notification.

Historical baseline resume:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --config configs/mrnet_fresh_01.yaml --resume
```

The historical single-checkpoint inference command below is retained only for the EfficientNet-B0 baseline. The active routed dashboard loads its models through `model_registry/active_models.json`; use the dashboard for the current three-route workflow.

```powershell
.\.venv\Scripts\python.exe -m src.inference.predictor sample_cases/mrnet_1130.zip --case EXAMPLE --json
```

Evaluation saves per-target and macro AUROC, F1, sensitivity, specificity, precision and accuracy; confusion matrices, ROC/precision-recall curves, training curves and prespecified prediction/attention examples. The selected run's `results` directory contains these artifacts. `results/model_comparison.json` records the selection. Checkpoint metadata identifies its dataset, architecture, configuration and training stage. Saved probabilities, threshold distance and exploratory MC-dropout variation do not establish confidence that a diagnosis is correct.

## Project map

- `app.py`, `launch.py`: dashboard and launcher
- `src/data`: safe inputs, dataset manifests and preprocessing
- `src/models`: encoders and study aggregation
- `src/training`: baseline and MRI fine-tuning pipelines
- `src/evaluation`: metrics, curves, calibration and external evaluations
- `src/inference`: independent prediction and summaries
- `configs`: versioned experiment settings; `config.yaml`: active deployment
- `runs`: checkpoints, logs, protocols and results for each experiment
- `notebooks`: nine clearly named notebooks, indexed in `notebooks/README.md`
- `tests`: software and inference checks
- `data`: preserved datasets, archives, manifests and external annotations
- `docs`: audit, provenance and historical documentation
- `history`: preserved earlier deliverables; these are not the active deployment

Run software checks with `.venv/Scripts/python.exe -m pytest tests -q`. Software test success is not evidence of clinical validity.

## Limits

Training on more data or fine-tuning does not guarantee improvement. This system lacks verified patient linkage, fresh independent clinical testing, prospective workflow validation, validated anatomy/out-of-distribution rejection, and diagnosis from photographs. It predicts only three findings; it does not grade injuries or detect all knee conditions. Grad-CAM indicates model attention, not a confirmed lesion or validated segmentation. Negative results do not rule out injury. Research and decision-support output requires clinical review.



