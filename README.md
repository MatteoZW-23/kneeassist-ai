# KneeAssist AI

## Development of a Deep Learning System for Study Level Classification of Normal and Abnormal Knee MRI Studies

Local knee MRI research software whose primary academic task is study-level normal/abnormal classification. ACL tear and meniscal tear are retained as secondary research scores. It is not an autonomous diagnostic system or a replacement for a radiologist.

## Current status

Ready for **research testing**, with important performance limitations. The selected model is calibrated EfficientNet-B0 from the frozen warm-up stage. Both fine-tuning experiments finished; neither beat that checkpoint on the internal selection criterion. Internal AUROC rose slightly, but MRNet official-validation and both external AUROCs fell. This is not a demonstrated overall model improvement.

The future platform direction, current capabilities and evidence required before adding adaptive model routing, detection or segmentation are documented in [docs/SYSTEM_ROADMAP.md](docs/SYSTEM_ROADMAP.md).

The isolated five-architecture MRNet benchmark under `runs/model_family_v2` is complete. It compared ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin-T without replacing the deployed V1 checkpoint. EfficientNet-B0 was best among the new candidates (internal-tuning macro AUROC 0.8472), but it did not beat the active calibrated baseline (0.8478), so it was not promoted. See [the benchmark decision](runs/model_family_v2/FINAL_DECISION.md).

The active research workflow previously passed nine notebooks and the real browser upload/prediction/attention/export/clear checks. After the benchmark, dashboard and guarded-ensemble additions, the current full software suite passed 33 automated tests. See [the readiness report](runs/mri_finetune_02/READINESS_REPORT.md) for measured results and limits. Active deployment: `config.yaml`.

## GitHub release contents

This repository includes the source code, the active 19 MB research checkpoint at `models/best_model.pth`, configuration, tests, notebooks, model card, experiment evidence, and academic report source. MRI datasets, source archives, example studies, cached encoder weights, logs, and private links are excluded. Obtain data only from the original providers and follow their terms; the released checkpoint was trained on MRNet only.

For the work that still requires independent clinical data and qualified review, see [the validation gaps and evidence plan](docs/CLINICAL_VALIDATION_GAPS.md). Software completion does not close those gaps.

The RSNA Kaggle pilot is separate from the active model. Its future review and training gate is documented in [the RSNA label-review protocol](docs/RSNA_LABEL_REVIEW_PROTOCOL.md).

## Launch and use

Double-click **Launch KneeAssist AI.cmd**. On a new installation, use **Install.cmd** first with a compatible Python installation. The installed environment on this machine is `.venv`. The interface runs locally at http://127.0.0.1:8501.

Equivalent launch command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload a NumPy stack, NIfTI volume, DICOM slices, or a study ZIP. Each processed array must be a finite numeric MRI volume shaped slices x height x width. The intake panel reports file type, shape, dimensions, slice count, datatype, intensity range and orientation evidence before inference. It prioritises DICOM geometry and NIfTI affine evidence. A processed NPY array with no anatomical metadata, such as `0000.npy`, requires a manual plane confirmation; the filename is not treated as proof of orientation. The app supports one- or two-plane studies and warns when planes are missing. Enter a pseudonymous case reference and select Analyse study. View the scores, slice viewer and Grad-CAM; export the summary as text or JSON. Clear current case resets the session. No example MRI study is included in the GitHub release.

JPEG/PNG picture uploads, internet scan screenshots, knee photographs and X-rays do not receive predictions. File checks cannot establish anatomy, patient identity, mixed-patient archives, acquisition suitability, or clinical validity. NIfTI and direct DICOM support are technically inspected and converted in memory, but have not been clinically validated against the MRNet-trained model.

## Smart input and model selection

`src/routing/smart_input_router.py` provides the read-only intake report. `model_registry/active_models.json` records the active model, compatible planes, targets, metrics and preprocessing reference. `src/selection/model_selector.py` chooses only from existing eligible registry entries using validation AUROC, sensitivity, PR-AUC and F1 weighting. The registry currently contains one calibrated EfficientNet-B0 research checkpoint, so every supported target routes to that checkpoint and no ensemble is claimed. See [docs/SYSTEM_ROADMAP.md](docs/SYSTEM_ROADMAP.md) before adding new models, detection or segmentation.

The project also includes guarded ensemble and localisation infrastructure. An ensemble can run only after validated member models, calibrated probabilities and non-guessed weights are registered. Detection boxes and segmentation masks can run only after trained, validated localisers with verified annotation provenance are registered. The active deployment has no such entries, so it continues to use one calibrated classifier and Grad-CAM attention. See [the ensemble and localisation gate](docs/ENSEMBLE_AND_LOCALIZATION_GATE.md).

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

Standalone inference uses the active checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.inference.predictor sample_cases/mrnet_1130.zip --case EXAMPLE --json
```

Evaluation saves per-target and macro AUROC, F1, sensitivity, specificity, precision and accuracy; confusion matrices, ROC/precision-recall curves, training curves and prespecified prediction/attention examples. The selected run's `results` directory contains these artifacts. `results/model_comparison.json` records the selection. Checkpoint metadata identifies its dataset, architecture, configuration and training stage. Saved probabilities do not establish confidence that a diagnosis is correct.

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
