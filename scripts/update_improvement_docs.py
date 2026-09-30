from src.utils import ROOT
import shutil

archive=ROOT/'docs/README_before_improvement_02.md'
if not archive.exists():shutil.copy2(ROOT/'README.md',archive)
(ROOT/'README.md').write_text('''# KneeAssist AI

Local knee MRI research software for study-level general abnormality, ACL tear and meniscal tear scores. It is not an autonomous diagnostic system or a replacement for a radiologist.

## Current status

The existing research baseline remains available while a new MRI fine-tuning experiment runs. New performance is **not yet established**. The process status is saved in `runs/mri_finetune_02/completion_status.json`; per-epoch progress is in `progress.json`. Notebook 08 displays these files. The active deployment is always specified by `config.yaml`.

The baseline's previously measured MRNet macro AUROC is 0.850. Its important weaknesses include 20% specificity for general abnormality, external KneeMRI ACL AUROC 0.635, and exploratory fastMRI meniscus AUROC 0.674. These are research results, not clinical acceptance criteria. See `docs/HONEST_MODEL_AUDIT.md` and `runs/mrnet_fresh_01/READINESS_REPORT.md`.

## Launch and use

Double-click **Launch KneeAssist AI.cmd**. On a new installation, use **Install.cmd** first with a compatible Python installation. The installed environment on this machine is `.venv`. The interface runs locally at http://127.0.0.1:8501.

Equivalent launch command:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

Upload a ZIP containing `axial.npy`, `coronal.npy` and `sagittal.npy`, or upload the named arrays individually. Each array must be a finite numeric MRI volume shaped slices x height x width. Enter a pseudonymous case reference and select Analyse study. View the scores, slice viewer and Grad-CAM; export the summary as text or JSON. Clear current case resets the session. The real local example is `sample_cases/mrnet_1130.zip`.

JPEG/PNG picture uploads are not supported. Internet scan screenshots, knee photographs and X-rays do not receive diagnostic predictions. File checks cannot establish anatomy or acquisition suitability. Direct DICOM/NIfTI import is not implemented. Missing MRI planes produce a warning; their clinical performance is not established.

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

The locked protocol is `runs/mri_finetune_02/protocol.json`. Model selection uses tuning macro AUROC. External results never select checkpoints or tune hyperparameters. If a candidate wins, positive-slope probability calibration and maximum-Youden-J thresholds use only the reserved calibration cohort. That threshold is a research operating point, not a clinically validated threshold. If no candidate wins, the existing baseline remains selected. One seed is planned; no multi-seed stability claim is made.

## Training, resume and evaluation

The improvement process runs in the background. **Do not launch a second run while it is active.** The pipeline uses an operating-system file lock to reject duplicate processes.

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.complete_improvement
```

The pipeline resumes completed stages and last saved epochs, compares models, calibrates eligible candidates, evaluates the selected model, verifies inference/Grad-CAM, runs tests and executes notebooks. It keeps the previous deployment until candidate inference succeeds, and restores the previous deployment config if software verification fails. A browser workflow check follows automated checks before the final readiness notification.

Historical baseline resume:

```powershell
.\\.venv\\Scripts\\python.exe -m src.training.train --config configs/mrnet_fresh_01.yaml --resume
```

Standalone inference uses the active checkpoint:

```powershell
.\\.venv\\Scripts\\python.exe -m src.inference.predictor sample_cases/mrnet_1130.zip --case EXAMPLE --json
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
''',encoding='utf-8')
(ROOT/'START_HERE.md').write_text('''# Start here

Double-click **Launch KneeAssist AI.cmd** to open the current research baseline.

An improvement experiment is now running separately. It trains both architectures with MRI encoder fine-tuning and tests whether either outperforms the baseline on internal validation. Do not start another training process. Keep the laptop plugged in and awake.

- Current experiment: `runs/mri_finetune_02/completion_status.json`
- Detailed progress: `runs/mri_finetune_02/progress.json`
- Human-readable progress notebook: `notebooks/08_MRI_Fine_Tuning_Calibration_and_Run_Status.ipynb`
- Active checkpoint: the path in `config.yaml`
- Existing measured results: `runs/mrnet_fresh_01/READINESS_REPORT.md`

The experiment is not complete until evaluation, inference, tests, notebooks and the browser workflow are checked. No performance improvement is claimed yet. Training uses MRNet only. KneeMRI and fastMRI remain evaluation references with documented limits. JPEG/PNG uploads are not supported. Research use; clinical review required.
''',encoding='utf-8')

if __name__=='__main__':print('Updated active documentation; previous README preserved in docs.')
