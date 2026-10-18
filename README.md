# KneeAssist XAI

## Explainable study-level knee MRI classification research prototype

KneeAssist XAI is local research software for supported knee MRI studies. It reports research scores for **general abnormality**, **ACL tear**, and **meniscal tear**. It is not a diagnostic device, does not recommend treatment, and does not replace a radiologist.

## Active model workflow

The active configuration uses five ImageNet-pretrained transfer-learning candidates trained from the same saved MRNet V1 study roles, preprocessing configuration and three reference-label targets:

| Candidate architectures |
|---|
| ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, Swin Transformer |

For each supported upload, the application runs all five candidates on the same prepared MRI study. A saved regularised selector then uses their current probabilities, prediction entropy, model disagreement and available-plane descriptors to estimate each architecture's mean Brier loss. The architecture with the lowest estimated loss is selected for **all three outputs** and for Grad-CAM generation. The selector cannot access the unknown study label. No retraining occurs during upload.

This is case-specific model selection, not a weighted ensemble and not a claim that the selected architecture is correct for an individual patient.

## Saved evidence

The active registry is [model_registry/dynamic_selection.json](model_registry/dynamic_selection.json). The saved final evaluation is [runs/dynamic_selection_v1/final_evaluation.json](runs/dynamic_selection_v1/final_evaluation.json). Dynamic selection runs all five candidates and is restricted to complete axial, coronal and sagittal studies because its missing-plane performance was not evaluated.

On the recorded 120-study MRNet development evaluation, the complete dynamic system achieved macro AUROC **0.835**, average precision **0.833**, accuracy **0.764**, sensitivity **0.782**, specificity **0.681**, precision **0.736**, F1 **0.757**, Brier score **0.162**, and ECE-10 **0.096**. These are descriptive development results. The cohort had prior project exposure and the local copy has no patient identifiers, so this is neither independent patient-level validation nor clinical validation. Dynamic selection varied across the 120 studies, but it did not exceed every fixed candidate's macro AUROC; the saved evidence does not establish superiority over all fixed models.

Saved charts are in [runs/dynamic_selection_v1/figures](runs/dynamic_selection_v1/figures). The dynamic end-to-end prototype timing averaged 4.53 seconds for one MRI study on local CUDA hardware, including five sequential candidate executions; this is not a production-throughput measurement.

## Inputs and preprocessing

The app accepts MRI NumPy stacks (`.npy`, `.npz`), NIfTI volumes, DICOM slices and supported DICOM ZIP studies. It validates readable image data, dimensions, finite values and available metadata; it asks for a plane confirmation when an array's orientation cannot be determined. JPEG/PNG screenshots, online pictures, photographs and X-rays are rejected because they were not used to develop the model.

The saved pipeline uses p1–p99 intensity normalisation, resizing to 224 × 224 pixels, and twelve evenly spaced slices per axial, coronal and sagittal plane. The active dynamic selector rejects an incomplete study instead of extrapolating a three-plane result to an unvalidated missing-plane setting.

## Explainability and boundaries

Grad-CAM displays coarse positive image regions that contributed to a selected classifier score on one slice. It is **not** a confirmed lesion, a detection box, a segmentation mask or a treatment recommendation. No validated lesion detector or segmentation model is registered.

MRNet is the only supervised source. The model does not establish performance for MCL/PCL injury, fracture, effusion, cartilage damage or osteoarthritis. Below-threshold results do not rule out injury. Independent patient-linked, multi-site evaluation, clinical calibration and a radiologist reader study are still required before patient-care use.

## Run locally

Double-click `Launch KneeAssist AI.cmd` in the project folder. The dashboard runs locally at `http://127.0.0.1:8501`.

## Project map

- [app.py](app.py): Streamlit interface.
- [src/inference/dynamic_predictor.py](src/inference/dynamic_predictor.py): five-model execution and selector-backed inference.
- [src/dynamic_selection](src/dynamic_selection): selector features and regularised expected-loss estimators.
- [model_registry/dynamic_selection.json](model_registry/dynamic_selection.json): active candidate checkpoints and integrity hashes.
- [runs/dynamic_selection_v1](runs/dynamic_selection_v1): fixed roles, checkpoints, evaluation results and figures.
- [docs/ACTIVE_SYSTEM_EVIDENCE.md](docs/ACTIVE_SYSTEM_EVIDENCE.md): concise evidence and limitation guide.
- [docs/academic](docs/academic): dissertation materials and figures.
- [tests](tests): regression coverage.
