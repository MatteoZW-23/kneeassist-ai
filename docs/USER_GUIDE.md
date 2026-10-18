# KneeAssist XAI dashboard user guide

## Before you analyse a study

1. Launch the application with `Launch KneeAssist AI.cmd`.
2. Upload one supported knee MRI study only. Supported formats are MRI NumPy stacks, NIfTI volumes, DICOM slices and structured study ZIP files.
3. Ensure that all uploaded sequences belong to the same case. The app cannot prove patient identity or confirm that a ZIP does not mix patients.
4. Enter a pseudonymous case reference if needed. It is a label for the browser session, not a diagnosis or patient identifier.

JPEG/PNG screenshots, photographs, X-rays and internet images are intentionally rejected. A metadata-free NPY stack needs manual axial, coronal or sagittal plane confirmation.

## What the intake panel does

The panel reports file type, shape, slice count, dimensions, numeric range and plane evidence. It checks whether the five registered candidates are available. It does not determine whether an image is clinically adequate or whether it contains a knee.

For the active dynamic selection workflow, use one axial, one coronal and one sagittal stack from the same MRI study. The app rejects a study with a missing plane because missing-plane dynamic-selection performance has not been evaluated.

## How to read the result screen

The score cards are study-level research scores. They are not confirmed diagnoses.

Above the cards, **Selected model for this study** shows the one architecture used for all three outputs. For every complete three-plane upload, the app runs ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0 and Swin Transformer. Its saved gradient-boosting selector uses current candidate probabilities, entropy, disagreement and study descriptors to estimate each candidate's mean Brier loss. The architecture with the lowest estimate is selected.

Each card repeats **Model used** directly below its score. The **Model selection and study compatibility** section shows all candidate probabilities, estimated losses and the selected architecture.

The selector cannot see the MRI's unknown label, does not retrain during upload, does not choose the model with the largest abnormality score and does not average the candidates as an ensemble. The estimated loss is a selection aid, not a probability that the selected model is correct. The principal-factor table explains the selector estimate only; it does not name anatomy or a lesion.

## Grad-CAM attention

Choose a finding, plane and sampled slice to view the original MRI slice and attention overlay. Warm colours show image regions that contributed positively to the selected model score. The attention map is coarse, enlarged for display and limited to that selected slice. It is not a confirmed tear, named anatomy, detection box, segmentation mask or treatment recommendation.

If a heatmap is unavailable for a particular slice, choose another sampled slice. A score can still be displayed when no concentrated positive attention is available for that view.

## Export and clear

The export button downloads a text or JSON research summary with the case reference, scores, selected models, thresholds and notices. **Clear current case** removes uploaded files and results from the browser session.

## Interpretation wording

Use: “This is a study-level research score produced by the listed prevalidated model. Grad-CAM explains model attention for this slice; it does not confirm a lesion. Clinical review remains required.”

Do not use: “The system diagnosed a tear,” “the heatmap proves the lesion location,” or “the result recommends treatment.”
