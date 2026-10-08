# KneeAssist XAI dashboard user guide

## Before you analyse a study

1. Launch the application with `Launch KneeAssist AI.cmd`.
2. Upload one supported knee MRI study only. Supported formats are MRI NumPy stacks, NIfTI volumes, DICOM slices and structured study ZIP files.
3. Ensure that all uploaded sequences belong to the same case. The app cannot prove patient identity or confirm that a ZIP does not mix patients.
4. Enter a pseudonymous case reference if needed. It is a label for the browser session, not a diagnosis or patient identifier.

JPEG/PNG screenshots, photographs, X-rays and internet images are intentionally rejected. A metadata-free NPY stack needs manual axial, coronal or sagittal plane confirmation.

## What the intake panel does

The panel reports file type, shape, slice count, dimensions, numeric range and plane evidence. It checks whether a compatible registered model exists. It does not determine whether an image is clinically adequate or whether it contains a knee.

For a presentation, use one axial, one coronal and one sagittal stack from the same MRI study. The app can process fewer planes, but displays a caution because missing-plane performance has not been separately validated.

## How to read the result screen

The score cards are study-level research scores. They are not confirmed diagnoses.

Above the cards, **Models used for this study** lists the route used for each output:

- General Abnormality → DenseNet-121
- ACL Tear → ResNet-18
- Meniscal Tear → Swin Transformer (Swin-T)

Each card repeats **Model used** directly below its score. The **Validated model selection and study compatibility** section shows the registered architecture, validation evidence and compatibility information.

The routing is determined from the saved registry before analysis. The app does not compare patient scores and then choose a model that produced the largest score. It also does not average the three models into an ensemble.

## Grad-CAM attention

Choose a finding, plane and sampled slice to view the original MRI slice and attention overlay. Warm colours show image regions that contributed positively to the selected model score. The attention map is coarse, enlarged for display and limited to that selected slice. It is not a confirmed tear, named anatomy, detection box, segmentation mask or treatment recommendation.

If a heatmap is unavailable for a particular slice, choose another sampled slice. A score can still be displayed when no concentrated positive attention is available for that view.

## Export and clear

The export button downloads a text or JSON research summary with the case reference, scores, selected models, thresholds and notices. **Clear current case** removes uploaded files and results from the browser session.

## Demonstration wording

Use: “This is a study-level research score produced by the listed prevalidated model. Grad-CAM explains model attention for this slice; it does not confirm a lesion. Clinical review remains required.”

Do not use: “The system diagnosed a tear,” “the heatmap proves the lesion location,” or “the result recommends treatment.”
