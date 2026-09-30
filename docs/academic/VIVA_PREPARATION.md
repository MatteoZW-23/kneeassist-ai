# KneeAssist AI Viva Preparation

Use this guide to prepare for a discussion of the project. Do not memorise answers word for word. Read the linked files, run the application yourself, and explain only the parts you understand. If asked about your personal contribution or use of tools, answer accurately and follow your institution's rules.

## One minute project explanation

KneeAssist AI is a local research application for analysing a complete knee MRI study. Its primary academic task is to classify the study as normal or abnormal. ACL tear and meniscal tear scores are secondary research outputs, not the main title claim. The project uses MRNet because it provides full MRI studies and labels for these targets. It compares ResNet-18 and EfficientNet-B0, selects a saved checkpoint using the planned internal AUROC rule, and provides a Streamlit interface to load a supported MRI study, show scores, display MRI slices and generate Grad-CAM attention maps. The project is a research prototype. Its external results are limited, so it is not presented as a clinical diagnostic system.

## The five objectives in simple language

| Objective | What was done | Evidence to open during a viva |
|---|---|---|
| 1. Prepare MRI data | MRI studies were checked, normalised by sequence and split at study level. Exact-file checks were used to reduce leakage risk. | `notebooks/00_Setup_and_Dataset_Checks.ipynb`, `data/mrnet_manifest.csv` |
| 2. Compare models | ResNet-18 and EfficientNet-B0 were trained as study-level models. | `notebooks/01_Train_MRI_Models_ResNet18_and_EfficientNetB0.ipynb`, `results/model_comparison.json` |
| 3. Evaluate the selected model | AUROC, F1, sensitivity, specificity, precision, accuracy, curves and confusion matrices were reported. | `notebooks/02_Compare_Models_and_Validate_on_MRNet.ipynb`, `runs/mri_finetune_02/validation_summary.json` |
| 4. Build independent inference | A saved checkpoint can load a new supported study, give scores and create Grad-CAM output. | `notebooks/03_Predict_MRI_Studies_and_View_GradCAM.ipynb`, `src/inference/predictor.py` |
| 5. Build the interface | The local Streamlit dashboard supports upload, review, explanation, text/JSON export and clearing the case. | `app.py`, `notebooks/04_Test_Dashboard_and_Input_Safety.ipynb` |

## Questions an examiner may ask

### What data trained the deployed model?

MRNet only. The active model was not trained on KneeMRI, fastMRI, X-ray archives or the Kaggle RSNA pilot. This prevents treating differently labelled datasets as if they were directly interchangeable.

### Why did you not use every dataset?

More files do not always mean better training data. The labels, image format, modality and study unit must match the task. KneeMRI was useful for an ACL reference evaluation. fastMRI-plus had limited meniscus references. The RSNA pilot had only 58 completely labelled studies. Combining them without a clear label review process would make the results less reliable.

### What happened with the Kaggle RSNA dataset?

It was examined and trained online in a private Kaggle notebook to save local storage. There were 4,407 studies, but only 58 had complete explicit labels for all 12 available conditions. The pilot was exploratory, with macro AUROC around 0.658 for both candidate encoders. It was not used to replace the local MRNet model. See `docs/RSNA_ONLINE_TRAINING_STATUS.md`.

### What is the final model performance?

On the 120-study MRNet official validation cohort, macro AUROC was 0.827. Per-target AUROC was 0.901 for general abnormality, 0.825 for ACL tear and 0.754 for meniscal tear. These are development results, not proof of clinical performance. The full values are in `runs/mri_finetune_02/READINESS_REPORT.md`.

### Why are external results important?

External evaluation checks whether behaviour transfers beyond the training source. The project obtained ACL AUROC of 0.603 on KneeMRI and meniscus AUROC of 0.656 on fastMRI reference annotations. These weaker results show why internal performance alone is not enough.

### How did you reduce data leakage?

The project treats a complete MRI study as the unit, so slices from the same study are not placed in different splits. It records data roles and uses exact-file checks. It cannot prove patient-level independence because MRNet does not supply usable patient linkage for this project.

### Why does the app reject normal pictures and screenshots?

The model was trained on MRI slice stacks, not photographs, internet screenshots or X-rays. Accepting those inputs would create a misleading impression that the output is meaningful. The app therefore rejects JPEG and PNG images.

### What does Grad-CAM show?

Grad-CAM is a visualisation of model attention for a selected output and MRI sequence. It is useful for inspecting the computation, but it does not confirm that a highlighted region is an anatomical lesion.

### Does the dashboard accept DICOM?

It accepts a guarded DICOM ZIP format: one readable, uncompressed DICOM series in each plane-labelled folder. This is an input convenience feature. It does not prove that the model works reliably across hospital scanners or that the archive contains one patient and the correct anatomy.

### What is the main limitation?

The lack of a new, independent, clinically reviewed multi-site dataset. The next stage is not simply more training. It requires qualified labels, a locked evaluation plan and clinical review before any clinical-use claim could be considered.

## Demonstration sequence

1. Open `Launch KneeAssist AI.cmd`.
2. Select **Load research example**.
3. Select **Analyse study**.
4. Explain the three scores as model probabilities, not diagnoses.
5. Open the MRI slice viewer and show the three planes.
6. Open model attention and explain the Grad-CAM warning.
7. Export the JSON summary and clear the current case.
8. Open `runs/mri_finetune_02/READINESS_REPORT.md` and point out both the internal results and external limitations.

## Before presenting

- Run the dashboard and the packaged example yourself.
- Read the five objectives and explain one project file for each.
- Check that you can explain AUROC, sensitivity, specificity and a confusion matrix in your own words.
- Do not claim that the model diagnoses patients, replaces radiologists or has been validated in Zimbabwe.
- Follow your institution's rules on acknowledgement and use of tools.
