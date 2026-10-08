# KneeAssist XAI Viva Preparation

Use this guide to prepare for a discussion of the project. Do not memorise answers word for word. Read the linked files, run the application yourself, and explain only the parts you understand. If asked about your personal contribution or use of tools, answer accurately and follow your institution's rules.

## One minute project explanation

KneeAssist XAI is a local explainable-AI research application for analysing a complete knee MRI study. Its primary academic task is to classify the study as normal or abnormal. ACL tear and meniscal tear scores are secondary research outputs, not the main title claim. The project uses MRNet because it provides full MRI studies and labels for these targets. It compares models using preserved validation evidence and provides a Streamlit interface to load a supported MRI study, show scores, display MRI slices, identify the selected model and generate Grad-CAM attention maps. The XAI features explain model scoring; they do not confirm or precisely locate a lesion. The project is a research prototype. Its external results are limited, so it is not presented as a clinical diagnostic system.

## Which model is used when I upload a study?

The formal academic comparison retained EfficientNet-B0 as the recorded baseline because it had the stronger validation AUROC in the required ResNet-18-versus-EfficientNet-B0 comparison. The live dashboard does not use that one baseline for every displayed finding. It reads the fixed registry before scoring the upload: DenseNet-121 produces the general-abnormality score, ResNet-18 produces the ACL score, and Swin-T produces the meniscal score. The available MRI planes are checked for compatibility. The system never chooses a model because it happened to give the highest score for that patient, and it does not average these models as an ensemble.

## The five objectives in simple language

| Objective | What was done | Evidence to open during a viva |
|---|---|---|
| 1. Develop the prototype | The Streamlit dashboard supports upload, review, explanation, text/JSON export and clearing the case. | `app.py`, `notebooks/04_Test_Dashboard_and_Input_Safety.ipynb` |
| 2. Prepare MRI studies | MRI studies were normalised by sequence, sampled by slice, checked for valid inputs and split at study level. | `notebooks/00_Setup_and_Dataset_Checks.ipynb`, `data/mrnet_manifest.csv` |
| 3. Develop the transfer-learning model | A study-level transfer-learning MRI classifier was implemented. | `src/models/study_model.py`, `src/training/train.py` |
| 4. Compare models | ResNet-18 and EfficientNet-B0 were compared using validation AUROC; EfficientNet-B0 is the recorded formal baseline. | `notebooks/01_Train_MRI_Models_ResNet18_and_EfficientNetB0.ipynb`, `results/model_comparison.json` |
| 5. Evaluate the selected model | AUROC, F1, sensitivity, specificity, precision, accuracy, curves and confusion matrices were reported against available reference labels. | `notebooks/02_Compare_Models_and_Validate_on_MRNet.ipynb`, `runs/mri_finetune_02/validation_summary.json` |

## Questions an examiner may ask

### What data trained the deployed model?

MRNet only. The active registered models were not trained on KneeMRI, fastMRI, X-ray archives or the Kaggle RSNA pilot. This prevents treating differently labelled datasets as if they were directly interchangeable.

### Why did you not use every dataset?

More files do not always mean better training data. The labels, image format, modality and study unit must match the task. KneeMRI was useful for an ACL reference evaluation. fastMRI-plus had limited meniscus references. The RSNA pilot had only 58 completely labelled studies. Combining them without a clear label review process would make the results less reliable.

### What happened with the Kaggle RSNA dataset?

It was examined and trained online in a private Kaggle notebook to save local storage. There were 4,407 studies, but only 58 had complete explicit labels for all 12 available conditions. The pilot was exploratory, with macro AUROC around 0.658 for both candidate encoders. It was not used to replace the local MRNet model. See `docs/RSNA_ONLINE_TRAINING_STATUS.md`.

### What is the final model performance?

For the formal EfficientNet-B0 baseline, macro AUROC was 0.827 on the previously used 120-study MRNet development-validation cohort. The active target-routed dashboard had macro AUROC 0.846 on that same reused cohort: 0.861 for general abnormality (DenseNet-121), 0.870 for ACL (ResNet-18) and 0.808 for meniscus (Swin-T). These are development results, not independent or clinical performance. See `models/MODEL_CARD.md` and `runs/model_portfolio_v1/final/metrics.json`.

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
6. Select a reported finding, click **Generate attention map**, then explain the Grad-CAM warning.
7. Export the JSON summary and clear the current case.
8. Open `model_registry/active_models.json`, `models/MODEL_CARD.md`, `runs/model_portfolio_v1/final/metrics.json` and `runs/mri_finetune_02/READINESS_REPORT.md` to distinguish the live routes, historical baseline and their limitations.

## Before presenting

- Run the dashboard and the packaged example yourself.
- Read the five objectives and explain one project file for each.
- Check that you can explain AUROC, sensitivity, specificity and a confusion matrix in your own words.
- Do not claim that the model diagnoses patients, replaces radiologists or has been validated in Zimbabwe.
- Follow your institution's rules on acknowledgement and use of tools.

