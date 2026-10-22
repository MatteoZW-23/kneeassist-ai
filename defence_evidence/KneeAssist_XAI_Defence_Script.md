# KneeAssist XAI: Defence Script and Technical Notes

**Project:** KneeAssist XAI – explainable, study-level knee MRI classification research prototype  
**Scope:** MRNet-v1.0 research data only. This is not a diagnostic device and does not prescribe treatment.  
**Prepared from:** the active application, `runs/dynamic_selection_v1` evidence, and a live local execution on 22 October 2026.

---

## 1. One-minute opening statement

Good morning. My project is KneeAssist XAI, an explainable deep-learning research prototype for classifying complete knee MRI studies. Rather than looking at one screenshot or one slice, it accepts a complete study with axial, coronal and sagittal MRI sequences. It checks the uploaded data, standardises the images, samples slices, runs five trained transfer-learning models, and uses a learned dynamic selector to choose one model for that particular study. The selected model then returns study-level research scores for general abnormality, ACL tear and meniscal tear. The system can also generate a Grad-CAM attention overlay so that the user can see the image regions that contributed positively to a selected score.

The main contribution is not a clinical diagnosis claim. It is a transparent workflow showing how different architectures can be compared and selected from saved validation-supported evidence before an MRI result is shown.

---

## 2. The problem being addressed

Knee MRI contains many slices and can be reviewed in different anatomical planes. A junior doctor or researcher may need a structured way to review a study and see a reproducible AI output. A single fixed model may not be equally suitable for every MRI study. Therefore, the system compares five candidate classifiers during analysis and uses a learned selector to choose the model expected to have the lowest error for that particular study.

The project supports review. It does **not** replace the radiologist or clinician. It cannot diagnose every knee condition, prescribe treatment, rule out injury, or establish clinical confidence.

---

## 3. Objectives and where they appear in the system

| Objective | Evidence to show during defence |
|---|---|
| 1. Develop a Streamlit prototype for knee MRI classification, prediction visualisation and explainable AI. | Open the local dashboard, show upload, results, model information, Grad-CAM and research-summary export. |
| 2. Design an automated MRI preprocessing pipeline for file-format detection, data validation, normalisation and slice sampling. | Explain the intake table, plane confirmation, 1st–99th percentile normalisation, 224 × 224 resizing and 12 sampled slices per plane. |
| 3. Develop five transfer-learning models for study-level classification of general abnormality, ACL tears and meniscal tears. | Show the five architectures and saved checkpoints in the model evaluation page. |
| 4. Implement a dynamic model-selection mechanism that compares five candidate models for each uploaded MRI study using predefined, validation-supported selection criteria. | Show that all five models run, 90 features are calculated, five Gradient Boosting competence estimators predict expected mean Brier loss, and one architecture is selected for the whole study. |
| 5. Evaluate the classification performance of the developed system using accuracy, sensitivity, specificity, precision, F1-score and AUROC. | Show the saved 120-study MRNet development evaluation table, confusion matrices and limitations. |

---

## 4. Key terms to explain clearly

| Term | Plain explanation |
|---|---|
| MRI | Magnetic Resonance Imaging. It produces a three-dimensional study made from many two-dimensional slices. |
| Study-level classification | The system combines information from multiple slices and all three supplied planes, then gives one score per target for the whole examination. It does not classify one ordinary photograph. |
| Axial plane | A cross-sectional view through the body from top to bottom. In a knee study it shows a horizontal-type slice through the joint. |
| Coronal plane | A front-facing view that divides the body into front and back portions. |
| Sagittal plane | A side view that divides the body into left and right portions. This view is useful for following structures such as the ACL. |
| ACL | Anterior Cruciate Ligament. It is a stabilising ligament inside the knee. The model estimates an ACL-tear score; it does not confirm a tear. |
| Meniscus | A cartilage structure that cushions and stabilises the knee joint. The model estimates a meniscal-tear score; it does not state which part or grade of tear. |
| General abnormality | The MRNet reference label indicating that the study was labelled abnormal for at least one relevant condition. It is broader than the ACL and meniscus outputs. |
| Transfer learning | Starting an image encoder from weights learned on a large image dataset, then training it for the knee-MRI task. |
| Calibration | Adjusting or checking whether a model score behaves like a probability on reserved development data. Calibration here is not clinical calibration. |
| Brier loss | A probability-error measure. Lower is better. It compares predicted probabilities with known labels during development; the true label is unavailable for a newly uploaded case. |
| AUROC | Area Under the Receiver Operating Characteristic curve. It measures how well scores rank positive above negative cases over all possible thresholds. |
| Sensitivity | Among labelled positive studies, the proportion flagged positive. |
| Specificity | Among labelled negative studies, the proportion correctly not flagged. |
| Precision | Among studies flagged positive, the proportion labelled positive. |
| F1-score | Harmonic balance of precision and sensitivity. |
| Grad-CAM | A gradient-based attention map showing image regions that contributed positively to a selected classifier score. It is not a diagnosis, lesion boundary, detection box or segmentation mask. |

---

## 5. System architecture: say this while pointing to the workflow page

> A complete three-plane MRI study enters the system. The intake layer checks the format, dimensions, numeric content and plane information. The preprocessing layer normalises each sequence, resizes images to 224 by 224 pixels, and samples 12 slices from each plane. The same prepared study goes to all five base models. Their three scores, entropy, disagreement and image-acquisition features form a 90-feature vector. Five learned competence estimators predict the expected mean Brier loss of each candidate. The architecture with the lowest estimated loss is selected for the entire study and supplies all three displayed scores. A Grad-CAM overlay can then be requested for a selected target and plane.

### The five base classifiers

1. ResNet-18
2. ResNet-50
3. DenseNet-121
4. EfficientNet-B0
5. Swin Transformer, also called Swin-T

All five were evaluated using the same saved MRNet split, target labels and preprocessing rules. The system does not train these models when a user uploads a study.

---

## 6. Data and split explanation

The active system was developed using MRNet-v1.0. The three labels are general abnormality, ACL tear and meniscal tear. The saved split assigns 1,250 examinations to fixed roles:

| Role | Studies | Purpose |
|---|---:|---|
| Fit | 704 | Base-model fitting |
| Tune | 100 | Candidate early stopping and tuning decisions |
| Calibration | 100 | Reserved probability-calibration stage |
| Selector training | 226 | Train the competence estimators that predict expected Brier loss |
| Recorded final evaluation | 120 | Score the five classifiers and the dynamic selector after selector fitting |

The run recorded zero study overlap and zero exact-series overlap across those roles. However, patient IDs are unavailable in this local MRNet copy, so patient-level independence cannot be proved. The 120-study evaluation also had prior development exposure. For those reasons, it is honest to call it a **recorded internal development evaluation**, not independent external or clinical validation.

---

## 7. Preprocessing: explain every stage

1. **File understanding:** the system recognises NumPy MRI stacks, ZIP studies, DICOM, DICOM ZIP and NIfTI where the local dependencies are available. JPEG and PNG pictures are rejected because an internet image or screenshot is not a validated MRI volume.
2. **Input checking:** each sequence must be a three-dimensional numeric array: slices × height × width. It must contain finite values, usable dimensions and intensity variation.
3. **Plane checking:** metadata is preferred. If a processed array has no trustworthy orientation metadata, the user confirms axial, coronal or sagittal in the dashboard. A filename alone is not accepted as proof of the plane.
4. **Normalisation:** intensities are normalised using the 1st and 99th percentiles. This reduces variation in image intensity scale.
5. **Resizing:** sampled slices are resized to 224 × 224 pixels, the model input size.
6. **Slice sampling:** 12 representative slices are sampled from each available plane. For the active dynamic selector, a complete axial, coronal and sagittal study is required.
7. **Tensor preparation:** images are transformed into the three-channel tensors expected by the pretrained image encoders.

A missing plane does not mean the knee is normal. It means the dynamic selector is not validated for that incomplete input, so the system should stop rather than make an unsupported claim.

---

## 8. Dynamic model selection: the most important defence explanation

### Short answer

The selector does **not** look at the patient’s label, and it does not simply choose the model with the highest probability. It uses previously trained competence estimators to estimate which candidate is likely to have the lowest mean Brier loss for the current unlabelled study.

### Detailed answer

1. The same preprocessed MRI study is passed through all five candidate architectures.
2. Each candidate produces three calibrated scores: general abnormality, ACL tear and meniscal tear.
3. The application derives 90 features from those current scores, their entropy, inter-model disagreement and MRI acquisition properties.
4. There are five saved `GradientBoostingRegressor` competence estimators—one per candidate architecture.
5. Each estimator outputs an **estimated** mean Brier loss for its candidate.
6. The system ranks the five estimates and selects the architecture with the lowest estimated loss.
7. The selected architecture provides **all three** displayed study-level outputs. It does not select a different architecture for each target in the active dynamic mode.

This is called case-specific dynamic model selection. It is different from a fixed route, where one architecture is chosen once for all future studies.

### What not to say

- Do not say the selector knows the true answer for the uploaded patient.
- Do not say it selects the model with the highest score.
- Do not say it proved a statistically significant improvement over every fixed model.
- Do not call the estimated Brier loss a probability that the model is correct.

---

## 9. Recorded development results: safe wording

The active five-model dynamic selector was evaluated on a recorded 120-study MRNet development cohort. Its macro results were:

| Measure | Dynamic selector result |
|---|---:|
| AUROC | 0.8472 |
| Average precision / PR-AUC measure | 0.8403 |
| Accuracy | 0.7444 |
| Sensitivity | 0.8482 |
| Specificity | 0.5495 |
| Precision | 0.6965 |
| F1-score | 0.7641 |
| Mean Brier loss | 0.1596 |
| ECE, 10 equal-width bins | 0.1010 |

On that cohort, the selector chose ResNet-18 for 25 studies, ResNet-50 for 23, DenseNet-121 for 12, EfficientNet-B0 for 45 and Swin-T for 15. This shows that the selector actually varied its choice across cases.

Against the fixed validation-selected DenseNet-121 reference, the selector’s macro AUROC was 0.8472 versus 0.8444: a difference of +0.0028. The paired bootstrap 95% interval was −0.0215 to +0.0277, which crosses zero. Therefore the correct conclusion is:

> The dynamic selection mechanism operated and changed architectures by study, but this recorded cohort does not establish an independent or statistically conclusive performance advantage over the fixed reference.

For mean Brier loss, the selector recorded 0.1596 versus 0.1640 for fixed DenseNet-121, a difference of −0.0044. The paired 95% interval was −0.0146 to +0.0058, which also crosses zero. The bootstrap used 2,000 study-level resamples with fixed seed 42 so that the calculation can be reproduced. The seed is not a performance improvement.

---

## 10. Live local execution record for the defence

On 22 October 2026, the active code was run against the included MRNet example archive `sample_cases/mrnet_1130.zip`.

- Intake succeeded with axial 25 × 256 × 256, coronal 25 × 256 × 256 and sagittal 27 × 256 × 256 stacks.
- All five candidate models executed.
- The five Gradient Boosting competence estimators used 90 selector features.
- **Selected architecture:** Swin-T (`dms_swin_t`), because it had the lowest estimated mean Brier loss for this study.
- Estimated losses: ResNet-18 0.1037; ResNet-50 0.0921; DenseNet-121 0.1031; EfficientNet-B0 0.1027; Swin-T 0.0847.
- Returned scores: general abnormality 46.3%; ACL tear 6.5%; meniscal tear 7.1%.
- The saved research thresholds were 55.1%, 11.8% and 25.5% respectively, so all three outputs were **below their research thresholds** for this example.
- Grad-CAM was also executed for the selected Swin-T model and returned 12 axial 224 × 224 attention overlays.

Use this safe explanation:

> These are research scores for the included example, not a diagnosis. Below threshold does not rule out injury. The result demonstrates the complete software workflow: three-plane intake, five candidate inferences, study-specific model selection, score display and attention generation.

---

## 11. Demonstration sequence

1. Start the local app: double-click `Launch KneeAssist AI.cmd` in the project folder.
2. Open `http://127.0.0.1:8501` if it does not open automatically.
3. Begin on **Research workflow** and explain the offline development stages versus local analysis.
4. Open **Model evaluation and selection** to show the five candidates, fixed data roles, metrics and selection-frequency evidence.
5. Return to **Clinical MRI review**.
6. Select **Load research example** and choose the complete three-plane MRNet ZIP.
7. Show the intake table: file type, dimensions and recognized plane(s).
8. Click **Analyse study**. Say: “The system now runs all five candidate models. It is not retraining.”
9. Point out the selected architecture and the estimated expected-loss ranking.
10. Explain each of the three scores and its research threshold.
11. Request Grad-CAM for one target and plane. State the attention limitation before discussing the map.
12. Open the exported research summary only if asked. End by showing the research-only limitation notice.

---

## 12. Questions an examiner may ask, with answers

### Why did you use three planes?

Each plane provides a different view of the same knee. Combining axial, coronal and sagittal sequences supports a study-level rather than single-image assessment. The dynamic selector was evaluated only on complete three-plane MRNet studies.

### Why do you need five models if one model can classify an MRI?

Different architectures can respond differently to image characteristics and label patterns. The project tests whether a learned case-specific selection rule can choose a suitable candidate for a new study rather than assuming one architecture is always best.

### Does the system choose a model for every injury separately?

No. In the active dynamic selection mode it chooses **one architecture for the whole examination**, then that architecture returns the three target scores. This avoids mixing outputs from different models without an evaluated combination rule.

### How does it choose that architecture without knowing the answer?

It cannot know the answer. It uses features from the models’ current outputs and the image acquisition to estimate each model’s expected mean Brier loss. It chooses the lowest estimate. The true label is used only offline during development to train and evaluate the selector.

### What is the difference between a probability and confidence?

The displayed value is a calibrated model score from development data. It is not a clinical probability that the model is correct and not a guarantee of injury. Distance from a threshold is not reliability.

### Why are results below 50% sometimes not called normal?

The threshold is selected during the reserved development procedure; it need not be 50%. More importantly, a below-threshold score does not rule out injury, particularly in an unvalidated clinical setting.

### What does Grad-CAM show?

It shows which image regions positively contributed to the selected classifier score. It is useful for transparency and review, but it does not confirm a lesion, identify its exact location, draw a bounding box or produce a segmentation mask.

### Why is there no detection box or segmentation mask?

Those features require trustworthy lesion-location annotations or pixel-level masks and separate validated detection or segmentation models. The MRNet classification labels used here do not provide that ground truth.

### Why not use an online knee photograph or JPEG?

The model was trained for structured knee MRI volumes, not photographs, screenshots or isolated web images. Accepting those would be unsupported and could produce misleading results.

### What does seed 42 do?

It fixes random operations, particularly the recorded bootstrap resampling, so another person can reproduce the same calculation. It does not make the AI more accurate.

### Is this ready for hospital use?

No. It is a local research decision-support prototype. It has MRNet-only development evidence, no verified patient-level independence in this copy, prior development exposure of the recorded 120-study cohort, and no independent external clinical validation.

### What would you do next?

Obtain ethically approved, independently labelled multi-site MRI data; verify patient-level splits; externally validate calibration and performance; acquire lesion-level annotations for detection or segmentation; assess prospective clinical workflow and safety with radiologists.

---

## 13. Final closing statement

KneeAssist XAI demonstrates a complete reproducible research workflow: structured MRI intake, fixed preprocessing, five trained candidate classifiers, case-specific dynamic model selection, study-level score display and Grad-CAM attention. The evidence shows that the selector changes its selected architecture between studies, but its small apparent improvement over the fixed reference is exploratory and not clinically conclusive. The correct use is as a research prototype that supports review and future validation—not as an autonomous diagnostic system.

---

## 14. Exact evidence locations

- Active application: `C:\Users\HP\Desktop\KneeAssist-AI\app.py`
- Dynamic inference and selection: `C:\Users\HP\Desktop\KneeAssist-AI\src\inference\dynamic_predictor.py`
- Preprocessing and file validation: `C:\Users\HP\Desktop\KneeAssist-AI\src\data\preprocessing.py`
- Selection artifact: `C:\Users\HP\Desktop\KneeAssist-AI\runs\dynamic_selection_v1\selector\dynamic_selector.joblib`
- Model registry: `C:\Users\HP\Desktop\KneeAssist-AI\model_registry\dynamic_selection.json`
- Protocol and split evidence: `C:\Users\HP\Desktop\KneeAssist-AI\runs\dynamic_selection_v1\protocol.json`
- Recorded final metrics: `C:\Users\HP\Desktop\KneeAssist-AI\runs\dynamic_selection_v1\final_evaluation.json`
- Recorded predictions: `C:\Users\HP\Desktop\KneeAssist-AI\runs\dynamic_selection_v1\final_predictions.csv`
- Evaluation figures: `C:\Users\HP\Desktop\KneeAssist-AI\runs\dynamic_selection_v1\figures`
- Included demo data: `C:\Users\HP\Desktop\KneeAssist-AI\sample_cases\mrnet_1130.zip`
