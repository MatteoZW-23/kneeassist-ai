# KneeAssist AI Project Evidence Index

This index links each academic claim to a project artefact. It is intended to support transparent marking, reproduction and discussion.

| Claim | Supporting artefact |
|---|---|
| Five objectives and scope | `docs/academic/KneeAssist_AI_Five_Objective_Project_Report.docx`, Chapter 1 |
| Dataset roles and limitations | `README.md`, `docs/ACTIVE_TRAINING_PLAN.md`, `docs/RSNA_ONLINE_TRAINING_STATUS.md` |
| Training configuration | `config.yaml`, `src/training/train.py` |
| ResNet-18 and EfficientNet-B0 comparison | `results/model_comparison.json`, `notebooks/01_Train_MRI_Models_ResNet18_and_EfficientNetB0.ipynb` |
| Selected checkpoint | `runs/mri_finetune_02/models/best_model.pth` and its SHA256 in `runs/mri_finetune_02/READINESS_REPORT.md` |
| Measured metrics and charts | `runs/mri_finetune_02/results`, `runs/mri_finetune_02/validation_summary.json` |
| External reference evaluation | `notebooks/05_Evaluate_ACL_on_External_KneeMRI.ipynb`, `notebooks/07_Evaluate_Meniscus_on_External_FastMRI.ipynb` |
| Inference and attention maps | `src/inference/predictor.py`, `src/evaluation/gradcam.py`, `notebooks/03_Predict_MRI_Studies_and_View_GradCAM.ipynb` |
| Streamlit interface | `app.py`, `notebooks/04_Test_Dashboard_and_Input_Safety.ipynb` |
| Automated checks | `tests/`; rerun `python -m pytest tests -q` from the project environment |
| Current limitations and future validation | `runs/mri_finetune_02/READINESS_REPORT.md`, `docs/CLINICAL_VALIDATION_GAPS.md` |

The project should be assessed from these reproducible artefacts and the stated evidence, rather than from unsupported claims about clinical readiness.
