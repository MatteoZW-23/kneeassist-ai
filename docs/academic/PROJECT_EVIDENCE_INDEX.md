# KneeAssist XAI Project Evidence Index

This index links each academic claim to an inspectable project artefact. It supports transparent marking, reproduction and defence discussion.

| Claim | Supporting artefact |
|---|---|
| Final academic report and five product-first objectives | `docs/academic/submission/Mabira_Mathew_R234371Y_XAI.docx`, Chapter 1 |
| Defence explanation and dashboard walkthrough | `docs/academic/submission/KneeAssist_XAI_Defence_Cheat_Sheet.docx` |
| Dataset roles and limitations | `README.md`, `docs/CLINICAL_VALIDATION_GAPS.md`, `docs/RSNA_ONLINE_TRAINING_STATUS.md` |
| Formal ResNet-18 versus EfficientNet-B0 comparison | `results/model_comparison.json`, `notebooks/01_Train_MRI_Models_ResNet18_and_EfficientNetB0.ipynb` |
| Active routing policy | `model_registry/active_models.json`, `runs/model_portfolio_v1/activation_decision.json`, `runs/model_portfolio_v1/selection_correction_20261008.json` |
| Current routed development metrics | `runs/model_portfolio_v1/final/metrics.json`, `models/MODEL_CARD.md` |
| Route weights and provenance | `runs/model_family_v2/models/` and checkpoint paths recorded in `model_registry/active_models.json` |
| Historical EfficientNet-B0 baseline checkpoint | `models/best_model.pth`, `models/checkpoint_manifest.json`, `runs/mri_finetune_02/READINESS_REPORT.md` |
| External reference evidence for historical baseline | `notebooks/05_Evaluate_ACL_on_External_KneeMRI.ipynb`, `notebooks/07_Evaluate_Meniscus_on_External_FastMRI.ipynb` |
| Inference, deterministic scoring and attention maps | `src/inference/routed_predictor.py`, `src/inference/predictor.py`, `src/evaluation/gradcam.py` |
| Streamlit interface and input safety | `app.py`, `tests/`, `notebooks/04_Test_Dashboard_and_Input_Safety.ipynb` |
| Current limitations and validation gaps | `docs/CLINICAL_VALIDATION_GAPS.md`, `docs/REMAINING_WORK.md` |

The project should be assessed from these reproducible artefacts and stated limits, rather than unsupported claims about clinical readiness.
