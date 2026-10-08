# Reproducibility and verification

## What is included

- `config.yaml`: active project configuration.
- `model_registry/active_models.json`: approved research model registry and selection rule.
- `models/best_model.pth`: preserved historical EfficientNet-B0 baseline checkpoint.
- `runs/model_family_v2/models/`: local DenseNet-121, ResNet-18 and Swin-T route checkpoints required by the active registry; these weights are intentionally excluded from Git.
- `data/mrnet_manifest.csv` and `runs/mri_finetune_02/splits.json`: study manifest and fixed split.
- `runs/mri_finetune_02/results/final/`: original single-model development-validation metrics, curves and prediction table.
- `runs/model_portfolio_v1/final/`: active target-routed development evaluation using tuning-selected routes and the previously used MRNet official validation cohort.
- `scripts/generate_error_analysis.py`: deterministic development error-analysis queue generator.
- `scripts/evaluate_model_portfolio.py`: separate candidate-calibration and routing comparison. `scripts/promote_routed_portfolio.py` creates the reviewed local routing registry only after fixed-split evidence is available.

## Verify core artifacts

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_gradcam_explanation.py tests\test_saved_inference.py tests\test_app.py -q
.\.venv\Scripts\python.exe scripts\generate_error_analysis.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Expected boundary

All active and historical checkpoints were trained on MRNet only. The official MRNet validation cohort was previously used during development and is not a new independent test. External KneeMRI and fastMRI results remain separate exploratory/reference evaluations. The prototype is not validated for clinical diagnosis or treatment recommendation.


