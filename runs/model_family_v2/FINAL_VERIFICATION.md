> **Historical baseline-verification note (8 October 2026):** this verification concerns the former single EfficientNet-B0 deployment. The active dashboard now uses the fixed target-routing registry in `../../model_registry/active_models.json`.

# Final software verification — 30 September 2026

The completed model-family benchmark did not change the recorded baseline deployment. The historical calibrated EfficientNet-B0 checkpoint remains `runs/mri_finetune_02/models/best_model.pth`.

## Checks completed after the benchmark

- Full automated suite: **33 passed** in 12.96 seconds after the guarded ensemble and localisation additions (`.venv\\Scripts\\python.exe -m pytest tests -q`).
- Independent command-line inference: the packaged three-plane MRI study `sample_cases/mrnet_1130.zip` loaded and produced three findings using the recorded baseline checkpoint.
- Streamlit health endpoint: `200 OK`.
- Browser workflow: the packaged research example loaded and analysed; probabilities, compatible-model evidence, MRI viewer and Grad-CAM attention rendered.
- Export controls: text and structured-JSON download controls rendered; the automated app test asserts both controls are present.
- Clear-case workflow: clearing the packaged example removed the previous case and findings.

## Limitations retained

This verification confirms software behavior, not diagnostic accuracy or clinical suitability. The historical baseline model remains trained on MRNet only; its probabilities are calibrated only on a small reserved MRNet development subset. The new benchmark did not introduce patient linkage, independent clinical validation, lesion segmentation or diagnostic clearance.


