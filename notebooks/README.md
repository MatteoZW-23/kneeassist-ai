# KneeAssist XAI notebooks

Run with the KneeAssist XAI Jupyter kernel. The notebooks preserve the formal ResNet-18 versus EfficientNet-B0 baseline experiment; the active dashboard routing is documented in `../model_registry/active_models.json` and `../models/MODEL_CARD.md`. Notebook 08 records the completed experiment and its limitations.

- [00 Setup and Dataset Checks](00_Setup_and_Dataset_Checks.ipynb)
- [01 Train MRI Models ResNet18 and EfficientNetB0](01_Train_MRI_Models_ResNet18_and_EfficientNetB0.ipynb)
- [02 Compare Models and Validate on MRNet](02_Compare_Models_and_Validate_on_MRNet.ipynb)
- [03 Predict MRI Studies and View GradCAM](03_Predict_MRI_Studies_and_View_GradCAM.ipynb)
- [04 Test Dashboard and Input Safety](04_Test_Dashboard_and_Input_Safety.ipynb)
- [05 Evaluate ACL on External KneeMRI](05_Evaluate_ACL_on_External_KneeMRI.ipynb)
- [06 Check FastMRI Images and Labels](06_Check_FastMRI_Images_and_Labels.ipynb)
- [07 Evaluate Meniscus on External FastMRI](07_Evaluate_Meniscus_on_External_FastMRI.ipynb)
- [08 MRI Fine Tuning Calibration and Run Status](08_MRI_Fine_Tuning_Calibration_and_Run_Status.ipynb)
