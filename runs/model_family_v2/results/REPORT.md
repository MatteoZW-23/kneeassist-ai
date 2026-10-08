> **Historical benchmark record.** This report describes the pre-routing model-family benchmark. The current live routing registry is `model_registry/active_models.json`.

# KneeAssist XAI — historical model-family evaluation

Selected architecture: **efficientnet_b0**. Selected checkpoint stage: frozen_encoder. Experiment mode: frozen_encoder_family_benchmark.

Selection used internal validation macro AUROC. Threshold procedure: internal_validation_f1. Official validation was not used for either decision.

| Model | Internal validation macro AUROC | Selected epoch |
|---|---:|---:|
| resnet18 | 0.839 | 6 |
| resnet50 | 0.828 | 2 |
| densenet121 | 0.835 | 4 |
| efficientnet_b0 | 0.847 | 4 |
| swin_t | 0.840 | 6 |

## Official validation (120 exams)

| Finding | AUROC | F1 | Sensitivity | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| General Abnormality | 0.907 | 0.900 | 1.000 | 0.160 | 0.819 | 0.825 |
| ACL Tear | 0.827 | 0.714 | 0.648 | 0.864 | 0.795 | 0.767 |
| Meniscal Tear | 0.766 | 0.709 | 0.865 | 0.559 | 0.600 | 0.692 |
| Macro average | 0.834 | 0.774 | 0.838 | 0.527 | 0.738 | 0.761 |

AUROC is a ranking metric, not percentage accuracy. The official validation cohort was used in the earlier prototype, so these are development-validation results, not a new external test. Patient independence is not verified without a patient mapping. Probabilities and score-separation indicators are not clinically calibrated. Attention maps are not confirmed lesions.

The current checkpoints support abnormality, ACL tear and meniscal tear only. RSNA and X-ray data were not used for these checkpoints. Research and clinical decision-support prototype; clinician review required.