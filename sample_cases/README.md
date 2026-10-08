# KneeAssist XAI three-plane research examples

These packaged examples are available only in this local project for the KneeAssist XAI defence demonstration. They contain one complete MRI slice stack per plane:

```text
axial.npy
coronal.npy
sagittal.npy
```

| File | Dataset reference | Reference labels | Recommended use |
|---|---|---|---|
| `mrnet_1130.zip` | MRNet official-validation study 1130 | Abnormal: 0, ACL: 0, meniscus: 0 | Normal-reference demonstration |
| `mrnet_1229_abnormal_research_example.zip` | MRNet official-validation study 1229 | Abnormal: 1, ACL: 1, meniscus: 1 | Abnormal-reference demonstration |

## Defence demonstration

1. Select **Load research example** to demonstrate a three-plane normal-reference study.
2. Select **Analyse study** and show the intake table confirms all three planes.
3. Open the slice viewer and Grad-CAM explanation. State that Grad-CAM is model attention, not a confirmed lesion.
4. Clear the case.
5. Select **Load abnormal research example** and repeat the workflow.
6. Present the output as research decision support against MRNet study-level reference labels. Do not describe either result as a clinical diagnosis.

The examples are not patient data and do not establish clinical performance. They must not be redistributed outside the dataset agreement or used to claim clinical readiness.

