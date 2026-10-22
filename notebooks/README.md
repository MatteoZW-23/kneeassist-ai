# KneeAssist XAI notebooks

The current submission evidence is drawn from the saved MRNet split manifest, five-candidate portfolio report, active model registry, final routed predictions and automated test logs. These files are linked in `../docs/academic/PROJECT_EVIDENCE_INDEX.md`.

For reproducible checks, use the launchers in the project root:

- **Show Objectives 4 and 5 Evidence.cmd** — verifies the fixed split, the five candidate rankings, registered checkpoints and the saved 120-study routed metrics without changing a model.
- **Run Tests.cmd** — runs the application and model regression tests.

`00_Setup_and_Dataset_Checks.ipynb` is retained as a simple MRNet intake and split-audit notebook. The current candidate comparison and routing evidence are intentionally read from the versioned JSON artifacts rather than being duplicated in a notebook.

Notebooks from exploratory work are not listed here because they are not evidence for the current submission model.
