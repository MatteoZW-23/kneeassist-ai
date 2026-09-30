from pathlib import Path
import nbformat
from src.utils import ROOT

for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        source=cell.source
        source=source.replace('ROOT / "data/splits.json"','ROOT / cfg["dataset"]["splits"]')
        source=source.replace('ROOT / "results/leakage_audit.json"','ROOT / cfg["evaluation"]["directory"] / "leakage_audit.json"')
        source=source.replace('ROOT / "results/final"','ROOT / cfg["evaluation"]["directory"] / "final"')
        source=source.replace('ROOT / "results/REPORT.md"','ROOT / cfg["evaluation"]["directory"] / "REPORT.md"')
        source=source.replace("out=ROOT/'runs/mrnet_fresh_01/results/external_fastmri'",
             "import sys\nsys.path.insert(0,str(ROOT))\nfrom src.utils import config\ncfg=config(ROOT/'config.yaml')\nout=ROOT/cfg['evaluation']['directory']/'external_fastmri'")
        if path.name.startswith('01_') and 'completed = subprocess.run' in source:
            source='''# Viewing a notebook never starts a second training process.
print("Frozen baseline resume: .venv/Scripts/python.exe -m src.training.train --config configs/mrnet_fresh_01.yaml --resume")
print("Improvement pipeline: .venv/Scripts/python.exe -m scripts.complete_improvement")
print("Check notebook 08 for the active improvement process before launching anything.")'''
        if path.name.startswith('01_') and 'resume_check =' in source:
            source='''print("Active checkpoint:", ROOT / cfg["evaluation"]["checkpoint"])
print("Existence:", (ROOT / cfg["evaluation"]["checkpoint"]).is_file())
print("Historical resume diagnostics are under results/diagnostics; they do not verify the current fine-tuning run.")'''
        if source!=cell.source:
            cell.source=source;cell.outputs=[];cell.execution_count=None
    nbformat.validate(nb);nbformat.write(nb,path)

path=ROOT/'notebooks/08_MRI_Fine_Tuning_Calibration_and_Run_Status.ipynb'
if not path.exists():
    nb=nbformat.v4.new_notebook()
    nb.metadata.kernelspec={'display_name':'KneeAssist AI','language':'python','name':'kneeassist-ai'}
    nb.cells=[nbformat.v4.new_markdown_cell('''# MRI fine-tuning, calibration and run status

This notebook inspects the versioned improvement experiment without starting training.
MRNet supplies training labels. KneeMRI and fastMRI remain external reference evaluations.
Study separation does not prove patient separation. Previous evaluation cohorts are not new untouched tests.
The 100 reserved calibration cases were removed from the candidate fitting set; the incumbent cannot be calibrated on them because it previously trained on them.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import json
from IPython.display import display
ROOT=Path.cwd()
if not (ROOT/'config.yaml').exists(): ROOT=ROOT.parent
run=ROOT/'runs/mri_finetune_02'
for filename in ['protocol.json','completion_status.json','progress.json']:
    path=run/filename
    if path.exists():
        print(filename);display(json.loads(path.read_text()))'''),
    nbformat.v4.new_code_cell('''for filename in ['results/model_comparison.json','results/calibration.json','validation_summary.json','verification.json']:
    path=run/filename
    if path.exists():
        print(filename);display(json.loads(path.read_text()))
    else:print(filename, 'not produced yet; no final result claimed')'''),
    nbformat.v4.new_markdown_cell('''Selection uses internal tuning macro AUROC only. Calibration and a balanced sensitivity/specificity threshold use the reserved calibration partition. External scores never trigger tuning or model selection. A candidate that does not beat the incumbent is retained as an experiment, and the incumbent remains selected. This experiment uses one seed and is not proof of clinical readiness.''')]
    nbformat.validate(nb);nbformat.write(nb,path)

index=['# KneeAssist AI notebooks','','Run with the KneeAssist AI Jupyter kernel. Notebooks read the active deployment config; notebook 08 tracks the pending experiment.','']
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    index.append(f'- [{path.stem.replace("_"," ")}]({path.name})')
(ROOT/'notebooks/README.md').write_text('\n'.join(index),encoding='utf-8')
