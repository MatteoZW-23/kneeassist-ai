"""Summarise active routed evaluation predictions without changing models or routes."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / 'runs' / 'model_portfolio_v1' / 'final' / 'predictions.csv'
OUT = ROOT / 'runs' / 'model_portfolio_v1' / 'final' / 'statistical_summary.json'
TARGETS = ('abnormal', 'acl', 'meniscus')

with PREDICTIONS.open(newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
labels = {target: np.asarray([int(row[f'{target}_label']) for row in rows]) for target in TARGETS}
scores = {target: np.asarray([float(row[f'{target}_probability']) for row in rows]) for target in TARGETS}
rng = np.random.default_rng(42)
intervals = {}
for target in TARGETS:
    values = []
    while len(values) < 1000:
        indices = rng.integers(0, len(rows), len(rows))
        sampled = labels[target][indices]
        if sampled.min() == sampled.max():
            continue
        values.append(float(roc_auc_score(sampled, scores[target][indices])))
    intervals[target] = {
        'point_auroc': float(roc_auc_score(labels[target], scores[target])),
        'bootstrap_95_percent_interval': [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        'brier_score': float(brier_score_loss(labels[target], scores[target])),
    }
macro = np.asarray([intervals[target]['point_auroc'] for target in TARGETS])
summary = {
    'dataset': 'MRNet-v1.0',
    'partition': 'official_valid',
    'rows': len(rows),
    'method': '1000 study-level nonparametric bootstrap resamples; seed 42; percentile interval for AUROC only.',
    'limitations': 'This is development evidence on an official-validation cohort with prior development exposure. It is not independent clinical validation or a statistical model-superiority test.',
    'targets': intervals,
    'macro_auroc': float(macro.mean()),
}
OUT.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
