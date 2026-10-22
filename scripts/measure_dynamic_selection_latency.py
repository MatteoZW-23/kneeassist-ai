"""Measure local all-five-model dynamic-selection inference after activation.

The timing includes sequential checkpoint loading and five model forwards because
that is the behaviour used by the local dashboard to bound GPU memory.  It is
not a claim of clinical throughput or a cloud deployment benchmark.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import read_manifest, volumes
from src.inference.dynamic_predictor import DynamicSelectionPredictor, dynamic_selection_status
from src.utils import ROOT, config, get_device


def main(repeats: int) -> None:
    if repeats < 1:
        raise ValueError('repeats must be at least 1')
    cfg = copy.deepcopy(config())
    cfg.setdefault('dynamic_selection', {})['enabled'] = True
    status = dynamic_selection_status(cfg)
    if not status['available']:
        raise RuntimeError('Cannot measure dynamic-selection latency: ' + status['reason'])
    split_path = ROOT / 'data' / 'splits' / 'dynamic_selection_v1.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    rows = {row['study_id']: row for row in read_manifest(cfg)}
    study = rows[split['official_valid'][0]]
    study_volumes = volumes(study, cfg)
    predictor = DynamicSelectionPredictor(cfg)

    # Warm-up is deliberately excluded from the reported repeat measurements.
    predictor.predict(study_volumes, 'latency-warmup')
    durations = []
    selected = []
    for index in range(repeats):
        start = time.perf_counter()
        result = predictor.predict(study_volumes, f'latency-{index}')
        durations.append(time.perf_counter() - start)
        selected.append(result['dynamic_selection']['selected_model_id'])

    output = {
        'measurement': 'local end-to-end all-five-model dynamic-selection inference, excluding one warm-up',
        'dataset': 'MRNet-v1.0',
        'partition_source': 'first official_valid study from dynamic_selection_v1 split',
        'repeat_count': repeats,
        'device': str(get_device(cfg)),
        'sequential_model_execution': True,
        'models_executed_per_repeat': 5,
        'seconds': {
            'mean': float(statistics.mean(durations)),
            'median': float(statistics.median(durations)),
            'minimum': float(min(durations)),
            'maximum': float(max(durations)),
        },
        'selected_models': selected,
        'limitation': 'This is local prototype timing on one MRNet study. It includes five sequential checkpoint loads and does not establish production or clinical throughput.',
    }
    output_path = ROOT / 'runs' / 'dynamic_selection_v1' / 'latency.json'
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    final_path = ROOT / 'runs' / 'dynamic_selection_v1' / 'final_evaluation.json'
    final = json.loads(final_path.read_text(encoding='utf-8'))
    final['latency'] = output
    final_path.write_text(json.dumps(final, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    arguments = parser.parse_args()
    main(arguments.repeats)
