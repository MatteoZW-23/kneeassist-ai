"""Expose only trained, validated localisation models to the dashboard."""
import json
from pathlib import Path

from src.utils import ROOT


def localization_status(path=None):
    registry_path = Path(path or ROOT / 'model_registry' / 'active_models.json')
    with registry_path.open(encoding='utf-8') as stream:
        registry = json.load(stream)
    entries = registry.get('localizers', [])
    status = {}
    for kind in ('detection', 'segmentation'):
        approved = []
        for entry in entries:
            checkpoint = ROOT / entry.get('checkpoint', '')
            if (entry.get('kind') == kind and checkpoint.is_file() and
                    entry.get('validation', {}).get('status') == 'validated' and
                    entry.get('label_provenance', {}).get('status') == 'verified'):
                approved.append(entry)
        if approved:
            status[kind] = {'available': True, 'models': approved}
        else:
            status[kind] = {
                'available': False,
                'reason': ('No trained and validated ' + kind + ' model is registered. '
                           'MRI localisation labels and a locked evaluation are required before this feature can be enabled.')
            }
    return status
