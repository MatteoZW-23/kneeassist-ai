"""Read-only deployment evidence for the dashboard, without loading model weights."""
import json
from src.utils import resolve

def deployment_evidence(cfg):
    folder=resolve(cfg['evaluation']['directory'])
    report=folder/'final/metrics.json'
    result={'checkpoint':cfg['evaluation']['checkpoint'],'metrics':None,'architecture':None}
    if report.exists():
        try:
            saved=json.loads(report.read_text(encoding='utf-8'))
            result.update(metrics=saved.get('metrics'),architecture=saved.get('architecture'))
        except (OSError,ValueError):pass
    return result
