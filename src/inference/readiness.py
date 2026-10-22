"""Read-only development evidence for the dashboard, without loading weights."""
import json
from src.utils import ROOT, resolve

def deployment_evidence(cfg):
    registry_path = ROOT / 'model_registry' / 'active_models.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {}
    routed = registry.get('routing_status') in {
        'active_development_routing',
        'active_upload_time_candidate_selection',
        'archived_fixed_routing_fallback',
    }
    folder = ROOT / 'runs/model_portfolio_v1' if routed else resolve(cfg['evaluation']['directory'])
    report=folder/'final/metrics.json'
    models = [
        model for model in registry.get('models', [])
        if model.get('deployment_status', 'selected') == 'selected'
    ]
    label = ', '.join(f"{item.get('architecture', 'unknown')} ({', '.join(item.get('supported_targets', []))})" for item in models)
    result={'checkpoint': label or cfg['evaluation']['checkpoint'],'metrics':None,'architecture':None,
            'routing_status': registry.get('routing_status', 'single-model registry')}
    if report.exists():
        try:
            saved=json.loads(report.read_text(encoding='utf-8'))
            result.update(metrics=saved.get('metrics'),architecture=saved.get('architecture'))
        except (OSError,ValueError):pass
    return result
