"""Promote preselected portfolio winners into the local research model registry."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ROOT, save_json


def main() -> None:
    portfolio_dir = ROOT / "runs/model_portfolio_v1"
    report = json.loads((portfolio_dir / "portfolio_report.json").read_text(encoding="utf-8"))
    final = json.loads((portfolio_dir / "final/metrics.json").read_text(encoding="utf-8"))
    candidates = {item["id"]: item for item in report["candidates"] if item["status"] == "evaluated"}
    routing = final["routing"]
    selection_winners = report["provisional_target_winners"]
    entries = []
    for target, winner in routing.items():
        candidate = candidates[winner["candidate_id"]]
        checkpoint = ROOT / candidate["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
        entries.append({
            "id": candidate["id"],
            "architecture": candidate["architecture"],
            "version": "portfolio-routing-v1",
            "checkpoint": candidate["checkpoint"].replace("\\", "/"),
            "training_dataset": "MRNet-v1.0",
            "supported_targets": [target],
            "supported_planes": candidate["supported_planes"],
            "minimum_planes": candidate["minimum_planes"],
            "preprocessing_config": "checkpoint embedded configuration",
            "calibration": candidate["calibration"],
            "thresholds": candidate["thresholds"],
            "calibration_metrics": candidate["calibration_reliability"],
            # Selection evidence comes from the locked tuning partition.  The
            # final development metrics are deliberately recorded separately
            # below so they cannot be reused to choose a model.
            "validation_metrics": {target: selection_winners[target]["metrics"]},
            "final_development_metrics": {target: final["metrics"][target]},
            "inference_profile": "Local PyTorch study-level inference; target-specific routing from pre-upload validation evidence.",
            "limitations": "MRNet-only development evidence; study-level split; patient linkage unavailable; official validation previously used in development; not clinically validated.",
        })
    registry = {
        "registry_version": "2.0",
        "routing_status": "active_development_routing",
        "selection_policy": {
            "name": "validated_research_score_v1",
            "weights": report["protocol"]["selection_weights"],
            "rule": "For each target, only a checkpoint preselected from the fixed tuning partition and compatible with the uploaded MRI planes is eligible. Individual prediction scores never select a model.",
        },
        "routing_evidence": {
            "portfolio_report": "runs/model_portfolio_v1/portfolio_report.json",
            "final_development_evaluation": "runs/model_portfolio_v1/final/metrics.json",
            "selection_partition": "MRNet tuning partition",
            "final_partition": "MRNet official validation previously used in development; not independent clinical validation.",
        },
        "ensemble_policy": {
            "status": "disabled",
            "rule": "This is target routing, not a probability ensemble. No weighted ensemble is active.",
        },
        "ensembles": [],
        "localizers": [],
        "models": entries,
    }
    path = ROOT / "model_registry/active_models.json"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    decision = {
        "status": "promoted_for_research_routing",
        "routing": routing,
        "macro_auroc": final["metrics"]["macro"]["auroc"],
        "limitations": final["protocol"]["limitations"],
        "active_registry": str(path.relative_to(ROOT)),
    }
    save_json(portfolio_dir / "activation_decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
