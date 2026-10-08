"""Fast non-clinical release audit for reproducible research artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ROOT


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "model_registry/active_models.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / cfg["evaluation"]["checkpoint"]
    manifest_path = ROOT / cfg["dataset"]["manifest"]
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    split = json.loads((ROOT / cfg["dataset"]["splits"]).read_text(encoding="utf-8"))
    roles = {study_id: role for role, ids in split.items() for study_id in ids}
    routed = registry.get("routing_status") == "active_development_routing"
    metrics_path = (ROOT / "runs/model_portfolio_v1/final/metrics.json" if routed
                    else ROOT / cfg["evaluation"]["directory"] / "final/metrics.json")
    required = [
        ROOT / "docs/REPRODUCIBILITY.md",
        ROOT / "docs/USABILITY_EVALUATION.md",
        ROOT / "docs/usability_responses_template.csv",
        ROOT / "runs/mri_finetune_02/results/final/error_analysis/README.md",
        ROOT / "runs/mri_finetune_02/results/final/error_analysis/review_queue.csv",
        metrics_path,
    ]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Active checkpoint is missing: {checkpoint}")
    if len(rows) != 1250 or len({row['study_id'] for row in rows}) != 1250:
        raise ValueError("MRNet manifest must contain 1,250 unique studies.")
    if set(roles) != {row['study_id'] for row in rows}:
        raise ValueError("The fixed split does not cover the manifest exactly once.")
    if len(roles) != sum(len(ids) for ids in split.values()):
        raise ValueError("A study appears in more than one split role.")
    if not all(path.is_file() for path in required):
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        raise FileNotFoundError("Required research artifacts are missing: " + ", ".join(missing))
    active = registry.get("models", [])
    if not active:
        raise ValueError("The active registry has no models.")
    for model in active:
        model_checkpoint = ROOT / model.get("checkpoint", "")
        if not model_checkpoint.is_file():
            raise FileNotFoundError(f"Registered checkpoint is missing: {model_checkpoint}")
    required_targets = {target["key"] for target in cfg["targets"]}
    registered_targets = {target for model in active for target in model.get("supported_targets", [])}
    if not required_targets.issubset(registered_targets):
        raise ValueError("The active registry does not cover every configured finding.")
    if routed and len(active) < len(required_targets):
        raise ValueError("The routed registry must contain an approved entry for every finding.")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = {
        "status": "passed",
        "active_checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": digest,
        "registered_models": [{"id": model["id"], "architecture": model["architecture"],
                                "targets": model["supported_targets"], "checkpoint": model["checkpoint"]}
                               for model in active],
        "routing_status": registry.get("routing_status", "single-model registry"),
        "manifest_studies": len(rows),
        "split_counts": {role: len(ids) for role, ids in split.items()},
        "metrics_partition": metrics.get("partition"),
        "limitations": "This audit checks artifact presence and split structure. It does not establish clinical validation or patient-level independence.",
    }
    destination = ROOT / "results/research_release_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
