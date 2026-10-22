"""Read-only evidence helpers for the model-selection dashboard.

The helpers deliberately read saved benchmark artefacts rather than executing
models.  This keeps model selection reproducible: validation evidence is set
before an uploaded study is scored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.selection.model_selector import metric_value
from src.utils import ROOT


DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "swin_t": "Swin Transformer (Swin-T)",
}


def display_name(architecture: str) -> str:
    return DISPLAY_NAMES.get(str(architecture).lower(), str(architecture))


def _registry(path: Path | None = None) -> dict[str, Any]:
    path = path or ROOT / "model_registry" / "active_models.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _score(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    return sum(metric_value(metrics, name) * float(weight) for name, weight in weights.items())


def candidate_evidence(path: Path | None = None) -> dict[str, Any]:
    """Return only saved benchmark evidence needed by the evaluation pages."""
    registry = _registry(path)
    candidates = registry.get("benchmark_candidates", registry.get("models", []))
    return {
        "registry": registry,
        "candidates": candidates,
        "weights": registry.get("selection_policy", {}).get("weights", {}),
        "protocol_path": ROOT / "runs" / "model_portfolio_v1" / "protocol.json",
        "portfolio_path": ROOT / "runs" / "model_portfolio_v1" / "portfolio_report.json",
    }


def formal_comparison_evidence() -> dict[str, Any]:
    """Read the primary two-model reproducibility evidence without loading weights.

    The dashboard uses this information to distinguish the dissertation's
    ResNet-18 versus EfficientNet-B0 comparison from a separate archived
    fixed-routing benchmark. Both are retained as evidence; neither drives
    dynamic selection or is recomputed when a new study is uploaded.
    """
    run_dir = ROOT / "runs" / "mrnet_repro_20261013"
    comparison_path = run_dir / "results" / "model_comparison.json"
    metrics_path = run_dir / "results" / "final" / "metrics.json"
    protocol_path = run_dir / "protocol.json"
    status_path = run_dir / "completion_status.json"
    required = (comparison_path, metrics_path, protocol_path, status_path)
    if not all(path.is_file() for path in required):
        return {"available": False, "reason": "The saved two-model reproducibility evidence is unavailable."}

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    rows = []
    for candidate in comparison.get("models", []):
        macro = candidate.get("metrics", {}).get("macro", {})
        rows.append({
            "Model": display_name(candidate.get("architecture", "unknown")),
            "Selected stage": (
                "Final encoder blocks fine-tuned"
                if candidate.get("architecture") == "resnet18"
                else "Frozen warm-up"
            ),
            "Validation macro AUROC": macro.get("auroc"),
            "PR-AUC": macro.get("average_precision"),
            "F1": macro.get("f1"),
            "Sensitivity": macro.get("sensitivity"),
            "Specificity": macro.get("specificity"),
            "Precision": macro.get("precision"),
            "Accuracy": macro.get("accuracy"),
        })
    return {
        "available": True,
        "winner": display_name(comparison.get("winner", status.get("winner", "unknown"))),
        "rows": rows,
        "final_metrics": metrics.get("metrics", {}),
        "counts": protocol.get("counts", {}),
        "selection": protocol.get("selection"),
        "checkpoint_sha256": status.get("checkpoint_sha256"),
        "limitation": (
            "This is MRNet development evidence from one repeat. The 120-study official-validation cohort had prior development exposure, "
            "so it is not independent external or clinical validation."
        ),
    }


def ranking_for_target(target: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Rank the fixed benchmark candidates for one finding from saved metrics."""
    evidence = candidate_evidence(path)
    rows: list[dict[str, Any]] = []
    active = {item["id"]: item for item in evidence["registry"].get("models", [])}
    for candidate in evidence["candidates"]:
        metrics = candidate.get("validation_metrics", {}).get(target)
        if not metrics:
            continue
        calibration = candidate.get("calibration_reliability", {}).get(target, {})
        deployed = active.get(candidate["id"])
        checkpoint = ROOT / deployed["checkpoint"] if deployed else None
        rows.append({
            "model_id": candidate["id"],
            "architecture": candidate["architecture"],
            "model": display_name(candidate["architecture"]),
            "status": "Checkpoint locally available" if checkpoint and checkpoint.is_file() else "Comparison result retained; checkpoint not installed",
            "locally_available": bool(checkpoint and checkpoint.is_file()),
            "selection_score": _score(metrics, evidence["weights"]),
            "metrics": metrics,
            "brier": calibration.get("brier"),
            "log_loss": calibration.get("log_loss"),
            "inference_time": candidate.get("inference_time_seconds"),
        })
    return sorted(rows, key=lambda row: row["selection_score"], reverse=True)


def split_summary(path: Path | None = None) -> list[dict[str, Any]]:
    """Summarise fixed split roles without reassigning or resampling studies."""
    evidence = candidate_evidence(path)
    protocol_path = evidence["protocol_path"]
    if not protocol_path.exists():
        return []
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    counts = protocol.get("leakage_audit", {}).get("split_counts", {})
    labels = {
        "fit": "Fitting",
        "tune": "Tuning validation (selection only)",
        "calibration": "Calibration (score mapping and thresholds)",
        "official_valid": "Official validation (previously used development cohort)",
    }
    role_order = ("fit", "tune", "calibration", "official_valid")
    return [
        {"Partition": labels.get(key, key), "Studies": counts[key]}
        for key in role_order
        if key in counts
    ]


def objective_traceability() -> list[dict[str, str]]:
    return [
        {
            "Objective": "Develop an explainable study-level knee MRI system",
            "Evidence": "MRI-only intake, independent inference, study-level pooling, prediction display and research-summary export.",
        },
        {
            "Objective": "Preprocess knee MRI studies",
            "Evidence": "Fixed MRNet split, percentile normalisation, resizing, slice sampling and guarded MRI input checks.",
        },
        {
            "Objective": "Compare ResNet-18 and EfficientNet-B0",
            "Evidence": "Saved two-model reproducibility evidence with internal-validation macro AUROC used for selection.",
        },
        {
            "Objective": "Evaluate the selected classifier",
            "Evidence": "Saved MRNet development metrics: accuracy, sensitivity, specificity, precision, F1-score, AUROC and PR-AUC.",
        },
        {
            "Objective": "Generate Grad-CAM visualisations",
            "Evidence": "Per-target original slice and Grad-CAM overlay, labelled as coarse classifier attention rather than lesion localisation.",
        },
    ]

