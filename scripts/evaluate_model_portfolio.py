"""Build a research-only, validation-based candidate portfolio for KneeAssist AI.

This evaluates existing checkpoints only. It uses MRNet fitting/tuning/calibration
partitions already defined in the project split, never external reference cohorts.
It does not update the active deployment or model registry.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import read_manifest, split_and_check, volumes
from src.evaluation.calibration import balanced_thresholds, fit_calibration, reliability
from src.evaluation.metrics import calculate
from src.inference.predictor import Predictor
from src.selection.model_selector import load_registry, metric_value
from src.utils import ROOT, config, save_json


CANDIDATES = {
    "active_efficientnet_b0": ROOT / "models/best_model.pth",
    "family_resnet18": ROOT / "runs/model_family_v2/models/resnet18_best.pth",
    "family_resnet50": ROOT / "runs/model_family_v2/models/resnet50_best.pth",
    "family_densenet121": ROOT / "runs/model_family_v2/models/densenet121_best.pth",
    "family_efficientnet_b0": ROOT / "runs/model_family_v2/models/efficientnet_b0_best.pth",
    "family_swin_t": ROOT / "runs/model_family_v2/models/swin_t_best.pth",
}


def raw_logits(predictor: Predictor, study_volumes: dict[str, np.ndarray]) -> np.ndarray:
    """Run one model without its saved calibration so all candidates are recalibrated fairly."""
    tensors, _ = predictor.tensors(study_volumes)
    with predictor.lock, torch.no_grad(), torch.autocast(device_type=predictor.device.type, enabled=predictor.amp):
        return predictor.model(tensors).float()[0].cpu().numpy()


def collect_logits(predictor: Predictor, rows: list[dict], cfg: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
    targets = [target["key"] for target in cfg["targets"]]
    labels, logits = [], []
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        labels.append([int(row[target]) for target in targets])
        logits.append(raw_logits(predictor, volumes(row, cfg)))
        if index % 20 == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            print(json.dumps({"stage": label, "completed": index, "total": len(rows), "elapsed_seconds": round(elapsed)}), flush=True)
    return np.asarray(labels, dtype=int), np.asarray(logits, dtype=float)


def score(metrics: dict, weights: dict[str, float]) -> float:
    return float(sum(metric_value(metrics, name) * float(weight) for name, weight in weights.items()))


def evaluate_candidate(candidate_id: str, checkpoint: Path, calibration_rows: list[dict], tuning_rows: list[dict], cfg: dict, weights: dict[str, float]) -> dict:
    if not checkpoint.is_file():
        return {"id": candidate_id, "status": "missing_checkpoint", "checkpoint": str(checkpoint.relative_to(ROOT))}
    predictor = Predictor(path=checkpoint, cfg=cfg)
    try:
        calibration_y, calibration_logits = collect_logits(predictor, calibration_rows, cfg, f"{candidate_id}:calibration")
        calibration = fit_calibration(calibration_y, calibration_logits, cfg["targets"])
        calibration_probabilities = expit(
            calibration_logits * np.asarray([item["scale"] for item in calibration["parameters"]])
            + np.asarray([item["bias"] for item in calibration["parameters"]])
        )
        thresholds = balanced_thresholds(calibration_y, calibration_probabilities, cfg["targets"])
        tuning_y, tuning_logits = collect_logits(predictor, tuning_rows, cfg, f"{candidate_id}:tuning")
        tuning_probabilities = expit(
            tuning_logits * np.asarray([item["scale"] for item in calibration["parameters"]])
            + np.asarray([item["bias"] for item in calibration["parameters"]])
        )
        metrics = calculate(tuning_y, tuning_probabilities, cfg["targets"], thresholds)
        return {
            "id": candidate_id,
            "status": "evaluated",
            "architecture": predictor.architecture,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "supported_targets": [target["key"] for target in predictor.model_cfg["targets"]],
            "supported_planes": list(predictor.model.planes),
            "minimum_planes": 1,
            "calibration": calibration,
            "calibration_reliability": reliability(calibration_y, calibration_probabilities, cfg["targets"]),
            "thresholds": thresholds,
            "tuning_metrics": metrics,
            "target_scores": {target["key"]: score(metrics[target["key"]], weights) for target in cfg["targets"]},
        }
    finally:
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing candidate checkpoints for target-specific research routing.")
    parser.add_argument("--output", default="runs/model_portfolio_v1")
    parser.add_argument("--candidates", nargs="*", choices=sorted(CANDIDATES), default=sorted(CANDIDATES))
    args = parser.parse_args()

    cfg = config()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    weights = registry["selection_policy"]["weights"]
    rows = read_manifest(cfg)
    split, leakage = split_and_check(rows, cfg)
    lookup = {row["study_id"]: row for row in rows}
    calibration_ids = split.get("calibration")
    if not calibration_ids:
        raise ValueError("The fixed MRNet split has no reserved calibration partition.")
    calibration_rows = [lookup[study_id] for study_id in calibration_ids]
    tuning_rows = [lookup[study_id] for study_id in split["tune"]]
    protocol = {
        "purpose": "Research candidate comparison for possible target-specific routing; not clinical validation.",
        "dataset": "MRNet-v1.0 only",
        "selection_partition": "fixed tuning partition",
        "calibration_partition": "reserved MRNet calibration partition",
        "excluded_from_selection": "official validation and all external cohorts",
        "selection_weights": weights,
        "study_counts": {"calibration": len(calibration_rows), "tuning": len(tuning_rows)},
        "leakage_audit": leakage,
        "limitation": "MRNet patient linkage is unavailable; this is study-level development evidence, not clinical validation.",
    }
    save_json(out / "protocol.json", protocol)
    results = []
    for candidate_id in args.candidates:
        print(json.dumps({"stage": "candidate_started", "candidate": candidate_id}), flush=True)
        result = evaluate_candidate(candidate_id, CANDIDATES[candidate_id], calibration_rows, tuning_rows, cfg, weights)
        results.append(result)
        save_json(out / "candidate_results.json", {"protocol": protocol, "candidates": results})
        print(json.dumps({"stage": "candidate_completed", "candidate": candidate_id, "status": result["status"]}), flush=True)

    eligible = [result for result in results if result["status"] == "evaluated"]
    winners = {}
    for target in cfg["targets"]:
        key = target["key"]
        if eligible:
            winner = max(eligible, key=lambda item: item["target_scores"][key])
            winners[key] = {
                "candidate_id": winner["id"],
                "architecture": winner["architecture"],
                "selection_score": winner["target_scores"][key],
                "metrics": winner["tuning_metrics"][key],
                "threshold": winner["thresholds"][key],
            }
    report = {"protocol": protocol, "candidates": results, "provisional_target_winners": winners,
              "activation_status": "pending_review",
              "activation_rule": "Do not update the active model registry until all candidates are reviewed and an evaluation report is produced."
    }
    save_json(out / "portfolio_report.json", report)
    print(json.dumps({"stage": "completed", "provisional_target_winners": winners}, indent=2), flush=True)


if __name__ == "__main__":
    main()
