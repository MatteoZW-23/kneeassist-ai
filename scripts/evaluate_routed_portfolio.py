"""Evaluate the prespecified routed portfolio on MRNet official validation.

This script consumes winners already fixed by the tuning partition. It reports
final development-validation performance and never changes winner selection.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import read_manifest, split_and_check, volumes
from src.evaluation.metrics import calculate
from src.evaluation.plots import curves
from src.inference.predictor import Predictor
from src.utils import ROOT, config, save_json


def main() -> None:
    cfg = config()
    portfolio_dir = ROOT / "runs/model_portfolio_v1"
    report = json.loads((portfolio_dir / "portfolio_report.json").read_text(encoding="utf-8"))
    winners = report["provisional_target_winners"]
    candidates = {candidate["id"]: candidate for candidate in report["candidates"] if candidate["status"] == "evaluated"}
    target_keys = [target["key"] for target in cfg["targets"]]
    if set(winners) != set(target_keys):
        raise ValueError("Portfolio report does not contain a fixed winner for every target.")

    rows = read_manifest(cfg)
    split, _ = split_and_check(rows, cfg)
    lookup = {row["study_id"]: row for row in rows}
    selected = [lookup[study_id] for study_id in split["official_valid"]]
    output = portfolio_dir / "final"
    output.mkdir(parents=True, exist_ok=True)

    predictors = {}
    for target in target_keys:
        candidate_id = winners[target]["candidate_id"]
        candidate = candidates[candidate_id]
        if candidate_id not in predictors:
            predictors[candidate_id] = Predictor(
                path=ROOT / candidate["checkpoint"], cfg=cfg,
                calibration_override=candidate["calibration"],
                thresholds_override=candidate["thresholds"],
            )

    labels, probabilities = [], []
    for index, row in enumerate(selected, start=1):
        study = volumes(row, cfg)
        outputs = {candidate_id: predictor.predict(study, "MRNet-" + row["study_id"], uncertainty_samples=1)
                   for candidate_id, predictor in predictors.items()}
        labels.append([int(row[key]) for key in target_keys])
        by_target = []
        for target in target_keys:
            candidate_id = winners[target]["candidate_id"]
            finding = next(item for item in outputs[candidate_id]["findings"] if item["key"] == target)
            by_target.append(float(finding["probability"]))
        probabilities.append(by_target)
        if index % 20 == 0 or index == len(selected):
            print(json.dumps({"stage": "official_validation", "completed": index, "total": len(selected)}), flush=True)

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    thresholds = {target: float(winners[target]["threshold"]) for target in target_keys}
    metrics = calculate(y, p, cfg["targets"], thresholds)
    protocol = {
        "selection_source": "Fixed tuning-partition portfolio report; final results did not select or alter winners.",
        "calibration_source": "Reserved MRNet calibration partition, per selected candidate.",
        "partition": "MRNet official validation (previously used in earlier development; not an independent external test).",
        "limitations": "Study-level split only; patient mapping unavailable; no clinical validation or reader study.",
    }
    result = {
        "dataset": "MRNet-v1.0", "partition": "official_valid", "exams": len(selected),
        "routing": {target: {"candidate_id": winners[target]["candidate_id"], "architecture": winners[target]["architecture"], "threshold": thresholds[target]} for target in target_keys},
        "metrics": metrics, "protocol": protocol,
    }
    save_json(output / "metrics.json", result)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["study_id"] + [key + "_label" for key in target_keys] + [key + "_probability" for key in target_keys])
        for row, truth, prediction in zip(selected, y, p):
            writer.writerow([row["study_id"], *truth, *prediction])
    curves(y, p, metrics, cfg["targets"], output)
    save_json(output / "activation_candidate.json", {"routing": result["routing"], "protocol": protocol, "metrics": metrics})
    print(json.dumps({"stage": "completed", "macro_auroc": metrics["macro"]["auroc"], "routing": result["routing"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
