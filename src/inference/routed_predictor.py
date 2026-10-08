"""Per-target routing across approved, locally stored research checkpoints."""
from __future__ import annotations

import copy
import threading
from pathlib import Path

from src.inference.predictor import Predictor
from src.selection.model_selector import load_registry, select_models
from src.utils import ROOT


class RoutedPredictor:
    """Run only the registered validated model assigned to each target.

    Routing is fixed from pre-upload validation evidence. It never picks the
    model with the largest score for an individual study.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.registry = load_registry()
        self.models = {entry["id"]: entry for entry in self.registry["models"]}
        self._predictors: dict[str, Predictor] = {}
        self._lock = threading.RLock()

    def _predictor(self, model_id: str) -> Predictor:
        with self._lock:
            if model_id not in self.models:
                raise ValueError(f"Registered model is unavailable: {model_id}")
            if model_id not in self._predictors:
                entry = self.models[model_id]
                self._predictors[model_id] = Predictor(
                    path=ROOT / entry["checkpoint"], cfg=self.cfg,
                    calibration_override=entry.get("calibration"),
                    thresholds_override=entry.get("thresholds"),
                )
            return self._predictors[model_id]

    def predict(self, volumes: dict, case_reference: str, selections: list[dict], uncertainty_samples: int = 1) -> dict:
        selected = {item["target"]: item for item in selections if item.get("status") == "Selected"}
        expected = {target["key"] for target in self.cfg["targets"]}
        if set(selected) != expected:
            missing = sorted(expected - set(selected))
            raise ValueError("No approved compatible model was selected for: " + ", ".join(missing))
        # Keep the configured target order.  A set here would make the copied
        # result metadata depend on hash iteration order rather than the route
        # that supplies each displayed finding.
        model_ids = tuple(dict.fromkeys(
            selected[target["key"]]["model_id"] for target in self.cfg["targets"]
        ))
        outputs = {
            model_id: self._predictor(model_id).predict(volumes, case_reference, uncertainty_samples=uncertainty_samples)
            for model_id in model_ids
        }
        base = copy.deepcopy(next(iter(outputs.values())))
        merged = []
        for target in self.cfg["targets"]:
            key = target["key"]
            selection = selected[key]
            model_id = selection["model_id"]
            finding = next(item for item in outputs[model_id]["findings"] if item["key"] == key)
            finding = copy.deepcopy(finding)
            finding["model_id"] = model_id
            finding["model_architecture"] = selection["architecture"]
            finding["selection_reason"] = selection["reason"]
            merged.append(finding)
        distances = [abs(float(item["probability"]) - float(item["threshold"])) for item in merged]
        margin = min(distances)
        if margin >= self.cfg["ui"]["margin_high"]:
            score_separation = "far from threshold"
        elif margin >= self.cfg["ui"]["margin_moderate"]:
            score_separation = "moderately separated"
        else:
            score_separation = "near threshold"
        average_variation = sum(float(item.get("uncertainty", 0.0)) for item in merged) / len(merged)
        variation_level = "small" if average_variation < 0.1 else "moderate" if average_variation < 0.2 else "large"

        # Recompute all score-dependent metadata from the merged routed
        # findings.  It must never inherit uncertainty or threshold separation
        # from an arbitrary single-model all-target output.
        base["findings"] = merged
        base["architecture"] = "validation-based target routing"
        base["training_stage"] = "multiple independently trained research routes"
        base["checkpoint_file"] = "multiple registered research checkpoints"
        base["checkpoint_sha256"] = None
        base["available_planes"] = list(volumes)
        base["missing_planes"] = [plane for plane in self.cfg["preprocessing"]["planes"] if plane not in volumes]
        base["score_separation"] = score_separation
        base["probability_note"] = (
            "Each route's probabilities were calibrated on a small reserved MRNet development subset; "
            "clinical and external calibration are not established."
        )
        base["uncertainty_note"] = (
            f"Exploratory MC-dropout probability variation: {average_variation:.3f} ({variation_level}). "
            "It does not change the deterministic calibrated score or research threshold status, and is not a "
            "calibrated measure of diagnostic uncertainty or correctness."
        )
        base["selected_models"] = {
            target["key"]: {
                "model_id": selected[target["key"]]["model_id"],
                "architecture": selected[target["key"]]["architecture"],
                "reason": selected[target["key"]]["reason"],
                "checkpoint_sha256": self.models[selected[target["key"]]["model_id"]].get("checkpoint_sha256"),
            }
            for target in self.cfg["targets"]
        }
        base["routing_note"] = (
            "Each finding uses the compatible registered model selected before upload from fixed development-validation evidence. "
            "The system does not choose a model from the individual study's predicted score."
        )
        return base

    def explain(self, volumes: dict, target: str, plane: str, model_id: str | None = None):
        """Generate attention from the exact registered route used for a finding."""
        if model_id is None:
            selection = select_models(list(volumes), [target])
            selected = selection["selections"][0]
            if selected.get("status") != "Selected":
                raise ValueError(f"No approved compatible model was selected for {target}.")
            model_id = selected["model_id"]
        entry = self.models.get(model_id)
        if entry is None or target not in entry.get("supported_targets", []):
            raise ValueError(f"The requested attention route is not registered for {target}.")
        return self._predictor(model_id).explain(volumes, target, plane)
