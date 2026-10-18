"""All-five-model inference and guarded case-specific dynamic selection."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.dynamic_selection.features import build_feature_vector
from src.dynamic_selection.selector import load_selector_artifact, runtime_artifact_diagnostics, select_model
from src.data.preprocessing import InputError, validate_volume
from src.utils import ROOT


class DynamicSelectionUnavailable(InputError):
    """Raised when all-five dynamic selection cannot be performed truthfully."""


def selector_artifact_path(cfg: Mapping[str, Any]) -> Path:
    dynamic_cfg = cfg.get("dynamic_selection", {})
    raw = dynamic_cfg.get("artifact", "runs/dynamic_selection_v1/selector/dynamic_selector.joblib")
    return ROOT / raw


def _registered_selector(cfg: Mapping[str, Any]) -> dict[str, Any]:
    path=selector_artifact_path(cfg)
    registry_value=cfg.get('dynamic_selection',{}).get('registry')
    if registry_value:
        registry_path=ROOT/registry_value
        registry=json.loads(registry_path.read_text(encoding='utf-8'))
        if (ROOT/registry['selector_artifact']).resolve()!=path.resolve():
            raise DynamicSelectionUnavailable('The dynamic registry and configured selector path disagree.')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=registry['selector_sha256']:
            raise DynamicSelectionUnavailable('The registered dynamic selector integrity check failed.')
        artifact=load_selector_artifact(path,require_active=True)
        runtime_artifact_diagnostics(artifact)
        registered={(entry['id'],entry['checkpoint_sha256']) for entry in registry['models']}
        embedded={(entry['id'],entry['checkpoint_sha256']) for entry in artifact['model_entries']}
        if registered!=embedded:
            raise DynamicSelectionUnavailable('The dynamic registry and selector checkpoint entries disagree.')
        return artifact
    return load_selector_artifact(path,require_active=True)


def dynamic_selection_status(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return availability without falling back to static routing."""
    if not cfg.get("dynamic_selection", {}).get("enabled", False):
        return {"available": False, "reason": "Dynamic selection is disabled in configuration."}
    try:
        artifact = _registered_selector(cfg)
        diagnostics = runtime_artifact_diagnostics(artifact)
        entries = {entry["id"]: entry for entry in artifact["model_entries"]}
        missing = [
            model_id
            for model_id in artifact["model_ids"]
            if model_id not in entries or not (ROOT / entries[model_id]["checkpoint"]).is_file()
        ]
        if missing:
            return {"available": False, "reason": "Missing dynamic-selection checkpoints: " + ", ".join(missing)}
        return {
            "available": True,
            "artifact": str(selector_artifact_path(cfg)),
            "model_ids": artifact["model_ids"],
            "selection_rule": artifact["selection_rule"],
            "required_planes": diagnostics["required_planes"],
            "feature_count": diagnostics["feature_count"],
            "estimator_type": diagnostics["estimator_type"],
            "provenance_warnings": diagnostics["provenance_warnings"],
            "checkpoint_stamps": [
                (model_id,str(ROOT/entries[model_id]['checkpoint']),
                 (ROOT/entries[model_id]['checkpoint']).stat().st_mtime_ns,
                 (ROOT/entries[model_id]['checkpoint']).stat().st_size)
                for model_id in artifact['model_ids']
            ],
        }
    except Exception as error:  # UI must show why DMS cannot be claimed.
        return {"available": False, "reason": str(error)}


class DynamicSelectionPredictor:
    """Run five real base models, then one learned loss estimator per upload.

    A base model is constructed only for the duration of the needed prediction.
    This bounds GPU memory on local hardware while ensuring every candidate is
    freshly executed for each uploaded study.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        try:
            self.artifact = _registered_selector(cfg)
        except Exception as error:
            raise DynamicSelectionUnavailable(str(error)) from error
        self.entries = {entry["id"]: entry for entry in self.artifact["model_entries"]}
        self.selector_diagnostics = runtime_artifact_diagnostics(self.artifact)
        self.lock = threading.RLock()
        self._validate_entries()

    def _validate_entries(self) -> None:
        if list(self.artifact["targets"]) != [target["key"] for target in self.cfg["targets"]]:
            raise DynamicSelectionUnavailable("The selector target order differs from the application configuration.")
        if list(self.artifact["planes"]) != list(self.cfg["preprocessing"]["planes"]):
            raise DynamicSelectionUnavailable("The selector plane configuration differs from the application configuration.")
        for model_id in self.artifact["model_ids"]:
            entry = self.entries.get(model_id)
            if entry is None:
                raise DynamicSelectionUnavailable(f"Dynamic selector is missing metadata for {model_id}.")
            checkpoint = ROOT / entry["checkpoint"]
            if not checkpoint.is_file():
                raise DynamicSelectionUnavailable(f"Dynamic selector checkpoint is unavailable: {checkpoint}")
            expected_hash = entry.get("checkpoint_sha256")
            with checkpoint.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if not expected_hash or actual_hash != expected_hash:
                raise DynamicSelectionUnavailable(f"Checkpoint integrity check failed for {model_id}.")

    def _predict_with_candidate(
        self,
        model_id: str,
        volumes: Mapping[str, np.ndarray],
        case_reference: str,
    ) -> dict[str, Any]:
        from src.inference.predictor import Predictor
        entry = self.entries[model_id]
        predictor = Predictor(
            path=ROOT / entry["checkpoint"],
            cfg=self.cfg,
            calibration_override=entry.get("calibration"),
            thresholds_override=entry.get("thresholds"),
        )
        try:
            # Calibration and the DMS evaluation were run in FP32. Keep exactly
            # that numerical path at upload time rather than changing to AMP.
            predictor.amp = False
            # Entropy from deterministic calibrated probabilities is retained as
            # a selector feature. MC-dropout remains optional visual context and
            # is not substituted for a calibrated reliability estimate.
            return predictor.predict(volumes, case_reference, uncertainty_samples=1)
        finally:
            del predictor
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict(self, volumes: Mapping[str, np.ndarray], case_reference: str = "Not supplied") -> dict[str, Any]:
        # Streamlit may share a cached predictor across sessions. Serialise GPU
        # use, and never cache a patient's result inside this predictor.
        with self.lock:
            if not volumes:
                raise InputError("Upload at least one MRI sequence before analysis.")
            if set(volumes) - set(self.artifact["planes"]):
                raise InputError("An unsupported imaging plane was supplied.")
            missing_planes = [plane for plane in self.artifact["planes"] if plane not in volumes]
            if missing_planes:
                raise InputError(
                    "Dynamic selection is validated only for complete three-plane MRI studies. "
                    "Upload axial, coronal, and sagittal sequences for the same study; missing: "
                    + ", ".join(missing_planes)
                    + "."
                )
            for volume in volumes.values():
                validate_volume(volume, self.cfg["preprocessing"])
            return self._predict_study(volumes, case_reference)

    def _predict_study(self, volumes: Mapping[str, np.ndarray], case_reference: str) -> dict[str, Any]:
        started = time.perf_counter()
        target_keys = self.artifact["targets"]
        outputs: dict[str, dict[str, Any]] = {}
        probabilities: dict[str, list[float]] = {}
        candidate_metadata: dict[str, dict[str, Any]] = {}
        for model_id in self.artifact["model_ids"]:
            candidate_started = time.perf_counter()
            try:
                output = self._predict_with_candidate(model_id, volumes, case_reference)
            except Exception as error:
                raise DynamicSelectionUnavailable(
                    f"Dynamic inference stopped because {model_id} failed. No partial five-model result was returned. {error}"
                ) from error
            by_target = {item["key"]: item for item in output["findings"]}
            if set(by_target) != set(target_keys):
                raise DynamicSelectionUnavailable(f"Candidate {model_id} returned incompatible targets.")
            outputs[model_id] = output
            probabilities[model_id] = [float(by_target[target]["probability"]) for target in target_keys]
            p = np.clip(np.asarray(probabilities[model_id]), 1e-7, 1 - 1e-7)
            candidate_metadata[model_id] = {
                "architecture": self.entries[model_id]["architecture"],
                "checkpoint_sha256": self.entries[model_id].get("checkpoint_sha256"),
                "seconds_including_checkpoint_load": time.perf_counter() - candidate_started,
                "predictive_entropy": (-(p * np.log(p) + (1 - p) * np.log(1 - p))).tolist(),
                "calibration_source": "reserved 100-study MRNet calibration partition",
                "precision": "float32",
            }

        is_extended = len(self.artifact.get("feature_names", [])) > 54 or self.artifact.get("extended_features", False)
        if is_extended:
            vector = build_feature_vector(
                probabilities,
                volumes,
                self.artifact["model_ids"],
                target_keys,
                self.artifact["planes"],
                True,
            )
        else:
            vector = build_feature_vector(
                probabilities,
                volumes,
                self.artifact["model_ids"],
                target_keys,
                self.artifact["planes"],
            )
        decision = select_model(self.artifact, vector)
        selected_id = decision["selected_model_id"]
        selected_entry = self.entries[selected_id]
        result = copy.deepcopy(outputs[selected_id])
        for finding in result["findings"]:
            # One deterministic pass cannot measure MC-dropout variation. Do
            # not present a zero standard deviation as measured confidence.
            finding.pop("uncertainty", None)
            finding.pop("uncertainty_level", None)
            finding["model_id"] = selected_id
            finding["model_architecture"] = selected_entry["architecture"]
            finding["selection_reason"] = (
                "All five candidate models were executed for this MRI study. "
                "The learned selector estimated expected mean Brier loss from their current probabilities, "
                "prediction entropy, inter-model disagreement and acquisition features; the lowest estimate was selected."
            )
        result["architecture"] = "dynamic case-specific model selection"
        result["training_stage"] = "five independently trained base classifiers plus a learned competence selector"
        result["checkpoint_file"] = Path(selected_entry.get("checkpoint", selected_id)).name
        result["checkpoint_sha256"] = selected_entry.get("checkpoint_sha256")
        result["uncertainty_note"] = (
            "Prediction entropy and inter-model disagreement inform the selector. They are research features; "
            "neither they nor the estimated Brier loss are calibrated probabilities of correctness. MC-dropout was not measured in this analysis."
        )
        result["selected_models"] = {
            target: {
                "model_id": selected_id,
                "architecture": selected_entry["architecture"],
                "checkpoint_sha256": selected_entry.get("checkpoint_sha256"),
                "reason": "Lowest learned expected mean Brier loss for this study among all five executed candidates.",
            }
            for target in target_keys
        }
        result["dynamic_selection"] = {
            "mode": "dynamic_case_specific_competence_estimation",
            "models_evaluated": len(outputs),
            "model_ids": list(self.artifact["model_ids"]),
            "selected_model_id": selected_id,
            "selected_architecture": selected_entry["architecture"],
            "expected_losses": decision["expected_losses"],
            "ranked_model_ids": decision["ranked_model_ids"],
            "base_probabilities": probabilities,
            "candidate_metadata": candidate_metadata,
            "total_seconds": time.perf_counter() - started,
            "selector_artifact_sha256": hashlib.sha256(selector_artifact_path(self.cfg).read_bytes()).hexdigest(),
            "split_sha256": self.artifact.get("protocol", {}).get("split_sha256"),
            "feature_count": int(vector.size),
            "selector_estimator": self.selector_diagnostics["estimator_type"],
            "selector_provenance_warnings": self.selector_diagnostics["provenance_warnings"],
            "selection_rule": decision["selection_rule"],
            "principal_factors": decision.get("principal_factors", []),
            "factor_interpretation": decision.get("factor_interpretation"),
            "interpretation": (
                "Expected loss is a selector estimate, not the true error, correctness probability, "
                "clinical confidence or diagnosis for this unlabelled study."
            ),
        }
        result["routing_note"] = (
            "Dynamic case-specific selection: all five compatible registered models were run on this upload. "
            "A selector trained on a separate labelled development partition chose one architecture for the entire study."
        )
        result["explanation_status"] = "Not requested yet. Attention is generated explicitly for the selected model."
        return result

    def explain(self, volumes: Mapping[str, np.ndarray], target: str, plane: str, model_id: str) -> dict[str, Any]:
        if model_id not in self.entries:
            raise DynamicSelectionUnavailable(f"Unknown dynamically selected model: {model_id}")
        from src.inference.predictor import Predictor
        entry = self.entries[model_id]
        predictor = Predictor(
            path=ROOT / entry["checkpoint"],
            cfg=self.cfg,
            calibration_override=entry.get("calibration"),
            thresholds_override=entry.get("thresholds"),
        )
        try:
            return predictor.explain(volumes, target, plane)
        finally:
            del predictor
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
