"""Train and apply a learned case-specific competence estimator.

The selector estimates each candidate's expected mean Brier loss.  It does not
use a patient's unknown label, AUROC or real-time accuracy at upload time.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dynamic_selection.features import feature_names


ARTIFACT_VERSION = "dynamic-competence-selector-v1"
SUPPORTED_ARTIFACT_VERSIONS = {"dynamic-competence-selector-v1", "dynamic-competence-selector-v2"}


def runtime_artifact_diagnostics(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the selector schema that is safe to use for an upload.

    Selection is only meaningful when the current five-model output vector and
    input-plane schema match the schema used to fit the saved estimator. This
    check does not reinterpret recorded performance evidence; it fails closed
    when a changed application configuration or malformed artifact would make
    the model choice unreliable.
    """
    model_ids = artifact.get("model_ids", [])
    targets = artifact.get("targets", [])
    planes = artifact.get("planes", [])
    names = artifact.get("feature_names", [])
    if not all(isinstance(value, list) and value for value in (model_ids, targets, planes, names)):
        raise ValueError("Dynamic selector is missing its model, target, plane, or feature schema.")

    base_names = feature_names(model_ids, targets, planes, extended=False)
    extended_names = feature_names(model_ids, targets, planes, extended=True)
    declared_extended = artifact.get("extended_features")
    if declared_extended is None:
        if names == base_names:
            extended = False
        elif names == extended_names:
            extended = True
        else:
            raise ValueError("Dynamic selector feature names do not match a supported runtime schema.")
    else:
        extended = bool(declared_extended)
        expected_names = extended_names if extended else base_names
        if names != expected_names:
            raise ValueError("Dynamic selector feature names do not match its declared feature schema.")

    estimator_types: set[str] = set()
    for model_id in model_ids:
        estimator = artifact["competence_models"][model_id]
        steps = getattr(estimator, "named_steps", {})
        if "gbr" in steps:
            estimator_types.add("gbr")
        elif "ridge" in steps:
            estimator_types.add("ridge")
        else:
            raise ValueError(f"Dynamic selector estimator for {model_id} is unsupported.")
    if len(estimator_types) != 1:
        raise ValueError("Dynamic selector contains mixed estimator families.")
    estimator_type = estimator_types.pop()
    if artifact.get("estimator_type") not in (None, estimator_type):
        raise ValueError("Dynamic selector estimator metadata does not match its fitted estimators.")

    warnings: list[str] = []
    protocol_text = str(artifact.get("protocol", {}).get("selector", "")).lower()
    protocol_estimator = "gbr" if "gradientboostingregressor" in protocol_text else "ridge" if "ridgecv" in protocol_text else None
    if protocol_estimator and protocol_estimator != estimator_type:
        warnings.append(
            "The archived protocol text names a different selector estimator than the active artifact. "
            "Runtime uses the fitted artifact metadata; the historical protocol remains unchanged."
        )
    return {
        "feature_count": len(names),
        "extended_features": extended,
        "estimator_type": estimator_type,
        "required_planes": list(planes),
        "provenance_warnings": warnings,
    }


def _pipeline(estimator_type: str = "ridge", seed: int = 42) -> Pipeline:
    if estimator_type == "gbr":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "gbr",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        min_samples_leaf=5,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        tol=1e-4,
                        random_state=seed,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )


def train_competence_models(
    features: np.ndarray,
    observed_losses: np.ndarray,
    model_ids: Sequence[str],
    estimator_type: str = "ridge",
    seed: int = 42,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]]]:
    """Fit one regularised expected-loss estimator per candidate architecture."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(observed_losses, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0], len(model_ids)):
        raise ValueError("Selector training features or loss targets have invalid dimensions.")
    if x.shape[0] < 30 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Selector training requires at least 30 finite labelled studies.")
    folds = KFold(n_splits=min(5, x.shape[0]), shuffle=True, random_state=seed)
    models: dict[str, Pipeline] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for index, model_id in enumerate(model_ids):
        estimator = _pipeline(estimator_type=estimator_type, seed=seed)
        out_of_fold = cross_val_predict(estimator, x, y[:, index], cv=folds, method="predict")
        fitted = _pipeline(estimator_type=estimator_type, seed=seed).fit(x, y[:, index])
        models[model_id] = fitted
        diag: dict[str, float] = {
            "selector_train_mean_observed_brier": float(np.mean(y[:, index])),
            "selector_train_oof_mae": float(np.mean(np.abs(out_of_fold - y[:, index]))),
            "selector_train_oof_rmse": float(np.sqrt(np.mean((out_of_fold - y[:, index]) ** 2))),
        }
        if "ridge" in fitted.named_steps:
            diag["selected_ridge_alpha"] = float(fitted.named_steps["ridge"].alpha_)
            diag["estimator_type"] = "RidgeCV"
        elif "gbr" in fitted.named_steps:
            gbr_model = fitted.named_steps["gbr"]
            diag["n_estimators_used"] = float(gbr_model.n_estimators_)
            ss_tot = max(float(np.sum((y[:, index] - y[:, index].mean()) ** 2)), 1e-12)
            diag["selector_train_oof_r2"] = float(1.0 - np.sum((out_of_fold - y[:, index]) ** 2) / ss_tot)
            diag["estimator_type"] = "GradientBoostingRegressor"
        diagnostics[model_id] = diag
    return models, diagnostics


def train_target_competence_models(
    features: np.ndarray,
    observed_losses_by_target: Mapping[str, np.ndarray],
    model_ids: Sequence[str],
    estimator_type: str = "ridge",
    seed: int = 42,
) -> tuple[dict[str, dict[str, Pipeline]], dict[str, dict[str, dict[str, float]]]]:
    """Fit independent candidate-loss estimators for each prediction target.

    The existing study-level selector estimates one loss averaged over all
    targets.  This helper is intentionally separate: a target-specific policy
    must be trained and evaluated as a new experiment rather than silently
    changing the interpretation of the active study-level artifact.
    """
    if not observed_losses_by_target:
        raise ValueError("At least one target-specific loss matrix is required.")
    models_by_target: dict[str, dict[str, Pipeline]] = {}
    diagnostics_by_target: dict[str, dict[str, dict[str, float]]] = {}
    for target, losses in observed_losses_by_target.items():
        models, diagnostics = train_competence_models(
            features,
            losses,
            model_ids,
            estimator_type=estimator_type,
            seed=seed,
        )
        models_by_target[str(target)] = models
        diagnostics_by_target[str(target)] = diagnostics
    return models_by_target, diagnostics_by_target


def build_target_selector_artifact(
    *,
    model_ids: Sequence[str],
    targets: Sequence[str],
    planes: Sequence[str],
    feature_names: Sequence[str],
    models_by_target: Mapping[str, Mapping[str, Pipeline]],
    diagnostics_by_target: Mapping[str, Mapping[str, Mapping[str, float]]],
    model_entries: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    estimator_type: str,
    extended_features: bool,
) -> dict[str, Any]:
    """Build an inactive artifact for a target-specific DMS experiment.

    It is deliberately not accepted by ``load_selector_artifact`` yet.  A
    benchmark must demonstrate benefit and pass a separate activation review
    before a different selection granularity reaches the clinical UI.
    """
    expected_targets = set(targets)
    if set(models_by_target) != expected_targets:
        raise ValueError("A target-specific selector is required for every target.")
    for target, models in models_by_target.items():
        if set(models) != set(model_ids):
            raise ValueError(f"Target '{target}' is missing candidate competence estimators.")
    return {
        "artifact_version": "target-specific-competence-selector-v1",
        "activation_status": "pending_verification",
        "selection_granularity": "per_target",
        "model_ids": list(model_ids),
        "targets": list(targets),
        "planes": list(planes),
        "feature_names": list(feature_names),
        "competence_models_by_target": {
            target: dict(models) for target, models in models_by_target.items()
        },
        "training_diagnostics_by_target": {
            target: {model_id: dict(values) for model_id, values in report.items()}
            for target, report in diagnostics_by_target.items()
        },
        "model_entries": [dict(entry) for entry in model_entries],
        "protocol": dict(protocol),
        "estimator_type": estimator_type,
        "extended_features": bool(extended_features),
        "selection_rule": "For each target, select the candidate with the lowest learned expected target-specific Brier loss.",
        "inference_restriction": "Only current base-model probabilities and uploaded-study features are used. Unknown case labels and final-evaluation metrics are excluded.",
    }


def select_models_by_target(
    artifact: Mapping[str, Any],
    feature_vector: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Return one candidate decision per target for an inactive research artifact."""
    vector = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    if vector.shape[1] != len(artifact["feature_names"]) or not np.isfinite(vector).all():
        raise ValueError("Target-specific selector received an invalid feature vector.")
    results: dict[str, dict[str, Any]] = {}
    for target in artifact["targets"]:
        estimators = artifact["competence_models_by_target"].get(target, {})
        if set(estimators) != set(artifact["model_ids"]):
            raise ValueError(f"Target-specific selector is incomplete for '{target}'.")
        expected_losses = {
            model_id: float(np.clip(estimators[model_id].predict(vector)[0], 0.0, 1.0))
            for model_id in artifact["model_ids"]
        }
        ranked = sorted(expected_losses, key=lambda model_id: (expected_losses[model_id], model_id))
        results[target] = {
            "selected_model_id": ranked[0],
            "expected_losses": expected_losses,
            "ranked_model_ids": ranked,
            "selection_rule": artifact["selection_rule"],
        }
    return results


def build_selector_artifact(
    *,
    model_ids: Sequence[str],
    targets: Sequence[str],
    planes: Sequence[str],
    feature_names: Sequence[str],
    models: Mapping[str, Pipeline],
    diagnostics: Mapping[str, Mapping[str, float]],
    model_entries: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if set(models) != set(model_ids):
        raise ValueError("A competence model is required for every base candidate.")
    artifact_dict = {
        "artifact_version": ARTIFACT_VERSION,
        "activation_status": "pending_verification",
        "model_ids": list(model_ids),
        "targets": list(targets),
        "planes": list(planes),
        "feature_names": list(feature_names),
        "competence_models": dict(models),
        "training_diagnostics": {key: dict(value) for key, value in diagnostics.items()},
        "model_entries": [dict(entry) for entry in model_entries],
        "protocol": dict(protocol),
        "selection_rule": "Select the architecture with the lowest learned expected mean Brier loss for the current study.",
        "inference_restriction": "Only uploaded MRI features and the five current base-model probability vectors are used. No case label, outcome or test-set metric is available at inference.",
    }
    return artifact_dict


def save_selector_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(dict(artifact), temporary)
    temporary.replace(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_selector_artifact(path: Path, require_active: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dynamic selector artifact is unavailable: {path}")
    artifact = joblib.load(path)
    if artifact.get("artifact_version") not in SUPPORTED_ARTIFACT_VERSIONS:
        raise ValueError(
            f"Dynamic selector artifact version '{artifact.get('artifact_version')}' is unsupported. "
            f"Supported versions: {sorted(SUPPORTED_ARTIFACT_VERSIONS)}."
        )
    if require_active and artifact.get("activation_status") != "active":
        raise RuntimeError("Dynamic selector has not passed verification and is not active.")
    ids = artifact.get("model_ids", [])
    if len(ids) != 5 or len(set(ids)) != 5 or set(ids) != set(artifact.get("competence_models", {})):
        raise ValueError("Dynamic selector does not contain one competence model for each of five candidates.")
    entries = artifact.get("model_entries", [])
    if len(entries) != 5 or {entry.get("id") for entry in entries} != set(ids):
        raise ValueError("Dynamic selector must have exactly five matching checkpoint entries.")
    return artifact


def select_model(artifact: Mapping[str, Any], feature_vector: Sequence[float]) -> dict[str, Any]:
    """Estimate current-study loss for every candidate and choose the minimum."""
    vector = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    if vector.shape[1] != len(artifact["feature_names"]) or not np.isfinite(vector).all():
        raise ValueError("Dynamic selector received an invalid feature vector.")
    expected_losses = {
        model_id: float(np.clip(artifact["competence_models"][model_id].predict(vector)[0], 0.0, 1.0))
        for model_id in artifact["model_ids"]
    }
    if not np.isfinite(list(expected_losses.values())).all():
        raise ValueError("The competence selector returned a non-finite expected loss.")
    ranked = sorted(expected_losses, key=lambda model_id: (expected_losses[model_id], model_id))
    contributions: list[dict[str, float | str]] = []
    selected_estimator = artifact["competence_models"][ranked[0]]
    is_gbr = False
    try:
        if "ridge" in selected_estimator.named_steps:
            scaler = selected_estimator.named_steps["scale"]
            ridge = selected_estimator.named_steps["ridge"]
            scaled = scaler.transform(vector)[0]
            names = artifact["feature_names"]
            terms = np.asarray(ridge.coef_, dtype=np.float64) * scaled
            contributions = [
                {"feature": str(name), "contribution_to_estimated_loss": float(term)}
                for name, term in zip(names, terms)
            ]
            contributions.sort(key=lambda item: abs(float(item["contribution_to_estimated_loss"])), reverse=True)
            contributions = contributions[:5]
        elif "gbr" in selected_estimator.named_steps:
            is_gbr = True
            gbr = selected_estimator.named_steps["gbr"]
            importances = gbr.feature_importances_
            names = artifact["feature_names"]
            top_idx = np.argsort(importances)[::-1][:5]
            contributions = [
                {"feature": str(names[i]), "feature_importance_mdi": float(importances[i])}
                for i in top_idx
            ]
    except (AttributeError, KeyError, TypeError, ValueError):
        contributions = []

    interpretation = (
        "MDI feature importances from the GBR competence estimator for the selected model. "
        "These terms explain the selector estimate only; they do not identify anatomy or lesions."
        if is_gbr
        else (
            "Positive values increase the selected model's estimated loss and negative values decrease it. "
            "These terms explain the selector estimate only; they do not identify anatomy or lesions."
        )
    )
    return {
        "selected_model_id": ranked[0],
        "expected_losses": expected_losses,
        "ranked_model_ids": ranked,
        "selection_rule": artifact["selection_rule"],
        "principal_factors": contributions,
        "factor_interpretation": interpretation,
    }
