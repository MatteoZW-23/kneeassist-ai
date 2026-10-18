"""Feature construction for case-specific, label-free selector inference."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))


def _pairwise_mean_abs_disagreement(matrix: np.ndarray) -> np.ndarray:
    """Mean absolute pairwise difference across all n*(n-1)/2 model pairs."""
    n = matrix.shape[0]
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(np.abs(matrix[i] - matrix[j]))
    return np.mean(diffs, axis=0) if diffs else np.zeros(matrix.shape[1], dtype=np.float64)


def _confidence_margins(matrix: np.ndarray) -> np.ndarray:
    """Distance of each model's probability from 0.5 (certainty proxy)."""
    return np.abs(matrix - 0.5).ravel()


def _consensus_features(matrix: np.ndarray) -> np.ndarray:
    """Consensus agreement ratio, plurality margin, and mean confidence per target."""
    n_models, n_targets = matrix.shape
    result = []
    for t in range(n_targets):
        col = matrix[:, t]
        majority = float(col.mean() >= 0.5)
        agree = np.sum(col >= 0.5) if majority else np.sum(col < 0.5)
        result.append(float(agree) / max(n_models, 1))
        result.append(float(abs(col.mean() - 0.5)))
        result.append(float(np.mean(np.abs(col - 0.5))))
    return np.asarray(result, dtype=np.float64)


def _soft_pool(matrix: np.ndarray, temperature: float = 4.0) -> np.ndarray:
    """Soft-max and soft-min pooling with temperature per target."""
    result = []
    t_pos = temperature
    for t in range(matrix.shape[1]):
        col = matrix[:, t]
        c_pos = col.max()
        soft_max = c_pos + np.log(np.mean(np.exp((col - c_pos) * t_pos))) / t_pos
        c_neg = (-col).max()
        soft_min_neg = c_neg + np.log(np.mean(np.exp((-col - c_neg) * t_pos))) / t_pos
        result.extend([float(np.clip(soft_max, 0.0, 1.0)), float(np.clip(-soft_min_neg, 0.0, 1.0))])
    return np.asarray(result, dtype=np.float64)


def _cross_model_entropy(matrix: np.ndarray) -> np.ndarray:
    """Second-order entropy across models for each target."""
    result = []
    for t in range(matrix.shape[1]):
        col = np.clip(matrix[:, t], 1e-7, 1.0 - 1e-7)
        col_norm = col / max(col.sum(), 1e-12)
        result.append(float(-np.sum(col_norm * np.log(col_norm))))
    return np.asarray(result, dtype=np.float64)


def feature_names(
    model_ids: Sequence[str],
    targets: Sequence[str],
    planes: Sequence[str],
    extended: bool = False,
) -> list[str]:
    """Return the fixed, auditable selector feature schema."""
    names: list[str] = []
    for model_id in model_ids:
        names.extend(f"probability::{model_id}::{target}" for target in targets)
    for model_id in model_ids:
        names.extend(f"entropy::{model_id}::{target}" for target in targets)
    for target in targets:
        names.append(f"disagreement_std::{target}")
        names.append(f"disagreement_range::{target}")
    for plane in planes:
        names.extend(
            [
                f"present::{plane}",
                f"slice_count_scaled::{plane}",
                f"height_scaled::{plane}",
                f"width_scaled::{plane}",
                f"intensity_spread::{plane}",
            ]
        )
    names.extend(["available_plane_fraction", "mean_slice_count_scaled", "mean_intensity_spread"])
    if extended:
        for model_id in model_ids:
            names.extend(f"margin::{model_id}::{target}" for target in targets)
        for target in targets:
            names.append(f"pairwise_disagree::{target}")
        for target in targets:
            names.extend(
                [
                    f"agreement_ratio::{target}",
                    f"plurality_margin::{target}",
                    f"mean_confidence::{target}",
                ]
            )
        for target in targets:
            names.extend([f"soft_max::{target}", f"soft_min::{target}"])
        for target in targets:
            names.append(f"cross_model_entropy::{target}")
    return names


def study_quality_features(volumes: Mapping[str, np.ndarray], planes: Sequence[str]) -> list[float]:
    """Return only image characteristics available for an unlabelled upload.

    Values are deliberately coarse.  They describe acquisition availability and
    intensity spread; they do not claim to recognise anatomy or image quality.
    """
    result: list[float] = []
    counts: list[float] = []
    spreads: list[float] = []
    for plane in planes:
        volume = volumes.get(plane)
        if volume is None:
            result.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        array = np.asarray(volume, dtype=np.float64)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError(f"Invalid raw volume supplied for dynamic-selection quality feature: {plane}.")
        slice_count, height, width = array.shape
        p01, p99 = np.percentile(array, [1, 99])
        spread = float(np.clip((p99 - p01) / (abs(p99) + abs(p01) + 1e-6), 0.0, 20.0))
        count_scaled = float(np.clip(slice_count / 128.0, 0.0, 4.0))
        result.extend(
            [
                1.0,
                count_scaled,
                float(np.clip(height / 512.0, 0.0, 4.0)),
                float(np.clip(width / 512.0, 0.0, 4.0)),
                spread,
            ]
        )
        counts.append(count_scaled)
        spreads.append(spread)
    result.extend(
        [
            len(counts) / max(len(planes), 1),
            float(np.mean(counts)) if counts else 0.0,
            float(np.mean(spreads)) if spreads else 0.0,
        ]
    )
    return result


def build_feature_vector(
    probabilities: Mapping[str, Sequence[float]],
    volumes: Mapping[str, np.ndarray],
    model_ids: Sequence[str],
    targets: Sequence[str],
    planes: Sequence[str],
    extended: bool = False,
) -> np.ndarray:
    """Build one selector feature vector from all base-model outputs and MRI data.

    Ground-truth labels, case identifiers and historic outcomes are deliberately
    excluded.  This exact function is used for selector training and uploads.
    """
    matrix = np.asarray([probabilities[model_id] for model_id in model_ids], dtype=np.float64)
    expected = (len(model_ids), len(targets))
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError(f"Expected finite base probabilities with shape {expected}, got {matrix.shape}.")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("Base-model probabilities must be in [0, 1].")
    result: list[float] = matrix.ravel().tolist()
    result.extend(_entropy(matrix).ravel().tolist())
    result.extend(np.std(matrix, axis=0, ddof=0).tolist())
    result.extend((np.max(matrix, axis=0) - np.min(matrix, axis=0)).tolist())
    result.extend(study_quality_features(volumes, planes))
    if extended:
        result.extend(_confidence_margins(matrix).tolist())
        result.extend(_pairwise_mean_abs_disagreement(matrix).tolist())
        result.extend(_consensus_features(matrix).tolist())
        result.extend(_soft_pool(matrix).tolist())
        result.extend(_cross_model_entropy(matrix).tolist())
    vector = np.asarray(result, dtype=np.float64)
    expected_width = len(feature_names(model_ids, targets, planes, extended=extended))
    if vector.shape != (expected_width,) or not np.isfinite(vector).all():
        raise ValueError("Dynamic-selector feature construction produced an invalid vector.")
    return vector


def brier_losses(probability_matrix: np.ndarray, labels: Sequence[float]) -> np.ndarray:
    """Per-model mean Brier loss across the three non-exclusive targets."""
    probabilities = np.asarray(probability_matrix, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != truth.size:
        raise ValueError("Probability matrix and target labels have incompatible shapes.")
    if not np.isfinite(probabilities).all() or not np.isfinite(truth).all():
        raise ValueError("Brier loss requires finite probabilities and labels.")
    return np.mean((probabilities - truth[None, :]) ** 2, axis=1)
