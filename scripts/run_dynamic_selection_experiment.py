"""Run an isolated all-five-model dynamic-selection experiment on MRNet.

This script deliberately does not modify the live registry or dashboard.  It
creates a new fixed split, trains five frozen-encoder study classifiers, fits
per-model calibration on a separate partition, trains a learned competence
selector on held-out development studies, and evaluates the complete system on
the locked MRNet validation studies.  Activation is a separate explicit step.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import read_manifest, split_and_check, volumes
from src.dynamic_selection.features import brier_losses, build_feature_vector, feature_names
from src.dynamic_selection.selector import (
    build_target_selector_artifact,
    build_selector_artifact,
    save_selector_artifact,
    select_models_by_target,
    train_competence_models,
    train_target_competence_models,
    select_model,
)
from src.evaluation.calibration import apply_calibration, balanced_thresholds, fit_calibration
from src.utils import ROOT, config, fingerprint, get_device, resolve, save_json, seed_everything


RUN_NAME = "dynamic_selection_v1"
PLANE_SPECIFIC_RUN_NAME = "plane_specific_dms_v1"
ARCHITECTURES = ("resnet18", "resnet50", "densenet121", "efficientnet_b0", "swin_t")
MODEL_IDS = tuple(f"dms_{architecture}" for architecture in ARCHITECTURES)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def labels(rows: list[dict], targets: list[dict]) -> np.ndarray:
    return np.asarray([[int(row[target["key"]]) for target in targets] for row in rows], dtype=np.int64)


def _stratification_labels(ids: list[str], by_id: dict[str, dict]) -> list[str]:
    combined = ["".join(by_id[study_id][key] for key in ("abnormal", "acl", "meniscus")) for study_id in ids]
    counts = {key: combined.count(key) for key in set(combined)}
    return combined if min(counts.values()) >= 2 else [by_id[study_id]["abnormal"] for study_id in ids]


def create_or_load_split(rows: list[dict], split_path: Path, csv_dir: Path, seed: int) -> dict[str, list[str]]:
    """Create the DMS split without borrowing final-evaluation labels for training."""
    if split_path.is_file():
        return json.loads(split_path.read_text(encoding="utf-8"))
    source = json.loads((ROOT / "data" / "splits" / "current_model_selection_splits.json").read_text(encoding="utf-8"))
    by_id = {row["study_id"]: row for row in rows}
    source_fit = list(source["fit"])
    fit_ids, base_tune_ids = train_test_split(
        source_fit,
        test_size=100,
        random_state=seed,
        stratify=_stratification_labels(source_fit, by_id),
    )
    split = {
        "fit": sorted(fit_ids),
        "tune": sorted(base_tune_ids),
        "calibration": sorted(source["calibration"]),
        "selector_train": sorted(source["tune"]),
        "official_valid": sorted(source["official_valid"]),
    }
    # Every study is assigned exactly once.  This makes accidental test-label
    # use by base models or the selector fail before any training begins.
    flattened = [study_id for values in split.values() for study_id in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(by_id):
        raise RuntimeError("Dynamic-selection split is incomplete or overlapping.")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(split_path, split)
    csv_dir.mkdir(parents=True, exist_ok=True)
    for role, ids in split.items():
        with (csv_dir / f"{role}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["study_id", "abnormal", "acl", "meniscus", "role"])
            writer.writeheader()
            for study_id in ids:
                row = by_id[study_id]
                writer.writerow({"study_id": study_id, "abnormal": row["abnormal"], "acl": row["acl"], "meniscus": row["meniscus"], "role": role})
    return split


def experiment_config() -> dict:
    cfg = copy.deepcopy(config())
    run_root = ROOT / "runs" / RUN_NAME
    cfg["dataset"]["splits"] = f"data/splits/{RUN_NAME}.json"
    cfg["model"]["architectures"] = list(ARCHITECTURES)
    cfg["model"]["selected_architecture"] = "dynamic_selector_pending"
    cfg["training"]["device"] = "auto"
    cfg["training"]["epochs"] = 6
    cfg["training"]["early_stopping_patience"] = 3
    cfg["training"]["batch_size"] = 1
    cfg["training"]["mixed_precision"] = True
    cfg["training"]["training_enabled"] = True
    cfg["lifecycle"]["training_enabled"] = True
    cfg["evaluation"]["directory"] = str(run_root / "base_models")
    cfg["paths"]["models"] = str(run_root / "models")
    cfg["paths"]["features"] = str(ROOT / "data" / "features" / RUN_NAME)
    cfg["run_directory"] = str(run_root)
    return cfg


def model_for_checkpoint(path: Path, device: Any) -> tuple[Any, dict]:
    from src.models.study_model import StudyModel
    saved = torch.load(path, map_location=device, weights_only=False)
    model = StudyModel(saved["architecture"], saved["config"], pretrained=False).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def probabilities_for_rows(
    checkpoint_path: Path,
    rows: list[dict],
    cfg: dict,
    device: Any,
    calibration: dict | None = None,
) -> np.ndarray:
    """Produce deterministic study predictions with no labels in the forward pass."""
    from src.data.preprocessing import prepare
    model, saved = model_for_checkpoint(checkpoint_path, device)
    calibration = calibration if calibration is not None else saved.get("calibration")
    results: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for row in rows:
                tensors = {
                    plane: prepare(volume, saved["config"]["preprocessing"])[0].to(device)
                    for plane, volume in volumes(row, cfg).items()
                }
                logits = model(tensors).float().cpu().numpy()[0]
                results.append(np.asarray(apply_calibration(logits, calibration), dtype=np.float64))
    finally:
        del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = np.stack(results)
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise RuntimeError("Base-model inference produced invalid calibrated probabilities.")
    return output


def calibrate_checkpoint(checkpoint_path: Path, calibration_rows: list[dict], cfg: dict, device: Any) -> dict:
    from src.data.preprocessing import prepare
    from src.training.train import checkpoint
    model, saved = model_for_checkpoint(checkpoint_path, device)
    logits: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for row in calibration_rows:
                tensors = {
                    plane: prepare(volume, saved["config"]["preprocessing"])[0].to(device)
                    for plane, volume in volumes(row, cfg).items()
                }
                logits.append(model(tensors).float().cpu().numpy()[0])
    finally:
        del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    z = np.stack(logits)
    y = labels(calibration_rows, cfg["targets"])
    calibration = fit_calibration(y, z, cfg["targets"])
    calibrated = apply_calibration(z, calibration)
    thresholds = balanced_thresholds(y, calibrated, cfg["targets"])
    saved["calibration"] = calibration
    saved["thresholds"] = thresholds
    saved["dynamic_selection"] = {
        "run": RUN_NAME,
        "calibration_role": "reserved calibration",
        "calibration_studies": len(calibration_rows),
        "selector_training_overlap": False,
        "final_evaluation_overlap": False,
    }
    checkpoint(checkpoint_path, saved)
    return {"calibration": calibration, "thresholds": thresholds, "checkpoint_sha256": sha256(checkpoint_path)}


def metric_report(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> dict:
    report: dict[str, dict] = {}
    for index, target in enumerate(("abnormal", "acl", "meniscus")):
        truth = y[:, index]
        score = p[:, index]
        predicted = (score >= thresholds[:, index]).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
        report[target] = {
            "auroc": float(roc_auc_score(truth, score)),
            "average_precision": float(average_precision_score(truth, score)),
            "accuracy": float(accuracy_score(truth, predicted)),
            "sensitivity": float(recall_score(truth, predicted, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "f1": float(f1_score(truth, predicted, zero_division=0)),
            "brier": float(np.mean((score - truth) ** 2)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
    keys = list(report)
    report["macro"] = {
        metric: float(np.mean([report[key][metric] for key in keys if report[key][metric] is not None]))
        for metric in ("auroc", "average_precision", "accuracy", "sensitivity", "specificity", "precision", "f1", "brier")
    }
    return report


def write_prediction_rows(path: Path, rows: list[dict], predictions: dict[str, np.ndarray], targets: list[dict], selected: list[str] | None = None) -> None:
    fieldnames = ["study_id"] + [target["key"] for target in targets]
    for model_id in predictions:
        fieldnames.extend(f"{model_id}::{target['key']}" for target in targets)
    if selected is not None:
        fieldnames.append("selected_model_id")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = {"study_id": row["study_id"]}
            output.update({target["key"]: row[target["key"]] for target in targets})
            for model_id, values in predictions.items():
                output.update({f"{model_id}::{target['key']}": float(values[index, target_index]) for target_index, target in enumerate(targets)})
            if selected is not None:
                output["selected_model_id"] = selected[index]
            writer.writerow(output)


def _load_prediction_csv(
    path: Path,
    model_ids: Sequence[str],
    targets: Sequence[dict],
) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    target_keys = [t["key"] for t in targets]
    predictions: dict[str, list] = {model_id: [] for model_id in model_ids}
    truth_list = []
    for row in rows:
        truth_list.append([int(row[k]) for k in target_keys])
        for model_id in model_ids:
            predictions[model_id].append([float(row[f"{model_id}::{k}"]) for k in target_keys])
    return rows, {m: np.asarray(v, dtype=np.float64) for m, v in predictions.items()}, np.asarray(truth_list, dtype=np.int64)


def build_features(
    rows: list[dict],
    predictions: dict[str, np.ndarray],
    cfg: dict,
    extended: bool = False,
    model_ids: Sequence[str] = MODEL_IDS,
) -> np.ndarray:
    target_keys = [target["key"] for target in cfg["targets"]]
    planes = cfg["preprocessing"]["planes"]
    vectors: list[np.ndarray] = []
    manifest_by_id: dict[str, dict] = {}
    try:
        manifest_rows = read_manifest(cfg)
        manifest_by_id = {r["study_id"]: r for r in manifest_rows}
    except Exception:
        pass
    for index, row in enumerate(rows):
        current = {model_id: values[index] for model_id, values in predictions.items()}
        study_id = row.get("study_id")
        target_row = manifest_by_id.get(study_id, row)
        try:
            vols = volumes(target_row, cfg)
        except Exception:
            vols = {}
        vectors.append(build_feature_vector(current, vols, model_ids, target_keys, planes, extended=extended))
    return np.stack(vectors)


def plane_specific_config(plane: str) -> dict:
    """Return an isolated, one-plane configuration using the fixed DMS split."""
    if plane not in ("axial", "coronal", "sagittal"):
        raise ValueError(f"Unsupported plane-specific experiment plane: {plane}")
    cfg = experiment_config()
    cfg["preprocessing"]["planes"] = [plane]
    run_root = ROOT / "runs" / PLANE_SPECIFIC_RUN_NAME / plane
    cfg["evaluation"]["directory"] = str(run_root / "base_models")
    cfg["paths"]["models"] = str(run_root / "models")
    cfg["paths"]["features"] = str(ROOT / "data" / "features" / PLANE_SPECIFIC_RUN_NAME / plane)
    cfg["run_directory"] = str(run_root)
    return cfg


def _selection_outputs(
    artifact: dict,
    rows: list[dict],
    predictions: dict[str, np.ndarray],
    cfg: dict,
    entries: list[dict],
    extended_features: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], list[dict[str, dict[str, float]]]]:
    """Select a candidate independently for each target without using labels."""
    target_keys = [target["key"] for target in cfg["targets"]]
    vectors = build_features(
        rows,
        predictions,
        cfg,
        extended=extended_features,
        model_ids=[entry["id"] for entry in entries],
    )
    entry_by_id = {entry["id"]: entry for entry in entries}
    selected_probabilities = np.empty((len(rows), len(target_keys)), dtype=np.float64)
    selected_thresholds = np.empty_like(selected_probabilities)
    selected_ids: list[dict[str, str]] = []
    estimated_losses: list[dict[str, dict[str, float]]] = []
    for index, vector in enumerate(vectors):
        decisions = select_models_by_target(artifact, vector)
        selected_ids.append({target: decisions[target]["selected_model_id"] for target in target_keys})
        estimated_losses.append({target: decisions[target]["expected_losses"] for target in target_keys})
        for target_index, target in enumerate(target_keys):
            model_id = decisions[target]["selected_model_id"]
            selected_probabilities[index, target_index] = predictions[model_id][index, target_index]
            selected_thresholds[index, target_index] = float(entry_by_id[model_id]["thresholds"][target])
    return selected_probabilities, selected_thresholds, selected_ids, estimated_losses


def _plane_model_id(plane: str, architecture: str) -> str:
    return f"plane_{plane}_{architecture}"


def run_plane_specific(resume: bool = False, estimator: str = "ridge", extended_features: bool = True) -> None:
    """Train 15 one-plane classifiers and evaluate target-specific selection.

    This is an isolated research benchmark. It shares the five-role split with
    the active experiment, writes no live registry entry, and remains pending
    review even if its internal development results look favourable.
    """
    root = ROOT / "runs" / PLANE_SPECIFIC_RUN_NAME
    root.mkdir(parents=True, exist_ok=True)
    planes = ("axial", "coronal", "sagittal")
    common_cfg = experiment_config()
    seed_everything(common_cfg["seed"], common_cfg["training"]["cpu_threads"])
    rows = read_manifest(common_cfg)
    split, leakage = split_and_check(rows, common_cfg)
    required_roles = {"fit", "tune", "calibration", "selector_train", "official_valid"}
    if set(split) != required_roles:
        raise RuntimeError(f"Unexpected fixed-split roles: {sorted(split)}")
    lookup = {row["study_id"]: row for row in rows}
    base_rows = [lookup[study_id] for study_id in split["fit"] + split["tune"]]
    calibration_rows = [lookup[study_id] for study_id in split["calibration"]]
    selector_rows = [lookup[study_id] for study_id in split["selector_train"]]
    final_rows = [lookup[study_id] for study_id in split["official_valid"]]
    target_keys = [target["key"] for target in common_cfg["targets"]]
    protocol = {
        "run": PLANE_SPECIFIC_RUN_NAME,
        "kind": "isolated_plane_specific_target_selection_benchmark",
        "candidate_architectures": list(ARCHITECTURES),
        "planes": list(planes),
        "candidate_count": len(ARCHITECTURES) * len(planes),
        "dataset": "MRNet-v1.0",
        "split_file": str(resolve(common_cfg["dataset"]["splits"]).relative_to(ROOT)),
        "split_sha256": sha256(resolve(common_cfg["dataset"]["splits"])),
        "roles": {role: len(ids) for role, ids in split.items()},
        "base_model_training": "Each plane-specific classifier uses fit only; its one-plane tune role controls early stopping.",
        "calibration": "Each plane-specific classifier and the equal-weight study fusion use the reserved calibration role only.",
        "selector_training": "Target-specific expected-Brier selectors use selector_train only.",
        "final_evaluation": "official_valid labels are excluded from all base, calibration, selector and threshold fitting.",
        "selection_rule": "For every plane and target, select the candidate with the lowest learned target-specific Brier loss; average the three selected calibrated plane scores.",
        "leakage_audit": leakage,
        "limitation": "MRNet patient IDs are unavailable in this copy and official_valid had prior project exposure; this is not an independent clinical test. The run is not active in the dashboard.",
    }
    save_json(root / "protocol.json", protocol)
    device = get_device(common_cfg)
    print(json.dumps({"stage": "plane_specific_start", "device": str(device), "planes": list(planes), "resume": resume}), flush=True)

    all_plane_data: dict[str, dict] = {}
    for plane in planes:
        cfg = plane_specific_config(plane)
        plane_root = root / plane
        plane_root.mkdir(parents=True, exist_ok=True)
        save_json(plane_root / "config.json", cfg)
        entries: list[dict] = []
        reports: dict[str, dict] = {}
        from src.training.train import train_arch
        for architecture in ARCHITECTURES:
            model_id = _plane_model_id(plane, architecture)
            print(json.dumps({"stage": "plane_train_base", "plane": plane, "architecture": architecture}), flush=True)
            report = train_arch(architecture, base_rows, split, cfg, device, resume=resume)
            if report is None:
                raise RuntimeError("Plane-specific training paused before completion; resume with --plane-specific --resume.")
            reports[model_id] = report
            checkpoint_path = resolve(cfg["paths"]["models"]) / f"{architecture}_best.pth"
            calibrated = calibrate_checkpoint(checkpoint_path, calibration_rows, cfg, device)
            entries.append({
                "id": model_id,
                "architecture": architecture,
                "plane": plane,
                "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": calibrated["checkpoint_sha256"],
                "calibration": calibrated["calibration"],
                "thresholds": calibrated["thresholds"],
                "supported_targets": target_keys,
                "supported_planes": [plane],
                "minimum_planes": 1,
            })
        save_json(plane_root / "base_model_reports.json", reports)
        save_json(plane_root / "model_entries.json", entries)

        def infer_partition(partition_rows: list[dict], name: str) -> dict[str, np.ndarray]:
            results: dict[str, np.ndarray] = {}
            for entry in entries:
                print(json.dumps({"stage": "plane_prediction_generation", "plane": plane, "partition": name, "model": entry["id"]}), flush=True)
                results[entry["id"]] = probabilities_for_rows(ROOT / entry["checkpoint"], partition_rows, cfg, device, entry["calibration"])
            return results

        calibration_predictions = infer_partition(calibration_rows, "calibration")
        selector_predictions = infer_partition(selector_rows, "selector_train")
        final_predictions = infer_partition(final_rows, "official_valid")
        write_prediction_rows(plane_root / "calibration_predictions.csv", calibration_rows, calibration_predictions, cfg["targets"])
        write_prediction_rows(plane_root / "selector_train_predictions.csv", selector_rows, selector_predictions, cfg["targets"])
        write_prediction_rows(plane_root / "final_predictions.csv", final_rows, final_predictions, cfg["targets"])

        selector_features = build_features(
            selector_rows,
            selector_predictions,
            cfg,
            extended=extended_features,
            model_ids=[entry["id"] for entry in entries],
        )
        observed_losses = {
            target: np.stack(
                [(selector_predictions[entry["id"]][:, target_index] - labels(selector_rows, cfg["targets"])[:, target_index]) ** 2 for entry in entries],
                axis=1,
            )
            for target_index, target in enumerate(target_keys)
        }
        target_models, target_diagnostics = train_target_competence_models(
            selector_features,
            observed_losses,
            [entry["id"] for entry in entries],
            estimator_type=estimator,
            seed=cfg["seed"],
        )
        artifact = build_target_selector_artifact(
            model_ids=[entry["id"] for entry in entries],
            targets=target_keys,
            planes=[plane],
            feature_names=feature_names([entry["id"] for entry in entries], target_keys, [plane], extended=extended_features),
            models_by_target=target_models,
            diagnostics_by_target=target_diagnostics,
            model_entries=entries,
            protocol=protocol,
            estimator_type=estimator,
            extended_features=extended_features,
        )
        artifact_path = plane_root / "selector" / "target_specific_selector.joblib"
        artifact_digest = save_selector_artifact(artifact_path, artifact)
        cal_selected, _, cal_ids, _ = _selection_outputs(artifact, calibration_rows, calibration_predictions, cfg, entries, extended_features)
        final_selected, final_selected_thresholds, final_ids, final_losses = _selection_outputs(artifact, final_rows, final_predictions, cfg, entries, extended_features)
        all_plane_data[plane] = {
            "cfg": cfg,
            "entries": entries,
            "reports": reports,
            "calibration_predictions": calibration_predictions,
            "final_predictions": final_predictions,
            "calibration_selected": cal_selected,
            "final_selected": final_selected,
            "final_thresholds": final_selected_thresholds,
            "final_ids": final_ids,
            "final_losses": final_losses,
            "artifact": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": artifact_digest,
            "diagnostics": target_diagnostics,
        }
        save_json(plane_root / "selection_summary.json", {
            "plane": plane,
            "artifact": all_plane_data[plane]["artifact"],
            "artifact_sha256": artifact_digest,
            "selection_frequency": {
                target: {model_id: sum(row[target] == model_id for row in final_ids) for model_id in [entry["id"] for entry in entries]}
                for target in target_keys
            },
            "selector_diagnostics": target_diagnostics,
        })

    calibration_truth = labels(calibration_rows, common_cfg["targets"])
    final_truth = labels(final_rows, common_cfg["targets"])
    fused_calibration = np.mean(np.stack([all_plane_data[plane]["calibration_selected"] for plane in planes]), axis=0)
    fused_final = np.mean(np.stack([all_plane_data[plane]["final_selected"] for plane in planes]), axis=0)
    fusion_threshold_map = balanced_thresholds(calibration_truth, fused_calibration, common_cfg["targets"])
    fusion_thresholds = np.tile(np.asarray([fusion_threshold_map[target] for target in target_keys]), (len(final_rows), 1))
    dms_metrics = metric_report(final_truth, fused_final, fusion_thresholds)

    fixed_models: dict[str, dict] = {}
    for architecture in ARCHITECTURES:
        calibration_scores = np.mean(np.stack([all_plane_data[plane]["calibration_predictions"][_plane_model_id(plane, architecture)] for plane in planes]), axis=0)
        final_scores = np.mean(np.stack([all_plane_data[plane]["final_predictions"][_plane_model_id(plane, architecture)] for plane in planes]), axis=0)
        threshold_map = balanced_thresholds(calibration_truth, calibration_scores, common_cfg["targets"])
        thresholds = np.tile(np.asarray([threshold_map[target] for target in target_keys]), (len(final_rows), 1))
        fixed_models[architecture] = metric_report(final_truth, final_scores, thresholds)

    final_rows_out = []
    for index, row in enumerate(final_rows):
        record = {"study_id": row["study_id"]}
        record.update({target: int(final_truth[index, target_index]) for target_index, target in enumerate(target_keys)})
        record.update({f"plane_specific_dms::{target}": float(fused_final[index, target_index]) for target_index, target in enumerate(target_keys)})
        for plane in planes:
            for target in target_keys:
                record[f"selected::{plane}::{target}"] = all_plane_data[plane]["final_ids"][index][target]
        final_rows_out.append(record)
    with (root / "final_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows_out[0]))
        writer.writeheader()
        writer.writerows(final_rows_out)

    evaluation = {
        "dataset": "MRNet-v1.0",
        "partition": "official_valid held out from all plane-model, calibration, selector and fusion-threshold fitting",
        "limitation": protocol["limitation"],
        "planes": list(planes),
        "candidate_architectures": list(ARCHITECTURES),
        "candidate_count": len(ARCHITECTURES) * len(planes),
        "fixed_plane_fusion_models": fixed_models,
        "target_specific_plane_dms": dms_metrics,
        "fusion": {"method": "equal mean of the three target-selected calibrated plane probabilities", "thresholds": fusion_threshold_map, "fit_partition": "reserved calibration"},
        "plane_selector_artifacts": {plane: {"path": data["artifact"], "sha256": data["artifact_sha256"], "diagnostics": data["diagnostics"]} for plane, data in all_plane_data.items()},
        "selection_frequency": {plane: {
            target: {model_id: sum(row[target] == model_id for row in all_plane_data[plane]["final_ids"]) for model_id in [entry["id"] for entry in all_plane_data[plane]["entries"]]}
            for target in target_keys
        } for plane in planes},
        "activation_status": "pending_verification",
        "not_deployed_reason": "A new plane-specific selector must show robust improvement over fixed baselines and pass independent validation before dashboard activation.",
    }
    save_json(root / "final_evaluation.json", evaluation)
    save_json(root / "completion_status.json", {
        "status": "completed",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "activation": "pending_verification",
        "candidate_count": len(ARCHITECTURES) * len(planes),
        "macro_auroc": dms_metrics["macro"]["auroc"],
        "artifact_paths": {plane: data["artifact"] for plane, data in all_plane_data.items()},
    })
    print(json.dumps({"stage": "plane_specific_completed", "macro_auroc": dms_metrics["macro"]["auroc"], "activation": "pending_verification"}), flush=True)


def run(
    activate: bool = False,
    estimator: str = "ridge",
    extended_features: bool = False,
    retrain_selector_only: bool = False,
) -> None:
    cfg = experiment_config()
    seed_everything(cfg["seed"], cfg["training"]["cpu_threads"])
    run_root = ROOT / "runs" / RUN_NAME
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    split_path = resolve(cfg["dataset"]["splits"])
    protocol_path = run_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}

    if retrain_selector_only:
        sel_path = run_root / "selector_train_predictions.csv"
        fin_path = run_root / "final_predictions.csv"
        entries_path = run_root / "model_entries.json"
        if not sel_path.is_file() or not fin_path.is_file():
            raise FileNotFoundError("Saved prediction CSVs are required for --retrain-selector-only.")
        selector_rows, selector_predictions, selector_truth = _load_prediction_csv(sel_path, MODEL_IDS, cfg["targets"])
        final_rows, final_predictions, final_truth = _load_prediction_csv(fin_path, MODEL_IDS, cfg["targets"])
        entries = json.loads(entries_path.read_text(encoding="utf-8")) if entries_path.is_file() else []
        print(json.dumps({"stage": "retrain_selector_from_saved_predictions", "estimator": estimator, "extended_features": extended_features}), flush=True)
    else:
        rows = read_manifest(cfg)
        split = create_or_load_split(rows, split_path, ROOT / "data" / "splits" / RUN_NAME, cfg["seed"])
        checked_split, leakage = split_and_check(rows, cfg)
        if checked_split != split:
            raise RuntimeError("Persisted dynamic-selection split differs from the checked split.")
        required_roles = {"fit", "tune", "calibration", "selector_train", "official_valid"}
        if set(split) != required_roles:
            raise RuntimeError(f"Unexpected dynamic-selection roles: {sorted(split)}")
        lookup = {row["study_id"]: row for row in rows}
        base_rows = [lookup[study_id] for study_id in split["fit"] + split["tune"]]
        calibration_rows = [lookup[study_id] for study_id in split["calibration"]]
        selector_rows = [lookup[study_id] for study_id in split["selector_train"]]
        final_rows = [lookup[study_id] for study_id in split["official_valid"]]
        protocol = {
            "run": RUN_NAME,
            "candidate_architectures": list(ARCHITECTURES),
            "candidate_model_ids": list(MODEL_IDS),
            "dataset": "MRNet-v1.0",
            "split_file": str(split_path.relative_to(ROOT)),
            "split_sha256": sha256(split_path),
            "roles": {role: len(ids) for role, ids in split.items()},
            "base_model_training": "fit only; tuning is used only for candidate early stopping",
            "calibration": "reserved calibration role only",
            "selector_training": "selector_train only; base-model probabilities are generated after base-model fitting",
            "final_evaluation": "official_valid never supplies labels to base-model or selector fitting",
            "observed_competence_target": "per-model mean Brier loss across abnormal, ACL and meniscus labels",
            "selector": f"Competence estimator ({estimator}); alpha/trees tuned via CV within selector_train",
            "leakage_audit": leakage,
            "limitation": "MRNet patient IDs are unavailable in this copy and official_valid had prior development exposure; this is not an independent clinical test.",
        }
        save_json(run_root / "protocol.json", protocol)
        device = get_device(cfg)
        print(json.dumps({"stage": "starting", "device": str(device), "roles": protocol["roles"]}), flush=True)

        reports: dict[str, dict] = {}
        entries = []
        from src.training.train import train_arch
        for architecture, model_id in zip(ARCHITECTURES, MODEL_IDS):
            print(json.dumps({"stage": "train_base_model", "architecture": architecture}), flush=True)
            report = train_arch(architecture, base_rows, split, cfg, device, resume=True)
            reports[model_id] = report
            checkpoint_path = resolve(cfg["paths"]["models"]) / f"{architecture}_best.pth"
            calibrated = calibrate_checkpoint(checkpoint_path, calibration_rows, cfg, device)
            entries.append(
                {
                    "id": model_id,
                    "architecture": architecture,
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
                    "checkpoint_sha256": calibrated["checkpoint_sha256"],
                    "calibration": calibrated["calibration"],
                    "thresholds": calibrated["thresholds"],
                    "supported_targets": [target["key"] for target in cfg["targets"]],
                    "supported_planes": list(cfg["preprocessing"]["planes"]),
                    "minimum_planes": 1,
                }
            )
        save_json(run_root / "base_model_reports.json", reports)
        save_json(run_root / "model_entries.json", entries)

        selector_predictions = {}
        final_predictions = {}
        for entry in entries:
            checkpoint_path = ROOT / entry["checkpoint"]
            print(json.dumps({"stage": "selector_prediction_generation", "model": entry["id"]}), flush=True)
            selector_predictions[entry["id"]] = probabilities_for_rows(checkpoint_path, selector_rows, cfg, device, entry["calibration"])
            final_predictions[entry["id"]] = probabilities_for_rows(checkpoint_path, final_rows, cfg, device, entry["calibration"])
        write_prediction_rows(run_root / "selector_train_predictions.csv", selector_rows, selector_predictions, cfg["targets"])
        selector_truth = labels(selector_rows, cfg["targets"])
        final_truth = labels(final_rows, cfg["targets"])

    selector_features = build_features(selector_rows, selector_predictions, cfg, extended=extended_features)
    selector_losses = np.stack(
        [
            brier_losses(
                np.stack([selector_predictions[model_id][index] for model_id in MODEL_IDS]),
                selector_truth[index],
            )
            for index in range(len(selector_rows))
        ]
    )
    models, diagnostics = train_competence_models(
        selector_features, selector_losses, MODEL_IDS, estimator_type=estimator, seed=cfg["seed"]
    )
    target_keys = [target["key"] for target in cfg["targets"]]
    artifact = build_selector_artifact(
        model_ids=MODEL_IDS,
        targets=target_keys,
        planes=cfg["preprocessing"]["planes"],
        feature_names=feature_names(MODEL_IDS, target_keys, cfg["preprocessing"]["planes"], extended=extended_features),
        models=models,
        diagnostics=diagnostics,
        model_entries=entries,
        protocol=protocol,
    )
    artifact["extended_features"] = extended_features
    artifact["estimator_type"] = estimator
    artifact_path = run_root / "selector" / "dynamic_selector.joblib"
    artifact_digest = save_selector_artifact(artifact_path, artifact)

    final_features = build_features(final_rows, final_predictions, cfg, extended=extended_features)
    selected_ids: list[str] = []
    estimated_losses: list[dict[str, float]] = []
    selected_probability_rows: list[np.ndarray] = []
    selected_threshold_rows: list[np.ndarray] = []
    for index, vector in enumerate(final_features):
        decision = select_model(artifact, vector)
        selected_id = decision["selected_model_id"]
        selected_ids.append(selected_id)
        estimated_losses.append(decision["expected_losses"])
        selected_probability_rows.append(final_predictions[selected_id][index])
        entry = next((item for item in entries if item["id"] == selected_id), None)
        thresholds_map = entry["thresholds"] if entry else {t: 0.5 for t in target_keys}
        selected_threshold_rows.append(np.asarray([thresholds_map[t] for t in target_keys], dtype=np.float64))
    selected_probabilities = np.stack(selected_probability_rows)
    selected_thresholds = np.stack(selected_threshold_rows)
    final_truth = np.asarray(final_truth, dtype=np.int64)

    final_base_metrics = {}
    for entry in entries:
        t_arr = np.tile(np.asarray([entry["thresholds"][t] for t in target_keys]), (len(final_rows), 1))
        final_base_metrics[entry["id"]] = metric_report(final_truth, final_predictions[entry["id"]], t_arr)

    dms_metrics = metric_report(final_truth, selected_probabilities, selected_thresholds)
    actual_losses = np.asarray(
        [
            brier_losses(
                np.stack([final_predictions[model_id][index] for model_id in MODEL_IDS]),
                final_truth[index],
            )
            for index in range(len(final_rows))
        ]
    )
    selected_indices = np.asarray([MODEL_IDS.index(model_id) for model_id in selected_ids])
    selection_frequency = {model_id: int(selected_ids.count(model_id)) for model_id in MODEL_IDS}
    mean_selected_brier = float(np.mean(actual_losses[np.arange(len(final_rows)), selected_indices]))
    mean_oracle_brier = float(np.mean(np.min(actual_losses, axis=1)))

    evaluation = {
        "dataset": "MRNet-v1.0",
        "partition": "official_valid held out from DMS fitting",
        "limitation": protocol.get("limitation", ""),
        "base_models": final_base_metrics,
        "dynamic_selection": dms_metrics,
        "selection_frequency": selection_frequency,
        "mean_selected_observed_brier": mean_selected_brier,
        "mean_oracle_observed_brier": mean_oracle_brier,
        "mean_estimated_selected_brier": float(
            np.mean([estimated_losses[index][model_id] for index, model_id in enumerate(selected_ids)])
        ),
        "selector_artifact_sha256": artifact_digest,
        "selector_training_diagnostics": diagnostics,
        "selection_varied_on_final_partition": len([count for count in selection_frequency.values() if count]) > 1,
        "activation_status": "active" if activate else "pending_verification",
        "estimator_type": estimator,
        "extended_features": extended_features,
    }
    save_json(run_root / "final_evaluation.json", evaluation)
    write_prediction_rows(run_root / "final_predictions.csv", final_rows, final_predictions, cfg["targets"], selected_ids)
    with (run_root / "selection_decisions.json").open("w", encoding="utf-8") as stream:
        json.dump(
            [
                {
                    "study_id": row["study_id"],
                    "selected_model_id": selected_ids[index],
                    "expected_losses": estimated_losses[index],
                    "observed_losses_for_evaluation_only": {
                        model_id: float(actual_losses[index, candidate_index])
                        for candidate_index, model_id in enumerate(MODEL_IDS)
                    },
                }
                for index, row in enumerate(final_rows)
            ],
            stream,
            indent=2,
        )

    if activate:
        artifact["activation_status"] = "active"
        artifact["activation_reason"] = "Explicitly activated after verification."
        artifact_digest = save_selector_artifact(artifact_path, artifact)
        evaluation["activation_status"] = "active"
        evaluation["selector_artifact_sha256"] = artifact_digest
        save_json(run_root / "final_evaluation.json", evaluation)

        # Update model registry
        reg_path = ROOT / cfg.get("dynamic_selection", {}).get("registry", "model_registry/dynamic_selection.json")
        if reg_path.is_file():
            reg_data = json.loads(reg_path.read_text(encoding="utf-8"))
            reg_data["selector_sha256"] = artifact_digest
            save_json(reg_path, reg_data)

    save_json(
        run_root / "completion_status.json",
        {
            "status": "completed",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifact": str(artifact_path),
            "activation": artifact["activation_status"],
            "estimator": estimator,
            "extended_features": extended_features,
            "mean_selected_brier": mean_selected_brier,
            "mean_oracle_brier": mean_oracle_brier,
            "macro_auroc": dms_metrics["macro"]["auroc"],
        },
    )

    print("\n" + "=" * 60)
    print(f"Dynamic Selection Run Completed [{estimator.upper()}, Extended: {extended_features}]")
    print("=" * 60)
    print(f"Selected Mean Brier: {mean_selected_brier:.5f} (Oracle: {mean_oracle_brier:.5f}, Gap: {mean_selected_brier - mean_oracle_brier:.5f})")
    print(f"Macro AUROC:        {dms_metrics['macro']['auroc']:.4f}")
    print(f"Macro F1:           {dms_metrics['macro']['f1']:.4f}")
    print(f"Selection frequency: {selection_frequency}")
    print(f"Artifact SHA-256:   {artifact_digest}")
    print(f"Activation status:  {artifact['activation_status']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dynamic selection experiment or retrain selector.")
    parser.add_argument("--activate", action="store_true", help="Mark the selector artifact active after the run.")
    parser.add_argument("--estimator", choices=["ridge", "gbr"], default="ridge", help="Competence estimator: ridge or gbr. Default: ridge.")
    parser.add_argument("--extended-features", action="store_true", help="Use extended label-free feature set (90 features).")
    parser.add_argument("--retrain-selector-only", action="store_true", help="Retrain selector using saved predictions without re-running base models.")
    parser.add_argument("--plane-specific", action="store_true", help="Run the isolated 15-model one-plane target-specific DMS benchmark.")
    parser.add_argument("--resume", action="store_true", help="Resume completed plane-specific candidate training safely.")
    args = parser.parse_args()
    if args.plane_specific:
        if args.activate:
            parser.error("Plane-specific results cannot be activated directly; they require a completed independent review.")
        if args.retrain_selector_only:
            parser.error("Plane-specific mode needs all 15 plane-specialised candidate checkpoints.")
        run_plane_specific(resume=args.resume, estimator=args.estimator, extended_features=True)
    else:
        if args.resume:
            parser.error("--resume is available only with --plane-specific.")
        run(
            activate=args.activate,
            estimator=args.estimator,
            extended_features=args.extended_features,
            retrain_selector_only=args.retrain_selector_only,
        )
