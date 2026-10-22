"""Run a locked MRNet ResNet-18/EfficientNet-B0 reproducibility experiment.

The script never writes to models/, model_registry/, or the dashboard registry.
It uses the fixed MRNet study split, selects only from the two prespecified
architectures by tuning macro AUROC, calibrates only on the reserved calibration
partition, and evaluates once on the official MRNet validation partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Python sets sys.path to this scripts/ directory when a file is launched
# directly on Windows. Add the project root explicitly so the src package is
# resolved in both a terminal and the hidden Start-Process runner.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import read_manifest, split_and_check, volumes
from src.evaluation.calibration import apply_calibration, balanced_thresholds, fit_calibration, reliability
from src.evaluation.evaluate import evaluate
from src.training.finetune import fit
from src.training.train import checkpoint
from src.utils import config, get_device, resolve, save_json, seed_everything


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def status(path: Path, stage: str, **details: object) -> None:
    save_json(
        path,
        {
            "stage": stage,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "pid": __import__("os").getpid(),
            **details,
        },
    )


def calibration_logits(checkpoint_path: Path, rows: list[dict[str, str]], cfg: dict, device: torch.device) -> np.ndarray:
    """Return deterministic pre-calibration logits for the reserved cases."""
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from src.models.study_model import StudyModel
    from src.data.preprocessing import prepare

    model = StudyModel(saved["architecture"], saved["config"], pretrained=False).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for position, row in enumerate(rows, start=1):
            tensors = {
                plane: prepare(volume, saved["config"]["preprocessing"])[0].to(device)
                for plane, volume in volumes(row, cfg).items()
            }
            logits.append(model(tensors).float().cpu().numpy()[0])
            if position == 1 or position % 20 == 0 or position == len(rows):
                print(f"Calibration: {position}/{len(rows)}", flush=True)
    result = np.stack(logits)
    if not np.isfinite(result).all():
        raise RuntimeError("Calibration logits include non-finite values.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Locked experiment configuration.")
    args = parser.parse_args()
    cfg = config(args.config)
    run_dir = resolve(cfg["run_directory"])
    result_dir = resolve(cfg["evaluation"]["directory"])
    completion = run_dir / "completion_status.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        status(completion, "preflight")
        device = get_device(cfg)
        if device.type != "cuda":
            raise RuntimeError("CUDA was not available; this locked run requires the verified GPU path.")
        seed_everything(cfg["seed"], cfg["training"]["cpu_threads"])
        rows = read_manifest(cfg)
        split, leakage_audit = split_and_check(rows, cfg)
        required_roles = {"fit", "tune", "calibration", "official_valid"}
        if set(split) != required_roles:
            raise RuntimeError(f"Expected fixed split roles {sorted(required_roles)}, got {sorted(split)}.")
        if set().union(*[set(split[key]) for key in split]) != {row["study_id"] for row in rows}:
            raise RuntimeError("Fixed split does not cover every MRNet study exactly once.")
        if set(split["fit"]) & set(split["calibration"]):
            raise RuntimeError("Calibration studies overlap the fitting partition.")
        if set(split["tune"]) & set(split["calibration"]):
            raise RuntimeError("Calibration studies overlap the tuning partition.")

        protocol = {
            "version": 1,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "MRNet-v1.0 only",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "config": str(Path(args.config)),
            "config_sha256": sha256(Path(args.config)),
            "split_file": cfg["dataset"]["splits"],
            "split_sha256": sha256(resolve(cfg["dataset"]["splits"])),
            "counts": {key: len(value) for key, value in split.items()},
            "prespecified_candidates": cfg["model"]["architectures"],
            "selection": "Highest internal tuning macro AUROC only; never a patient-specific score.",
            "calibration": "Positive-slope Platt scaling and Youden thresholding on the reserved calibration partition only.",
            "evaluation": "One pass over the MRNet official validation split after selection and calibration.",
            "not_used": [
                "RSNA competition material",
                "Osteoarthritis radiographs",
                "fastMRI+ annotations or single-coil files pending image-label identity verification",
                "Any external evaluation metric for model selection",
            ],
            "limits": [
                "MRNet patient linkage is unavailable in this local copy; study separation does not prove patient independence.",
                "The official MRNet validation cohort was used previously during development; this is a reproduction evaluation, not a new external test.",
                "Calibration diagnostics are in-sample on a small development partition and do not establish clinical probability calibration.",
                "This run does not validate additional findings such as MCL injury, fracture, effusion or osteoarthritis.",
            ],
        }
        save_json(run_dir / "protocol.json", protocol)
        save_json(run_dir / "leakage_audit.json", leakage_audit)
        print(json.dumps({"preflight": "passed", "device": str(device), "counts": protocol["counts"]}), flush=True)

        reports: list[dict] = []
        for architecture in cfg["model"]["architectures"]:
            status(completion, "training", architecture=architecture, completed_candidates=len(reports))
            report = fit(architecture, cfg, rows, split, device)
            reports.append(report)
            print(json.dumps({"completed_candidate": architecture, "macro_auroc": report["metrics"]["macro"]["auroc"]}), flush=True)

        winner = max(reports, key=lambda report: report["metrics"]["macro"]["auroc"])
        source = resolve(cfg["paths"]["models"]) / f"{winner['architecture']}_best.pth"
        selected = resolve(cfg["evaluation"]["checkpoint"])
        selected.parent.mkdir(parents=True, exist_ok=True)
        temp = selected.with_suffix(".tmp")
        shutil.copy2(source, temp)
        temp.replace(selected)
        save_json(
            result_dir / "model_comparison.json",
            {
                "criterion": "Internal tuning macro AUROC only",
                "winner": winner["architecture"],
                "models": reports,
                "dataset": "MRNet-v1.0",
                "selection_disclosure": "The winner was chosen once after training, before calibration and official-validation evaluation. It is not chosen by an uploaded case score.",
            },
        )

        status(completion, "calibrating", winner=winner["architecture"])
        lookup = {row["study_id"]: row for row in rows}
        calibration_rows = [lookup[study_id] for study_id in split["calibration"]]
        y_calibration = np.asarray(
            [[int(row[target["key"]]) for target in cfg["targets"]] for row in calibration_rows], dtype=np.int64
        )
        logits = calibration_logits(selected, calibration_rows, cfg, device)
        calibration = fit_calibration(y_calibration, logits, cfg["targets"])
        uncalibrated = apply_calibration(logits)
        calibrated = apply_calibration(logits, calibration)
        thresholds = balanced_thresholds(y_calibration, calibrated, cfg["targets"])
        saved = torch.load(selected, map_location=device, weights_only=False)
        saved["calibration"] = calibration
        saved["thresholds"] = thresholds
        saved["calibration_partition"] = {
            "role": "reserved calibration",
            "studies": len(calibration_rows),
            "in_sample_warning": "Calibration fit and diagnostics use the same reserved development cases.",
        }
        checkpoint(selected, saved)
        save_json(
            result_dir / "calibration.json",
            {
                "calibration": calibration,
                "thresholds": thresholds,
                "before": reliability(y_calibration, uncalibrated, cfg["targets"]),
                "after": reliability(y_calibration, calibrated, cfg["targets"]),
                "warning": "In-sample calibration diagnostics; they are not held-out or clinical reliability estimates.",
            },
        )

        status(completion, "evaluating", winner=winner["architecture"])
        evaluate(cfg, rows, split)
        metrics = json.loads((result_dir / "final" / "metrics.json").read_text(encoding="utf-8"))
        final_checkpoint_sha256 = sha256(selected)
        status(
            completion,
            "completed",
            winner=winner["architecture"],
            checkpoint=str(selected),
            checkpoint_sha256=final_checkpoint_sha256,
            internal_tuning_macro_auroc=winner["metrics"]["macro"]["auroc"],
            official_validation_macro_auroc=metrics["metrics"]["macro"]["auroc"],
            disclaimer="Unpromoted reproduction artifact. The dashboard registry and active checkpoints were not changed.",
        )
        print(json.dumps({"status": "completed", "winner": winner["architecture"], "checkpoint_sha256": final_checkpoint_sha256}), flush=True)
        return 0
    except Exception as exc:
        status(completion, "failed", error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
