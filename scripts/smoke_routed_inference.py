"""Run a real three-plane sample through every registered routed model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import load_uploads
from src.inference.routed_predictor import RoutedPredictor
from src.selection.model_selector import select_models
from src.utils import config


def main() -> None:
    cfg = config()
    example = ROOT / "sample_cases" / "mrnet_1130.zip"
    volumes = load_uploads([(example.name, example.read_bytes())], cfg["preprocessing"])
    selection = select_models(list(volumes), [target["key"] for target in cfg["targets"]])
    if any(item["status"] != "Selected" for item in selection["selections"]):
        raise RuntimeError("Registry did not select every configured target.")

    predictor = RoutedPredictor(cfg)
    result = predictor.predict(volumes, "ROUTED-SMOKE", selection["selections"], uncertainty_samples=1)
    model_map = {item["key"]: item.get("model_architecture") for item in result["findings"]}
    expected = {"abnormal": "densenet121", "acl": "resnet18", "meniscus": "swin_t"}
    if model_map != expected:
        raise RuntimeError(f"Unexpected target routing: {model_map}")

    attention = predictor.explain(volumes, "acl", "sagittal")
    if attention.get("schema_version") != 2 or len(attention.get("indices", [])) != len(attention.get("regions", [])):
        raise RuntimeError("Grad-CAM payload is not current or is internally inconsistent.")

    print(json.dumps({
        "status": "passed",
        "models": model_map,
        "probabilities": {item["key"]: item["probability"] for item in result["findings"]},
        "attention_slices": len(attention["indices"]),
    }, indent=2))


if __name__ == "__main__":
    main()
