import numpy as np
import pytest
import threading
from sklearn.linear_model import Ridge

from src.dynamic_selection.features import build_feature_vector, feature_names
from src.dynamic_selection.selector import (
    build_selector_artifact,
    load_selector_artifact,
    runtime_artifact_diagnostics,
    save_selector_artifact,
    select_model,
    train_competence_models,
)
from src.inference.dynamic_predictor import DynamicSelectionPredictor
from src.utils import config


MODEL_IDS = ["dms_resnet18", "dms_resnet50", "dms_densenet121", "dms_efficientnet_b0", "dms_swin_t"]
TARGETS = ["abnormal", "acl", "meniscus"]
PLANES = ["axial", "coronal", "sagittal"]


def _probabilities(value):
    return {
        model_id: np.asarray([value + index * 0.01, 0.2, 0.8 - index * 0.01])
        for index, model_id in enumerate(MODEL_IDS)
    }


def _volumes():
    return {plane: np.linspace(0, 1, 8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16) for plane in PLANES}


def test_feature_schema_is_current_case_only_and_stable():
    vector = build_feature_vector(_probabilities(0.1), _volumes(), MODEL_IDS, TARGETS, PLANES)
    assert vector.shape == (len(feature_names(MODEL_IDS, TARGETS, PLANES)),)
    assert np.isfinite(vector).all()
    # Different current model outputs change features without needing a label,
    # historic patient result, or model-level validation score.
    changed = build_feature_vector(_probabilities(0.6), _volumes(), MODEL_IDS, TARGETS, PLANES)
    assert not np.array_equal(vector, changed)


def test_learned_selector_changes_winner_for_controlled_current_case_patterns():
    x = np.zeros((20, 2), dtype=float)
    x[:, 0] = np.linspace(0, 1, len(x))
    models = {}
    for index, model_id in enumerate(MODEL_IDS):
        # Candidate zero is favoured at low x0 and candidate one at high x0.
        y = x[:, 0] if index == 0 else (1 - x[:, 0] if index == 1 else np.full(len(x), 0.8 + index * 0.01))
        models[model_id] = Ridge(alpha=1e-6).fit(x, y)
    artifact = {
        "model_ids": MODEL_IDS,
        "feature_names": ["first", "second"],
        "competence_models": models,
        "selection_rule": "lowest estimated loss",
    }
    low = select_model(artifact, [0.0, 0.0])
    high = select_model(artifact, [1.0, 0.0])
    assert low["selected_model_id"] == MODEL_IDS[0]
    assert high["selected_model_id"] == MODEL_IDS[1]


def test_artifact_round_trip_requires_explicit_activation(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(40, 4))
    losses = rng.uniform(size=(40, len(MODEL_IDS)))
    models, diagnostics = train_competence_models(features, losses, MODEL_IDS, seed=42)
    artifact = build_selector_artifact(
        model_ids=MODEL_IDS,
        targets=TARGETS,
        planes=PLANES,
        feature_names=[f"f{index}" for index in range(4)],
        models=models,
        diagnostics=diagnostics,
        model_entries=[{"id": model_id, "checkpoint": f"{model_id}.pth"} for model_id in MODEL_IDS],
        protocol={"selector_train": "synthetic unit-test fixture"},
    )
    path = tmp_path / "selector.joblib"
    save_selector_artifact(path, artifact)
    with pytest.raises(RuntimeError, match="not passed verification"):
        load_selector_artifact(path, require_active=True)
    artifact["activation_status"] = "active"
    save_selector_artifact(path, artifact)
    loaded = load_selector_artifact(path, require_active=True)
    assert set(loaded["competence_models"]) == set(MODEL_IDS)


def test_dynamic_predictor_calls_every_candidate_and_returns_the_selected_output(monkeypatch, tmp_path):
    """The orchestration layer must not use a static route or only one candidate."""
    predictor = object.__new__(DynamicSelectionPredictor)
    predictor.cfg = config()
    predictor.lock = threading.RLock()
    artifact_path = tmp_path / 'selector.joblib'
    artifact_path.write_bytes(b'unit-test-selector-fixture')
    monkeypatch.setattr('src.inference.dynamic_predictor.selector_artifact_path', lambda cfg: artifact_path)
    predictor.artifact = {
        "targets": TARGETS,
        "planes": PLANES,
        "model_ids": MODEL_IDS,
    }
    predictor.entries = {
        model_id: {"architecture": model_id.replace("dms_", ""), "checkpoint_sha256": f"hash-{index}"}
        for index, model_id in enumerate(MODEL_IDS)
    }
    predictor.selector_diagnostics = {"estimator_type": "ridge", "provenance_warnings": []}
    calls = []
    received_volumes = []
    probabilities = {
        model_id: [0.05 + index / 10, 0.15 + index / 10, 0.25 + index / 10]
        for index, model_id in enumerate(MODEL_IDS)
    }

    def fake_candidate(model_id, _volumes, _case_reference):
        calls.append(model_id)
        received_volumes.append(_volumes)
        return {
            "findings": [
                {"key": key, "probability": value, "finding": key.title()}
                for key, value in zip(TARGETS, probabilities[model_id])
            ],
            "architecture": model_id,
            "selected_models": {},
        }

    monkeypatch.setattr(predictor, "_predict_with_candidate", fake_candidate)
    monkeypatch.setattr("src.inference.dynamic_predictor.build_feature_vector", lambda *args: np.ones(23))
    monkeypatch.setattr(
        "src.inference.dynamic_predictor.select_model",
        lambda _artifact, _vector: {
            "selected_model_id": "dms_swin_t",
            "expected_losses": {model_id: float(index) / 10 for index, model_id in enumerate(MODEL_IDS)},
            "ranked_model_ids": ["dms_swin_t"] + [model_id for model_id in MODEL_IDS if model_id != "dms_swin_t"],
            "selection_rule": "test learned selector rule",
        },
    )

    study = {
        plane: np.arange(4 * 16 * 16, dtype=np.float32).reshape(4, 16, 16)
        for plane in PLANES
    }
    result = predictor.predict(study, "case-1")
    assert calls == MODEL_IDS
    assert result["dynamic_selection"]["models_evaluated"] == 5
    assert result["dynamic_selection"]["selected_model_id"] == "dms_swin_t"
    assert [finding["probability"] for finding in result["findings"]] == probabilities["dms_swin_t"]
    assert {finding["model_id"] for finding in result["findings"]} == {"dms_swin_t"}
    assert all(received is study for received in received_volumes)
    predictor.predict(study, 'case-2')
    assert calls == MODEL_IDS + MODEL_IDS


def test_extended_feature_schema_and_building():
    probs = _probabilities(0.2)
    vols = _volumes()
    names_std = feature_names(MODEL_IDS, TARGETS, PLANES, extended=False)
    names_ext = feature_names(MODEL_IDS, TARGETS, PLANES, extended=True)
    assert len(names_std) == 54
    assert len(names_ext) == 90
    assert names_ext[:54] == names_std

    vec_std = build_feature_vector(probs, vols, MODEL_IDS, TARGETS, PLANES, extended=False)
    vec_ext = build_feature_vector(probs, vols, MODEL_IDS, TARGETS, PLANES, extended=True)
    assert vec_std.shape == (54,)
    assert vec_ext.shape == (90,)
    assert np.allclose(vec_ext[:54], vec_std)
    assert np.isfinite(vec_ext).all()


def test_runtime_artifact_diagnostics_rejects_a_schema_mismatch():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(40, 90))
    losses = rng.uniform(0.05, 0.35, size=(40, len(MODEL_IDS)))
    models, diagnostics = train_competence_models(features, losses, MODEL_IDS, estimator_type="gbr", seed=42)
    artifact = build_selector_artifact(
        model_ids=MODEL_IDS,
        targets=TARGETS,
        planes=PLANES,
        feature_names=feature_names(MODEL_IDS, TARGETS, PLANES, extended=True),
        models=models,
        diagnostics=diagnostics,
        model_entries=[{"id": mid, "checkpoint": f"{mid}.pth"} for mid in MODEL_IDS],
        protocol={"selector": "GradientBoostingRegressor"},
    )
    artifact["extended_features"] = True
    artifact["estimator_type"] = "gbr"
    details = runtime_artifact_diagnostics(artifact)
    assert details["feature_count"] == 90
    assert details["estimator_type"] == "gbr"
    artifact["feature_names"] = artifact["feature_names"][:-1]
    with pytest.raises(ValueError, match="feature names"):
        runtime_artifact_diagnostics(artifact)


def test_gbr_competence_models_training_and_selection(tmp_path):
    rng = np.random.default_rng(123)
    features = rng.normal(size=(50, 90))
    losses = rng.uniform(0.05, 0.35, size=(50, len(MODEL_IDS)))
    models, diagnostics = train_competence_models(features, losses, MODEL_IDS, estimator_type="gbr", seed=42)

    assert set(models) == set(MODEL_IDS)
    for mid in MODEL_IDS:
        assert diagnostics[mid]["estimator_type"] == "GradientBoostingRegressor"
        assert diagnostics[mid]["selector_train_oof_mae"] > 0
        assert "n_estimators_used" in diagnostics[mid]

    artifact = build_selector_artifact(
        model_ids=MODEL_IDS,
        targets=TARGETS,
        planes=PLANES,
        feature_names=feature_names(MODEL_IDS, TARGETS, PLANES, extended=True),
        models=models,
        diagnostics=diagnostics,
        model_entries=[{"id": mid, "checkpoint": f"{mid}.pth"} for mid in MODEL_IDS],
        protocol={"test": "gbr_test"},
    )
    artifact["extended_features"] = True
    artifact["estimator_type"] = "gbr"
    artifact["activation_status"] = "active"

    path = tmp_path / "gbr_selector.joblib"
    save_selector_artifact(path, artifact)
    loaded = load_selector_artifact(path, require_active=True)
    assert loaded["artifact_version"] == "dynamic-competence-selector-v1"

    test_vec = rng.normal(size=90)
    decision = select_model(loaded, test_vec)
    assert decision["selected_model_id"] in MODEL_IDS
    assert len(decision["expected_losses"]) == len(MODEL_IDS)
    assert "MDI" in decision["factor_interpretation"]
    assert len(decision["principal_factors"]) <= 5
