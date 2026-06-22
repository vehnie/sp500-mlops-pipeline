from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.serving.app import create_app
from sp500_mlops_pipeline.serving.model_loader import (
    DEFAULT_TRACKING_URI,
    FEATURE_NAMES,
    MODEL_ALIAS,
    MODEL_NAME,
    ModelService,
    load_candidate_champion_model,
)


VALID_PAYLOAD = {
    "simple_return": 0.001,
    "log_return": 0.001,
    "sma_10": 6000.0,
    "sma_20": 5980.0,
    "sma_ratio_10": 1.0033,
    "rsi_14": 55.0,
    "volatility_10": 0.012,
    "volume_change": 0.05,
}


class FakeModel:
    """Minimal immutable sklearn-like classifier for API tests."""

    classes_ = np.array([0, 1])

    def __init__(self):
        self.received_columns = None

    def predict(self, model_input):
        self.received_columns = model_input.columns.tolist()
        return np.array([0])

    def predict_proba(self, model_input):
        self.received_columns = model_input.columns.tolist()
        return np.array([[0.6, 0.4]])


def make_service() -> ModelService:
    return ModelService(
        model=FakeModel(),
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_version="1",
        model_family="logistic_regression",
        feature_names=FEATURE_NAMES,
    )


def test_health_returns_http_200() -> None:
    with TestClient(create_app(model_loader=make_service)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


def test_model_info_returns_http_200() -> None:
    with TestClient(create_app(model_loader=make_service)) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_version"] == "1"
    assert response.json()["feature_names"] == FEATURE_NAMES
    assert response.json()["expected_number_of_features"] == 8


def test_predict_accepts_valid_input_and_returns_probabilities() -> None:
    with TestClient(create_app(model_loader=make_service)) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == 0
    assert result["prediction_label"] == "down"
    assert 0.0 <= result["probability_down"] <= 1.0
    assert 0.0 <= result["probability_up"] <= 1.0
    assert result["probability_down"] + result["probability_up"] == pytest.approx(1.0)


def test_predict_rejects_missing_feature() -> None:
    payload = VALID_PAYLOAD.copy()
    payload.pop("volume_change")

    with TestClient(create_app(model_loader=make_service)) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_nan_and_infinity() -> None:
    app = create_app(model_loader=make_service)
    with TestClient(app) as client:
        nan_response = client.post(
            "/predict",
            content=(
                '{"simple_return":NaN,"log_return":0.001,"sma_10":6000.0,'
                '"sma_20":5980.0,"sma_ratio_10":1.0033,"rsi_14":55.0,'
                '"volatility_10":0.012,"volume_change":0.05}'
            ),
            headers={"Content-Type": "application/json"},
        )
        infinity_response = client.post(
            "/predict",
            content=(
                '{"simple_return":Infinity,"log_return":0.001,'
                '"sma_10":6000.0,"sma_20":5980.0,'
                '"sma_ratio_10":1.0033,"rsi_14":55.0,'
                '"volatility_10":0.012,"volume_change":0.05}'
            ),
            headers={"Content-Type": "application/json"},
        )

    assert nan_response.status_code == 422
    assert infinity_response.status_code == 422


def test_predict_uses_trained_feature_order() -> None:
    service = make_service()
    reversed_payload = {
        key: VALID_PAYLOAD[key] for key in reversed(list(VALID_PAYLOAD))
    }

    with TestClient(create_app(model_loader=lambda: service)) as client:
        response = client.post("/predict", json=reversed_payload)

    assert response.status_code == 200
    assert service.model.received_columns == FEATURE_NAMES


def test_api_tests_do_not_modify_model() -> None:
    service = make_service()
    model_before = pickle.dumps(service.model)

    with TestClient(create_app(model_loader=lambda: service)) as client:
        client.post("/predict", json=VALID_PAYLOAD)

    service.model.received_columns = None
    assert pickle.dumps(service.model) == model_before


def test_loader_uses_tracking_uri_environment_variable(monkeypatch) -> None:
    import sp500_mlops_pipeline.serving.model_loader as loader_module

    tracking_uris = []
    fake_model = SimpleNamespace(feature_names_in_=np.array(FEATURE_NAMES))
    fake_version = SimpleNamespace(
        version="1",
        tags={"model_family": "logistic_regression"},
    )
    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.return_value = fake_version

    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://host.docker.internal:5000",
    )
    monkeypatch.setattr(
        loader_module.mlflow,
        "set_tracking_uri",
        tracking_uris.append,
    )
    monkeypatch.setattr(
        loader_module,
        "MlflowClient",
        lambda tracking_uri: (
            tracking_uris.append(tracking_uri) or fake_client
        ),
    )
    monkeypatch.setattr(
        loader_module.mlflow.sklearn,
        "load_model",
        lambda _: fake_model,
    )

    service = load_candidate_champion_model()

    assert tracking_uris == [
        "http://host.docker.internal:5000",
        "http://host.docker.internal:5000",
    ]
    assert service.model_version == "1"


def test_default_tracking_uri_remains_local() -> None:
    assert DEFAULT_TRACKING_URI == "http://127.0.0.1:5000"
