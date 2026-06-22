from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.model_train_challenger.nodes import (
    train_and_evaluate_challenger_model,
)


MODEL_PARAMETERS = {
    "model_type": "random_forest",
    "random_state": 42,
    "n_estimators": 20,
    "max_depth": 4,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "class_weight": None,
}
MLFLOW_PARAMETERS = {
    "tracking_uri": "http://127.0.0.1:5000",
    "experiment_name": "sp500_direction_prediction",
    "artifact_location": "data/08_reporting/mlflow_artifacts",
}


@pytest.fixture()
def training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rows = 120
    data = pd.DataFrame(
        {
            "simple_return": [((index % 11) - 5) / 100 for index in range(rows)],
            "log_return": [((index % 13) - 6) / 100 for index in range(rows)],
            "sma_10": [100 + index * 0.1 for index in range(rows)],
            "sma_20": [99 + index * 0.1 for index in range(rows)],
            "sma_ratio_10": [((index % 7) - 3) / 100 for index in range(rows)],
            "rsi_14": [30 + index % 40 for index in range(rows)],
            "volatility_10": [0.01 + (index % 9) / 1000 for index in range(rows)],
            "volume_change": [((index % 5) - 2) / 100 for index in range(rows)],
        }
    )
    target = pd.Series([index % 2 for index in range(rows)])
    return data.iloc[:90], target.iloc[:90], data.iloc[90:], target.iloc[90:]


@pytest.fixture()
def trained_challenger(training_data, monkeypatch):
    X_train, y_train, X_val, y_val = training_data
    run = SimpleNamespace(info=SimpleNamespace(run_id="test-run-id"))
    run_context = MagicMock()
    run_context.__enter__.return_value = run
    run_context.__exit__.return_value = False

    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_train_challenger.nodes."
        "mlflow.get_experiment_by_name",
        lambda _: SimpleNamespace(experiment_id="test-experiment-id"),
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_train_challenger.nodes."
        "mlflow.start_run",
        lambda **_: run_context,
    )
    for method_name in [
        "set_tracking_uri",
        "set_experiment",
        "log_params",
        "log_metrics",
        "set_tags",
        "log_dict",
    ]:
        monkeypatch.setattr(
            f"sp500_mlops_pipeline.pipelines.model_train_challenger.nodes."
            f"mlflow.{method_name}",
            lambda *args, **kwargs: None,
        )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_train_challenger.nodes."
        "mlflow.sklearn.log_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_train_challenger.nodes."
        "infer_signature",
        lambda *args, **kwargs: None,
    )

    model, metrics = train_and_evaluate_challenger_model(
        X_train,
        y_train,
        X_val,
        y_val,
        MODEL_PARAMETERS,
        MLFLOW_PARAMETERS,
    )
    return model, metrics, X_val


def test_trained_model_is_random_forest_classifier(trained_challenger) -> None:
    model, _, _ = trained_challenger

    assert isinstance(model, RandomForestClassifier)


def test_validation_metrics_contain_expected_keys(trained_challenger) -> None:
    _, metrics, _ = trained_challenger

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }


def test_validation_roc_auc_is_between_zero_and_one(trained_challenger) -> None:
    _, metrics, _ = trained_challenger

    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_challenger_can_predict_validation_data(trained_challenger) -> None:
    model, _, X_val = trained_challenger

    predictions = model.predict(X_val)

    assert len(predictions) == len(X_val)
    assert set(predictions).issubset({0, 1})
