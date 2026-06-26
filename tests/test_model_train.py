from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.model_train.nodes import (
    train_and_evaluate_baseline_model,
)


MODEL_PARAMETERS = {
    "model_type": "logistic_regression",
    "random_state": 42,
    "max_iter": 1000,
}
MLFLOW_PARAMETERS = {
    "tracking_uri": "http://127.0.0.1:5000",
    "experiment_name": "sp500_direction_prediction",
    "run_name": "logistic_regression_baseline_v2",
    "artifact_location": "data/08_reporting/mlflow_artifacts",
    "model_name": "sp500_direction_model",
}


@pytest.fixture()
def trained_baseline(monkeypatch):
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
    X_train, y_train = data.iloc[:90], target.iloc[:90]
    X_val, y_val = data.iloc[90:], target.iloc[90:]

    run = SimpleNamespace(info=SimpleNamespace(run_id="test-run-id"))
    run_context = MagicMock()
    run_context.__enter__.return_value = run
    run_context.__exit__.return_value = False

    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_train.nodes."
        "mlflow.get_experiment_by_name",
        lambda _: SimpleNamespace(experiment_id="test-experiment-id"),
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_train.nodes.mlflow.start_run",
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
            f"sp500_mlops_pipeline.pipelines.model_train.nodes."
            f"mlflow.{method_name}",
            lambda *args, **kwargs: None,
        )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_train.nodes.mlflow.sklearn.log_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_train.nodes.infer_signature",
        lambda *args, **kwargs: None,
    )

    model, metrics = train_and_evaluate_baseline_model(
        X_train,
        y_train,
        X_val,
        y_val,
        MODEL_PARAMETERS,
        MLFLOW_PARAMETERS,
    )
    return model, metrics, X_val


def test_trained_baseline_is_sklearn_pipeline(trained_baseline) -> None:
    model, _, _ = trained_baseline

    assert isinstance(model, Pipeline)


def test_baseline_validation_metrics_contain_expected_keys(trained_baseline) -> None:
    _, metrics, _ = trained_baseline

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }


def test_baseline_can_predict_validation_data(trained_baseline) -> None:
    model, _, X_val = trained_baseline

    predictions = model.predict(X_val)

    assert len(predictions) == len(X_val)
    assert set(predictions).issubset({0, 1})
