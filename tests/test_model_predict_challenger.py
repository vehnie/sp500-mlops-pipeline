from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.model_predict_challenger.nodes import (
    evaluate_challenger_model,
)


MLFLOW_PARAMETERS = {
    "tracking_uri": "http://127.0.0.1:5000",
    "experiment_name": "sp500_direction_prediction",
    "artifact_location": "data/08_reporting/mlflow_artifacts",
}


@pytest.fixture()
def evaluation_result(monkeypatch):
    rows = 12
    X_test = pd.DataFrame(
        {
            "simple_return": np.linspace(-0.03, 0.03, rows),
            "log_return": np.linspace(-0.02, 0.02, rows),
        }
    )
    y_test = pd.Series([0, 1] * (rows // 2))
    dates_test = pd.Series(pd.bdate_range("2026-01-01", periods=rows))

    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    model.predict_proba.return_value = np.column_stack(
        [
            1.0 - np.linspace(0.1, 0.9, rows),
            np.linspace(0.1, 0.9, rows),
        ]
    )

    run = SimpleNamespace(info=SimpleNamespace(run_id="test-run-id"))
    run_context = MagicMock()
    run_context.__enter__.return_value = run
    run_context.__exit__.return_value = False

    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_predict_challenger.nodes."
        "mlflow.get_experiment_by_name",
        lambda _: SimpleNamespace(experiment_id="test-experiment-id"),
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_predict_challenger.nodes."
        "mlflow.start_run",
        lambda **_: run_context,
    )
    for method_name in [
        "set_tracking_uri",
        "set_experiment",
        "log_metrics",
        "set_tags",
        "log_dict",
        "log_text",
    ]:
        monkeypatch.setattr(
            f"sp500_mlops_pipeline.pipelines.model_predict_challenger.nodes."
            f"mlflow.{method_name}",
            lambda *args, **kwargs: None,
        )
    signature = MagicMock()
    signature.to_dict.return_value = {"inputs": [], "outputs": []}
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_predict_challenger.nodes."
        "infer_signature",
        lambda *args, **kwargs: signature,
    )

    predictions, metrics = evaluate_challenger_model(
        model,
        X_test,
        y_test,
        dates_test,
        MLFLOW_PARAMETERS,
    )
    return predictions, metrics, X_test


def test_prediction_output_contains_expected_columns(evaluation_result) -> None:
    predictions, _, _ = evaluation_result

    assert list(predictions.columns) == [
        "Date",
        "actual_target",
        "predicted_target",
        "probability_up",
    ]


def test_test_metrics_contain_expected_keys(evaluation_result) -> None:
    _, metrics, _ = evaluation_result

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }


def test_test_roc_auc_is_between_zero_and_one(evaluation_result) -> None:
    _, metrics, _ = evaluation_result

    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_prediction_count_matches_test_rows(evaluation_result) -> None:
    predictions, _, X_test = evaluation_result

    assert len(predictions) == len(X_test)
