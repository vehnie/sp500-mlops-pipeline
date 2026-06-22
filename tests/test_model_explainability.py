from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    MARKET_FEATURE_COLUMNS,
)
from sp500_mlops_pipeline.pipelines.model_explainability.nodes import (
    create_logistic_regression_shap_explanation,
)


MLFLOW_PARAMETERS = {
    "tracking_uri": "http://127.0.0.1:5000",
    "experiment_name": "sp500_direction_prediction",
    "artifact_location": "data/08_reporting/mlflow_artifacts",
}


@pytest.fixture()
def explanation_result(monkeypatch):
    rows = 60
    X_val = pd.DataFrame(
        {
            feature: np.linspace(index, index + 1, rows)
            for index, feature in enumerate(MARKET_FEATURE_COLUMNS)
        }
    )
    y_val = pd.Series([index % 2 for index in range(rows)])
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(X_val, y_val)
    model_before = pickle.dumps(model)

    run = SimpleNamespace(info=SimpleNamespace(run_id="test-run-id"))
    run_context = MagicMock()
    run_context.__enter__.return_value = run
    run_context.__exit__.return_value = False

    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_explainability.nodes."
        "mlflow.get_experiment_by_name",
        lambda _: SimpleNamespace(experiment_id="test-experiment-id"),
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.model_explainability.nodes."
        "mlflow.start_run",
        lambda **_: run_context,
    )
    for method_name in [
        "set_tracking_uri",
        "set_experiment",
        "set_tags",
        "log_text",
        "log_figure",
        "log_dict",
        "log_params",
        "log_metrics",
    ]:
        monkeypatch.setattr(
            f"sp500_mlops_pipeline.pipelines.model_explainability.nodes."
            f"mlflow.{method_name}",
            lambda *args, **kwargs: None,
        )

    importance, summary_plot, bar_plot, summary = (
        create_logistic_regression_shap_explanation(
            model,
            X_val,
            y_val,
            MLFLOW_PARAMETERS,
        )
    )
    return {
        "importance": importance,
        "summary_plot": summary_plot,
        "bar_plot": bar_plot,
        "summary": summary,
        "model_before": model_before,
        "model_after": pickle.dumps(model),
        "X_val": X_val,
    }


def test_importance_contains_all_eight_features(explanation_result) -> None:
    importance = explanation_result["importance"]

    assert set(importance["feature"]) == set(MARKET_FEATURE_COLUMNS)
    assert len(importance) == len(MARKET_FEATURE_COLUMNS)


def test_importance_table_has_expected_columns(explanation_result) -> None:
    importance = explanation_result["importance"]

    assert list(importance.columns) == [
        "feature",
        "mean_abs_shap_value",
        "rank",
    ]


def test_importance_ranks_are_consecutive(explanation_result) -> None:
    importance = explanation_result["importance"]

    assert importance["rank"].tolist() == list(range(1, 9))


def test_importance_values_are_non_negative(explanation_result) -> None:
    importance = explanation_result["importance"]

    assert (importance["mean_abs_shap_value"] >= 0).all()


def test_plot_artefacts_can_be_created(
    explanation_result,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.png"
    bar_path = tmp_path / "bar.png"
    explanation_result["summary_plot"].savefig(summary_path)
    explanation_result["bar_plot"].savefig(bar_path)

    assert summary_path.exists() and summary_path.stat().st_size > 0
    assert bar_path.exists() and bar_path.stat().st_size > 0


def test_pipeline_does_not_modify_trained_model(explanation_result) -> None:
    assert explanation_result["model_before"] == explanation_result["model_after"]


def test_summary_observation_count_matches_x_val(explanation_result) -> None:
    summary = explanation_result["summary"]

    assert summary["n_observations"] == len(explanation_result["X_val"])
    assert summary["n_features"] == len(MARKET_FEATURE_COLUMNS)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
