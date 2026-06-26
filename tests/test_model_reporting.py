from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.model_reporting.nodes import (
    create_logistic_regression_prediction_overlay_plot,
    create_random_forest_prediction_overlay_plot,
)


def make_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=6)
    close = [100.0, 101.0, 99.0, 102.0, 103.0, 101.5]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": [value + 1.0 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
            "Adj Close": close,
            "Volume": [1_000_000 + index for index in range(len(close))],
        }
    )


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-01", periods=6),
            "actual_target": [1, 0, 1, 1, 0, 0],
            "predicted_target": [1, 0, 1, 0, 1, 0],
            "probability_up": [0.7, 0.3, 0.6, 0.4, 0.8, 0.2],
        }
    )


def test_logistic_regression_prediction_overlay_plot_returns_figure() -> None:
    figure = create_logistic_regression_prediction_overlay_plot(
        make_market_data(),
        make_predictions(),
    )

    assert isinstance(figure, plt.Figure)
    assert figure.axes[0].get_title() == (
        "Logistic Regression: Real Close Price with Predicted Direction"
    )
    assert len(figure.axes[0].lines) >= 7
    plt.close(figure)


def test_random_forest_prediction_overlay_plot_returns_figure() -> None:
    figure = create_random_forest_prediction_overlay_plot(
        make_market_data(),
        make_predictions(),
    )

    assert isinstance(figure, plt.Figure)
    assert figure.axes[0].get_title() == (
        "Random Forest: Real Close Price with Predicted Direction"
    )
    assert len(figure.axes[0].lines) >= 7
    plt.close(figure)


def test_prediction_overlay_plot_rejects_missing_prediction_columns() -> None:
    malformed_predictions = make_predictions().drop(columns=["predicted_target"])

    with pytest.raises(ValueError, match="missing required columns"):
        create_logistic_regression_prediction_overlay_plot(
            make_market_data(),
            malformed_predictions,
        )


def test_prediction_overlay_plot_rejects_unaligned_prediction_dates() -> None:
    predictions = make_predictions()
    predictions.loc[0, "Date"] = "2030-01-01"

    with pytest.raises(ValueError, match="Could not align predictions"):
        create_logistic_regression_prediction_overlay_plot(
            make_market_data(),
            predictions,
        )
