from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.model_reporting.nodes import (
    create_confusion_matrices_plot,
    create_logistic_regression_prediction_overlay_plot,
    create_model_comparison_plot,
    create_probability_distribution_plot,
    create_random_forest_prediction_overlay_plot,
    create_roc_curves_plot,
)


def make_metrics() -> dict:
    return {
        "accuracy": 0.55,
        "precision": 0.54,
        "recall": 0.92,
        "f1_score": 0.68,
        "roc_auc": 0.50,
    }


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


def test_roc_curves_plot_returns_figure_with_both_models() -> None:
    figure = create_roc_curves_plot(make_predictions(), make_predictions())

    assert isinstance(figure, plt.Figure)
    assert figure.axes[0].get_title() == "ROC Curves: Champion vs Challenger"
    legend_texts = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any("Logistic Regression (AUC =" in text for text in legend_texts)
    assert any("Random Forest (AUC =" in text for text in legend_texts)
    plt.close(figure)


def test_confusion_matrices_plot_returns_two_axes() -> None:
    figure = create_confusion_matrices_plot(make_predictions(), make_predictions())

    assert isinstance(figure, plt.Figure)
    plot_axes = [axis for axis in figure.axes if axis.images]
    assert len(plot_axes) == 2
    plt.close(figure)


def test_model_comparison_plot_returns_figure() -> None:
    figure = create_model_comparison_plot(make_metrics(), make_metrics())

    assert isinstance(figure, plt.Figure)
    assert figure.axes[0].get_title() == "Model Comparison: Test Metrics"
    plt.close(figure)


def test_model_comparison_plot_rejects_missing_metric_keys() -> None:
    incomplete_metrics = make_metrics()
    del incomplete_metrics["roc_auc"]

    with pytest.raises(ValueError, match="missing required keys"):
        create_model_comparison_plot(make_metrics(), incomplete_metrics)


def test_probability_distribution_plot_returns_figure() -> None:
    figure = create_probability_distribution_plot(
        make_predictions(),
        make_predictions(),
    )

    assert isinstance(figure, plt.Figure)
    assert len(figure.axes) == 2
    plt.close(figure)


def test_evaluation_plot_rejects_missing_probability_column() -> None:
    malformed = make_predictions().drop(columns=["probability_up"])

    with pytest.raises(ValueError, match="missing required columns"):
        create_roc_curves_plot(make_predictions(), malformed)
