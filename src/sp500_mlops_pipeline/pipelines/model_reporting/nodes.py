"""Model-reporting plots for prediction outputs."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from sp500_mlops_pipeline.pipelines.data_cleaning.nodes import clean_raw_market_data


def create_prediction_overlay_plot(
    raw_market_data: pd.DataFrame,
    test_predictions: pd.DataFrame,
    model_label: str,
) -> plt.Figure:
    """Plot real Close price with vertical bars coloured by predicted direction."""
    plot_data = _prepare_prediction_plot_data(raw_market_data, test_predictions)

    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.plot(
        plot_data["Date"],
        plot_data["Close"],
        color="#1F4E79",
        linewidth=1.6,
        label="Actual Close",
        zorder=3,
    )

    y_min = float(plot_data["Close"].min())
    y_max = float(plot_data["Close"].max())
    y_padding = (y_max - y_min) * 0.06 if y_max > y_min else 1.0
    axis.set_ylim(y_min - y_padding, y_max + y_padding)

    for row in plot_data.itertuples(index=False):
        color = "#2CA02C" if row.predicted_target == 1 else "#D62728"
        axis.axvline(
            row.Date,
            color=color,
            alpha=0.28,
            linewidth=4,
            zorder=1,
        )

    legend_handles = [
        Line2D([0], [0], color="#1F4E79", linewidth=1.8, label="Actual Close"),
        Line2D([0], [0], color="#2CA02C", linewidth=4, label="Predicted up"),
        Line2D([0], [0], color="#D62728", linewidth=4, label="Predicted down"),
    ]
    axis.legend(handles=legend_handles, loc="best")
    axis.set_title(f"{model_label}: Real Close Price with Predicted Direction")
    axis.set_xlabel("Date")
    axis.set_ylabel("Close")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def create_logistic_regression_prediction_overlay_plot(
    raw_market_data: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> plt.Figure:
    """Create the baseline Logistic Regression prediction overlay plot."""
    return create_prediction_overlay_plot(
        raw_market_data=raw_market_data,
        test_predictions=test_predictions,
        model_label="Logistic Regression",
    )


def create_random_forest_prediction_overlay_plot(
    raw_market_data: pd.DataFrame,
    random_forest_test_predictions: pd.DataFrame,
) -> plt.Figure:
    """Create the Random Forest challenger prediction overlay plot."""
    return create_prediction_overlay_plot(
        raw_market_data=raw_market_data,
        test_predictions=random_forest_test_predictions,
        model_label="Random Forest",
    )


def _prepare_prediction_plot_data(
    raw_market_data: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Join prediction dates to the corresponding actual Close values."""
    required_prediction_columns = {"Date", "predicted_target"}
    missing_prediction_columns = required_prediction_columns - set(
        test_predictions.columns
    )
    if missing_prediction_columns:
        raise ValueError(
            "Prediction data is missing required columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    market_data = clean_raw_market_data(raw_market_data)
    market_data["Date"] = pd.to_datetime(market_data["Date"], errors="raise")
    predictions = test_predictions.copy()
    predictions["Date"] = pd.to_datetime(predictions["Date"], errors="raise")

    plot_data = predictions.merge(
        market_data.loc[:, ["Date", "Close"]],
        on="Date",
        how="left",
        validate="one_to_one",
    ).sort_values("Date")

    if plot_data["Close"].isna().any():
        missing_dates = plot_data.loc[
            plot_data["Close"].isna(),
            "Date",
        ].dt.strftime("%Y-%m-%d")
        raise ValueError(
            "Could not align predictions with raw Close prices for dates: "
            f"{missing_dates.head(5).tolist()}"
        )

    invalid_predictions = set(plot_data["predicted_target"].dropna().unique()) - {0, 1}
    if invalid_predictions:
        raise ValueError(
            "predicted_target must contain only 0 or 1. Found: "
            f"{sorted(invalid_predictions)}"
        )

    plot_data["predicted_target"] = plot_data["predicted_target"].astype(int)
    return plot_data.reset_index(drop=True)
