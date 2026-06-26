"""Model-reporting plots for prediction outputs."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from sp500_mlops_pipeline.pipelines.data_cleaning.nodes import clean_raw_market_data


EVALUATION_METRIC_KEYS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")
EVALUATION_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1",
    "roc_auc": "ROC AUC",
}
LOGISTIC_REGRESSION_COLOR = "#1F4E79"
RANDOM_FOREST_COLOR = "#D62728"


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


def create_roc_curves_plot(
    test_predictions: pd.DataFrame,
    random_forest_test_predictions: pd.DataFrame,
) -> plt.Figure:
    """Overlay ROC curves (with AUC) for the baseline and challenger models."""
    models = [
        ("Logistic Regression", LOGISTIC_REGRESSION_COLOR, test_predictions),
        ("Random Forest", RANDOM_FOREST_COLOR, random_forest_test_predictions),
    ]

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    for model_label, color, predictions in models:
        validated = _validate_evaluation_predictions(predictions, model_label)
        false_positive_rate, true_positive_rate, _ = roc_curve(
            validated["actual_target"],
            validated["probability_up"],
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            linewidth=2.0,
            label=f"{model_label} (AUC = {roc_auc:.3f})",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        color="#7F7F7F",
        linestyle="--",
        linewidth=1.0,
        label="Chance",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("ROC Curves: Champion vs Challenger")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def create_confusion_matrices_plot(
    test_predictions: pd.DataFrame,
    random_forest_test_predictions: pd.DataFrame,
) -> plt.Figure:
    """Plot side-by-side test-set confusion matrices for both models."""
    models = [
        ("Logistic Regression", test_predictions),
        ("Random Forest", random_forest_test_predictions),
    ]
    class_labels = ["Down (0)", "Up (1)"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, (model_label, predictions) in zip(axes, models):
        validated = _validate_evaluation_predictions(
            predictions,
            model_label,
            require_probability=False,
        )
        matrix = confusion_matrix(
            validated["actual_target"],
            validated["predicted_target"],
            labels=[0, 1],
        )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(model_label)
        axis.set_xticks([0, 1], labels=class_labels)
        axis.set_yticks([0, 1], labels=class_labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

        threshold = matrix.max() / 2.0 if matrix.max() else 0
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Confusion Matrices (Test Set)")
    figure.tight_layout()
    return figure


def create_model_comparison_plot(
    test_metrics: dict,
    random_forest_test_metrics: dict,
) -> plt.Figure:
    """Plot grouped bars comparing test metrics for both models."""
    models = [
        ("Logistic Regression", LOGISTIC_REGRESSION_COLOR, test_metrics),
        ("Random Forest", RANDOM_FOREST_COLOR, random_forest_test_metrics),
    ]
    for model_label, _, metrics in models:
        missing_keys = [key for key in EVALUATION_METRIC_KEYS if key not in metrics]
        if missing_keys:
            raise ValueError(
                f"{model_label} metrics are missing required keys: {missing_keys}"
            )

    metric_labels = [EVALUATION_METRIC_LABELS[key] for key in EVALUATION_METRIC_KEYS]
    positions = np.arange(len(EVALUATION_METRIC_KEYS))
    bar_width = 0.38

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for offset, (model_label, color, metrics) in zip(
        (-bar_width / 2, bar_width / 2),
        models,
    ):
        values = [float(metrics[key]) for key in EVALUATION_METRIC_KEYS]
        bars = axis.bar(
            positions + offset,
            values,
            width=bar_width,
            color=color,
            label=model_label,
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    axis.set_xticks(positions, labels=metric_labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Model Comparison: Test Metrics")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def create_probability_distribution_plot(
    test_predictions: pd.DataFrame,
    random_forest_test_predictions: pd.DataFrame,
) -> plt.Figure:
    """Plot predicted-probability distributions split by actual class per model."""
    models = [
        ("Logistic Regression", test_predictions),
        ("Random Forest", random_forest_test_predictions),
    ]
    bins = np.linspace(0.0, 1.0, 21)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, (model_label, predictions) in zip(axes, models):
        validated = _validate_evaluation_predictions(predictions, model_label)
        actual = validated["actual_target"].astype(int)
        probability_up = validated["probability_up"]
        axis.hist(
            probability_up[actual == 1],
            bins=bins,
            color="#2CA02C",
            alpha=0.6,
            label="Actual up",
        )
        axis.hist(
            probability_up[actual == 0],
            bins=bins,
            color="#D62728",
            alpha=0.6,
            label="Actual down",
        )
        axis.axvline(0.5, color="#333333", linestyle="--", linewidth=1.0)
        axis.set_title(model_label)
        axis.set_xlabel("Predicted P(up)")
        axis.legend(loc="upper center")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Count")
    figure.suptitle("Predicted Probability Distribution by Actual Class")
    figure.tight_layout()
    return figure


def _validate_evaluation_predictions(
    predictions: pd.DataFrame,
    model_label: str,
    require_probability: bool = True,
) -> pd.DataFrame:
    """Validate the columns and target encoding of an evaluation prediction frame."""
    required_columns = {"actual_target", "predicted_target"}
    if require_probability:
        required_columns = required_columns | {"probability_up"}

    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"{model_label} prediction data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions.empty:
        raise ValueError(f"{model_label} prediction data is empty.")

    invalid_targets = set(predictions["actual_target"].dropna().unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(
            f"{model_label} actual_target must contain only 0 or 1. "
            f"Found: {sorted(invalid_targets)}"
        )

    return predictions
