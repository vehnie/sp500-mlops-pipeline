"""Reporting plots for the market-data MLOps workflow."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sp500_mlops_pipeline.pipelines.data_cleaning.nodes import clean_raw_market_data
from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    TARGET_COLUMN,
    create_market_feature_dataset,
)


def create_close_price_plot(raw_market_data: pd.DataFrame) -> plt.Figure:
    """Create a historical S&P 500 Close-price plot for the project report."""
    cleaned_data = _prepare_clean_market_data(raw_market_data)

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(cleaned_data["Date"], cleaned_data["Close"], linewidth=1.1)
    axis.set_title("S&P 500 Close Price")
    axis.set_xlabel("Date")
    axis.set_ylabel("Close")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def create_daily_returns_plot(raw_market_data: pd.DataFrame) -> plt.Figure:
    """Create a daily-return time-series plot from the persisted raw snapshot."""
    cleaned_data = _prepare_clean_market_data(raw_market_data)
    cleaned_data["daily_return"] = cleaned_data["Close"].pct_change()

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(
        cleaned_data["Date"],
        cleaned_data["daily_return"],
        linewidth=0.8,
        alpha=0.85,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("S&P 500 Daily Returns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Daily return")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def create_rolling_volatility_plot(raw_market_data: pd.DataFrame) -> plt.Figure:
    """Create a 10-day rolling-volatility plot from engineered market features."""
    feature_data = _prepare_feature_data(raw_market_data)

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(
        feature_data["Date"],
        feature_data["volatility_10"],
        linewidth=1.0,
        color="#8B1E3F",
    )
    axis.set_title("S&P 500 10-Day Rolling Volatility")
    axis.set_xlabel("Date")
    axis.set_ylabel("Volatility")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def create_target_distribution_plot(raw_market_data: pd.DataFrame) -> plt.Figure:
    """Create a binary target-distribution plot for classification reporting."""
    feature_data = _prepare_feature_data(raw_market_data)
    counts = (
        feature_data[TARGET_COLUMN]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={0: "Down / not up", 1: "Up"})
    )

    figure, axis = plt.subplots(figsize=(7, 4.8))
    bars = axis.bar(counts.index, counts.values, color=["#4C78A8", "#59A14F"])
    axis.set_title("Target Distribution")
    axis.set_xlabel("Next-day direction")
    axis.set_ylabel("Rows")
    axis.grid(axis="y", alpha=0.25)

    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        share = value / total if total else 0.0
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}\n{share:.1%}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    return figure


def create_temporal_split_plot(
    raw_market_data: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> plt.Figure:
    """Create a visual summary of the chronological train/validation/test split."""
    feature_data = _prepare_feature_data(raw_market_data)
    split_labels = _assign_fractional_split_labels(
        feature_data,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    plot_data = pd.DataFrame(
        {
            "Date": feature_data["Date"],
            "Close": feature_data["Close"],
            "split": split_labels,
        }
    )

    colors = {
        "train": "#4C78A8",
        "validation": "#F58518",
        "test": "#54A24B",
    }
    figure, axis = plt.subplots(figsize=(12, 4.8))
    for split_name, split_data in plot_data.groupby("split", sort=False):
        axis.plot(
            split_data["Date"],
            split_data["Close"],
            linewidth=1.1,
            color=colors[split_name],
            label=split_name,
        )

    axis.set_title("Chronological Train / Validation / Test Split")
    axis.set_xlabel("Date")
    axis.set_ylabel("Close")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def _prepare_clean_market_data(raw_market_data: pd.DataFrame) -> pd.DataFrame:
    """Return sorted clean market data with parsed dates."""
    cleaned_data = clean_raw_market_data(raw_market_data)
    cleaned_data["Date"] = pd.to_datetime(cleaned_data["Date"], errors="raise")
    return cleaned_data.sort_values("Date").reset_index(drop=True)


def _prepare_feature_data(raw_market_data: pd.DataFrame) -> pd.DataFrame:
    """Return engineered feature data for reporting plots only."""
    cleaned_data = _prepare_clean_market_data(raw_market_data)
    feature_data = create_market_feature_dataset(cleaned_data)
    feature_data["Date"] = pd.to_datetime(feature_data["Date"], errors="raise")
    return feature_data.sort_values("Date").reset_index(drop=True)


def _assign_fractional_split_labels(
    data: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
) -> pd.Series:
    """Assign split labels using the same chronological fraction convention."""
    train_end = int(len(data) * train_fraction)
    val_end = train_end + int(len(data) * val_fraction)

    labels = pd.Series("test", index=data.index)
    labels.iloc[:train_end] = "train"
    labels.iloc[train_end:val_end] = "validation"
    return labels
