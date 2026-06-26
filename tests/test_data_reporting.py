from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_reporting.nodes import (
    create_close_price_plot,
    create_daily_returns_plot,
    create_rolling_volatility_plot,
    create_target_distribution_plot,
    create_temporal_split_plot,
)


def make_market_data(rows: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=rows)
    close = pd.Series(range(100, 100 + rows), dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": range(1_000_000, 1_000_000 + rows),
        }
    )


def assert_valid_figure(figure: plt.Figure, title: str) -> None:
    assert isinstance(figure, plt.Figure)
    assert figure.axes
    assert figure.axes[0].get_title() == title
    plt.close(figure)


def test_create_close_price_plot_returns_figure() -> None:
    figure = create_close_price_plot(make_market_data())

    assert_valid_figure(figure, "S&P 500 Close Price")


def test_create_daily_returns_plot_returns_figure() -> None:
    figure = create_daily_returns_plot(make_market_data())

    assert_valid_figure(figure, "S&P 500 Daily Returns")


def test_create_rolling_volatility_plot_returns_figure() -> None:
    figure = create_rolling_volatility_plot(make_market_data())

    assert_valid_figure(figure, "S&P 500 10-Day Rolling Volatility")


def test_create_target_distribution_plot_returns_figure() -> None:
    figure = create_target_distribution_plot(make_market_data())

    assert_valid_figure(figure, "Target Distribution")


def test_create_temporal_split_plot_returns_figure_with_split_legend() -> None:
    figure = create_temporal_split_plot(make_market_data())

    labels = {text.get_text() for text in figure.axes[0].get_legend().get_texts()}
    assert labels == {"train", "validation", "test"}
    assert_valid_figure(figure, "Chronological Train / Validation / Test Split")
