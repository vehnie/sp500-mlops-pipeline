"""Standalone data-reporting pipeline for project/manual plots."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_close_price_plot,
    create_daily_returns_plot,
    create_rolling_volatility_plot,
    create_target_distribution_plot,
    create_temporal_split_plot,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create reporting plots from the persisted local raw CSV snapshot.

    This pipeline is intended to run after an optional manual data refresh. It is
    independent from the default modelling pipeline and does not call Yahoo Finance.
    """
    return pipeline(
        [
            node(
                func=create_close_price_plot,
                inputs="sp500_raw_data",
                outputs="sp500_close_price_plot",
                name="create_close_price_plot_node",
            ),
            node(
                func=create_daily_returns_plot,
                inputs="sp500_raw_data",
                outputs="sp500_daily_returns_plot",
                name="create_daily_returns_plot_node",
            ),
            node(
                func=create_rolling_volatility_plot,
                inputs="sp500_raw_data",
                outputs="sp500_rolling_volatility_plot",
                name="create_rolling_volatility_plot_node",
            ),
            node(
                func=create_target_distribution_plot,
                inputs="sp500_raw_data",
                outputs="sp500_target_distribution_plot",
                name="create_target_distribution_plot_node",
            ),
            node(
                func=create_temporal_split_plot,
                inputs="sp500_raw_data",
                outputs="sp500_temporal_split_plot",
                name="create_temporal_split_plot_node",
            ),
        ]
    )
