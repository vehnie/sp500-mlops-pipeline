"""Data cleaning pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_raw_market_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the market data cleaning pipeline."""
    return pipeline(
        [
            node(
                func=clean_raw_market_data,
                inputs="sp500_validated_data",
                outputs="sp500_cleaned_data",
                name="clean_raw_market_data_node",
            )
        ]
    )
