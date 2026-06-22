"""Data quality pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import validate_raw_market_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the raw market data validation pipeline."""
    return pipeline(
        [
            node(
                func=validate_raw_market_data,
                inputs="sp500_raw_data",
                outputs="sp500_validated_data",
                name="validate_raw_market_data_node",
            )
        ]
    )
