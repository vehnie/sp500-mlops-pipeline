"""Feature engineering pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_market_feature_dataset


def create_pipeline(**kwargs) -> Pipeline:
    """Create the market feature engineering pipeline."""
    return pipeline(
        [
            node(
                func=create_market_feature_dataset,
                inputs="sp500_cleaned_data",
                outputs="sp500_feature_data",
                name="create_market_feature_dataset_node",
            )
        ]
    )
