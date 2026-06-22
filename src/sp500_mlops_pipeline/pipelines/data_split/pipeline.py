"""Time-based data split pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_time_based_train_val_test_split


def create_pipeline(**kwargs) -> Pipeline:
    """Create the chronological train, validation, and test split pipeline."""
    outputs = {
        dataset_name: dataset_name
        for dataset_name in (
            "X_train",
            "y_train",
            "dates_train",
            "X_val",
            "y_val",
            "dates_val",
            "X_test",
            "y_test",
            "dates_test",
        )
    }

    return pipeline(
        [
            node(
                func=create_time_based_train_val_test_split,
                inputs="sp500_feature_data",
                outputs=outputs,
                name="create_time_based_train_val_test_split_node",
            )
        ]
    )
