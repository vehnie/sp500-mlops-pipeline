"""Evidently data drift pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_data_drift_report,
    select_numeric_drift_features,
    split_reference_current_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the chronological feature data drift pipeline."""
    return pipeline(
        [
            node(
                func=split_reference_current_data,
                inputs="sp500_feature_data",
                outputs=["reference_data", "current_data"],
                name="split_reference_current_data_node",
            ),
            node(
                func=select_numeric_drift_features,
                inputs=["reference_data", "current_data"],
                outputs=[
                    "reference_drift_features",
                    "current_drift_features",
                ],
                name="select_numeric_drift_features_node",
            ),
            node(
                func=create_data_drift_report,
                inputs=[
                    "reference_drift_features",
                    "current_drift_features",
                ],
                outputs=[
                    "data_drift_report",
                    "data_drift_summary",
                ],
                name="create_data_drift_report_node",
            ),
        ]
    )
