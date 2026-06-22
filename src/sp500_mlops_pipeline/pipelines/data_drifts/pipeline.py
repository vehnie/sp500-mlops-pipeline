"""Evidently data drift evaluation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_observed_drift_report,
    create_synthetic_drift_report,
    evaluate_feature_drift,
    generate_synthetic_current_data,
    summarize_drift,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create observed and synthetic feature-drift evaluations."""
    return pipeline(
        [
            node(
                func=evaluate_feature_drift,
                inputs=["X_train", "X_test", "params:data_drifts"],
                outputs="observed_drift_evidently_result",
                name="evaluate_observed_feature_drift_node",
            ),
            node(
                func=create_observed_drift_report,
                inputs=["observed_drift_evidently_result", "params:data_drifts"],
                outputs="observed_feature_drift_report",
                name="create_observed_drift_report_node",
            ),
            node(
                func=generate_synthetic_current_data,
                inputs=["X_train", "params:data_drifts"],
                outputs="synthetic_current_features",
                name="generate_synthetic_current_data_node",
            ),
            node(
                func=evaluate_feature_drift,
                inputs=[
                    "X_train",
                    "synthetic_current_features",
                    "params:data_drifts",
                ],
                outputs="synthetic_drift_evidently_result",
                name="evaluate_synthetic_feature_drift_node",
            ),
            node(
                func=create_synthetic_drift_report,
                inputs=["synthetic_drift_evidently_result", "params:data_drifts"],
                outputs="synthetic_feature_drift_report",
                name="create_synthetic_drift_report_node",
            ),
            node(
                func=summarize_drift,
                inputs=[
                    "observed_feature_drift_report",
                    "synthetic_feature_drift_report",
                ],
                outputs="data_drift_summary",
                name="summarize_drift_node",
            ),
        ]
    )
