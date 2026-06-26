"""Standalone reporting pipeline for model prediction plots."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_confusion_matrices_plot,
    create_logistic_regression_prediction_overlay_plot,
    create_model_comparison_plot,
    create_probability_distribution_plot,
    create_random_forest_prediction_overlay_plot,
    create_roc_curves_plot,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create prediction plots from persisted model prediction CSV files.

    Run this after the model prediction outputs exist, for example after
    ``kedro run`` or after the explicit prediction pipelines.
    """
    return pipeline(
        [
            node(
                func=create_logistic_regression_prediction_overlay_plot,
                inputs=["sp500_raw_data", "test_predictions"],
                outputs="logistic_regression_prediction_overlay_plot",
                name="create_logistic_regression_prediction_overlay_plot_node",
            ),
            node(
                func=create_random_forest_prediction_overlay_plot,
                inputs=["sp500_raw_data", "random_forest_test_predictions"],
                outputs="random_forest_prediction_overlay_plot",
                name="create_random_forest_prediction_overlay_plot_node",
            ),
            node(
                func=create_roc_curves_plot,
                inputs=["test_predictions", "random_forest_test_predictions"],
                outputs="roc_curves_plot",
                name="create_roc_curves_plot_node",
            ),
            node(
                func=create_confusion_matrices_plot,
                inputs=["test_predictions", "random_forest_test_predictions"],
                outputs="confusion_matrices_plot",
                name="create_confusion_matrices_plot_node",
            ),
            node(
                func=create_model_comparison_plot,
                inputs=["test_metrics", "random_forest_test_metrics"],
                outputs="model_comparison_plot",
                name="create_model_comparison_plot_node",
            ),
            node(
                func=create_probability_distribution_plot,
                inputs=["test_predictions", "random_forest_test_predictions"],
                outputs="probability_distribution_plot",
                name="create_probability_distribution_plot_node",
            ),
        ]
    )
