"""Final baseline model evaluation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_baseline_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the chronological test evaluation pipeline."""
    return pipeline(
        [
            node(
                func=evaluate_baseline_model,
                inputs=[
                    "trained_baseline_model",
                    "X_test",
                    "y_test",
                    "dates_test",
                ],
                outputs=["test_predictions", "test_metrics"],
                name="evaluate_baseline_model_node",
            )
        ]
    )
