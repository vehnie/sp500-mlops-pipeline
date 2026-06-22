"""Final Random Forest challenger evaluation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_challenger_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the chronological final-test evaluation pipeline."""
    return pipeline(
        [
            node(
                func=evaluate_challenger_model,
                inputs=[
                    "trained_challenger_model",
                    "X_test",
                    "y_test",
                    "dates_test",
                    "params:mlflow",
                ],
                outputs=[
                    "random_forest_test_predictions",
                    "random_forest_test_metrics",
                ],
                name="evaluate_challenger_model_node",
            )
        ]
    )
