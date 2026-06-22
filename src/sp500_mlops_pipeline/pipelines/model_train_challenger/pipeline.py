"""Random Forest challenger model training pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_and_evaluate_challenger_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the Random Forest challenger training pipeline."""
    return pipeline(
        [
            node(
                func=train_and_evaluate_challenger_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "params:model_train_challenger",
                    "params:mlflow",
                ],
                outputs=[
                    "trained_challenger_model",
                    "random_forest_validation_metrics",
                ],
                name="train_and_evaluate_challenger_model_node",
            )
        ]
    )
