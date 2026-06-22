"""Baseline model training pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_and_evaluate_baseline_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the logistic-regression baseline training pipeline."""
    return pipeline(
        [
            node(
                func=train_and_evaluate_baseline_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "params:model_train",
                    "params:mlflow",
                ],
                outputs=["trained_baseline_model", "validation_metrics"],
                name="train_and_evaluate_baseline_model_node",
            )
        ]
    )
