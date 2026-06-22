"""SHAP explainability pipeline for Logistic Regression v2."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_logistic_regression_shap_explanation


def create_pipeline(**kwargs) -> Pipeline:
    """Create the validation-set SHAP explainability pipeline."""
    return pipeline(
        [
            node(
                func=create_logistic_regression_shap_explanation,
                inputs=[
                    "trained_baseline_model",
                    "X_val",
                    "y_val",
                    "params:mlflow",
                ],
                outputs=[
                    "logistic_regression_v2_global_feature_importance",
                    "logistic_regression_v2_summary_plot",
                    "logistic_regression_v2_feature_importance_bar",
                    "logistic_regression_v2_explainability_summary",
                ],
                name="create_logistic_regression_shap_explanation_node",
            )
        ]
    )
