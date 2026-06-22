"""Global SHAP explainability helpers for Logistic Regression v2."""

import io
import logging
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)

MODEL_NAME = "sp500_direction_model"
MODEL_VERSION = "1"
MODEL_ALIAS = "candidate_champion"
RUN_NAME = "logistic_regression_v2_shap_explainability"


def create_logistic_regression_shap_explanation(
    trained_baseline_model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame | pd.Series,
    mlflow_parameters: dict,
) -> tuple[pd.DataFrame, plt.Figure, plt.Figure, dict]:
    """Explain the persisted Logistic Regression pipeline on validation data."""
    validation_target = _to_series(y_val)
    if len(validation_target) != len(X_val):
        raise ValueError("X_val and y_val must contain the same number of rows")

    feature_names = _get_feature_names(trained_baseline_model, X_val)
    ordered_X_val = X_val.loc[:, feature_names].copy()
    scaler = trained_baseline_model.named_steps["scaler"]
    classifier = trained_baseline_model.named_steps["classifier"]

    scaled_values = scaler.transform(ordered_X_val)
    scaled_X_val = pd.DataFrame(
        scaled_values,
        columns=feature_names,
        index=ordered_X_val.index,
    )

    explainer = shap.LinearExplainer(classifier, scaled_X_val)
    shap_values = np.asarray(explainer.shap_values(scaled_X_val))
    if shap_values.ndim != 2:
        raise ValueError(
            "Expected two-dimensional SHAP values for binary Logistic Regression"
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
        }
    )
    importance = importance.sort_values(
        "mean_abs_shap_value",
        ascending=False,
    ).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)

    summary_figure = _create_summary_plot(
        shap_values=shap_values,
        feature_values=ordered_X_val,
    )
    importance_figure = _create_importance_bar_plot(importance)
    summary = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_alias": MODEL_ALIAS,
        "explainer_type": "shap.LinearExplainer",
        "explained_dataset": "validation",
        "n_observations": int(len(ordered_X_val)),
        "n_features": int(len(feature_names)),
        "positive_class": 1,
        "top_features": importance.head(5).to_dict(orient="records"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _log_explainability_run(
        importance=importance,
        summary_figure=summary_figure,
        importance_figure=importance_figure,
        summary=summary,
        mlflow_parameters=mlflow_parameters,
    )

    return importance, summary_figure, importance_figure, summary


def _create_summary_plot(
    shap_values: np.ndarray,
    feature_values: pd.DataFrame,
) -> plt.Figure:
    """Create the global SHAP beeswarm summary plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        feature_values,
        feature_names=feature_values.columns.tolist(),
        show=False,
    )
    figure = plt.gcf()
    figure.tight_layout()
    return figure


def _create_importance_bar_plot(importance: pd.DataFrame) -> plt.Figure:
    """Create a ranked mean absolute SHAP importance chart."""
    figure, axis = plt.subplots(figsize=(9, 5))
    ordered = importance.sort_values("mean_abs_shap_value", ascending=True)
    axis.barh(
        ordered["feature"],
        ordered["mean_abs_shap_value"],
        color="#2E86AB",
    )
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("Feature")
    axis.set_title("Logistic Regression v2 global feature importance")
    figure.tight_layout()
    return figure


def _log_explainability_run(
    importance: pd.DataFrame,
    summary_figure: plt.Figure,
    importance_figure: plt.Figure,
    summary: dict,
    mlflow_parameters: dict,
) -> None:
    """Log SHAP artefacts and metadata to a dedicated MLflow run."""
    mlflow.set_tracking_uri(mlflow_parameters["tracking_uri"])
    experiment_name = mlflow_parameters["experiment_name"]
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = Path(
            mlflow_parameters["artifact_location"]
        ).resolve().as_uri()
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)
    with redirect_stdout(io.StringIO()), mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.set_tags(
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "model_alias": MODEL_ALIAS,
                "model_family": "logistic_regression",
                "pipeline_stage": "explainability",
                "explained_dataset": "validation",
            }
        )
        mlflow.log_text(
            importance.to_csv(index=False),
            "logistic_regression_v2_global_feature_importance.csv",
        )
        mlflow.log_figure(
            summary_figure,
            "logistic_regression_v2_summary_plot.png",
        )
        mlflow.log_figure(
            importance_figure,
            "logistic_regression_v2_feature_importance_bar.png",
        )
        mlflow.log_dict(
            summary,
            "logistic_regression_v2_explainability_summary.json",
        )
        mlflow.log_params(
            {
                f"rank_{row.rank}_feature": row.feature
                for row in importance.head(5).itertuples()
            }
        )
        mlflow.log_metrics(
            {
                f"rank_{row.rank}_mean_abs_shap": float(
                    row.mean_abs_shap_value
                )
                for row in importance.head(5).itertuples()
            }
        )
        logger.info(
            "MLflow SHAP run created: experiment=%s, run_id=%s",
            experiment_name,
            run.info.run_id,
        )


def _get_feature_names(
    trained_baseline_model: Pipeline,
    X_val: pd.DataFrame,
) -> list[str]:
    """Return training feature names in their original order."""
    feature_names = getattr(trained_baseline_model, "feature_names_in_", None)
    if feature_names is None:
        return X_val.columns.tolist()

    names = list(feature_names)
    missing = [name for name in names if name not in X_val.columns]
    if missing:
        raise ValueError(f"X_val is missing trained feature columns: {missing}")
    return names


def _to_series(target: pd.DataFrame | pd.Series) -> pd.Series:
    """Return a one-dimensional target series."""
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("y_val must contain exactly one column")
        return target.iloc[:, 0]
    return target
