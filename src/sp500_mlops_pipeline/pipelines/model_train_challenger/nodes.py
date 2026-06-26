"""Random Forest challenger model training helpers."""

import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sp500_mlops_pipeline.pipelines.mlflow_utils import (
    active_or_new_mlflow_run,
    prefix_keys,
)


logger = logging.getLogger(__name__)

RUN_NAME = "random_forest_challenger_v2"
MODEL_NAME = "random_forest_challenger_model"
MLFLOW_STAGE = "challenger_training"


def train_and_evaluate_challenger_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame | pd.Series,
    model_parameters: dict,
    mlflow_parameters: dict,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Train a Random Forest challenger and evaluate it on validation data."""
    if model_parameters["model_type"] != "random_forest":
        raise ValueError("Only random_forest is supported")

    train_target = _to_series(y_train)
    validation_target = _to_series(y_val)

    model = RandomForestClassifier(
        n_estimators=model_parameters["n_estimators"],
        max_depth=model_parameters["max_depth"],
        min_samples_split=model_parameters["min_samples_split"],
        min_samples_leaf=model_parameters["min_samples_leaf"],
        class_weight=model_parameters["class_weight"],
        random_state=model_parameters["random_state"],
    )

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
    with redirect_stdout(io.StringIO()), active_or_new_mlflow_run(
        run_name=RUN_NAME
    ) as run:
        model.fit(X_train, train_target)

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(validation_target, predictions)),
            "precision": float(
                precision_score(validation_target, predictions, zero_division=0)
            ),
            "recall": float(recall_score(validation_target, predictions)),
            "f1_score": float(f1_score(validation_target, predictions)),
            "roc_auc": float(roc_auc_score(validation_target, probabilities)),
        }

        mlflow.log_params(prefix_keys(model_parameters, MLFLOW_STAGE))
        mlflow.log_metrics(prefix_keys(metrics, MLFLOW_STAGE))
        mlflow.set_tags(
            prefix_keys(
                {
                    "dataset": "sp500_feature_data",
                    "split_strategy": "chronological_70_15_15",
                    "model_family": "random_forest",
                    "model_role": "challenger",
                    "pipeline_stage": MLFLOW_STAGE,
                },
                MLFLOW_STAGE,
            )
        )
        mlflow.log_dict(
            {"features": X_train.columns.tolist()},
            f"{MLFLOW_STAGE}/feature_columns.json",
        )

        input_example = X_train.head(5)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name=MODEL_NAME,
            input_example=input_example,
            signature=signature,
        )
        logger.info(
            "MLflow run created: experiment=%s, run_name=%s, run_id=%s",
            experiment_name,
            RUN_NAME,
            run.info.run_id,
        )

    return model, metrics


def _to_series(target: pd.DataFrame | pd.Series) -> pd.Series:
    """Return a one-dimensional target series."""
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("Target data must contain exactly one column")
        return target.iloc[:, 0]

    return target
