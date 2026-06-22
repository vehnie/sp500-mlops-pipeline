"""Final Random Forest challenger evaluation helpers."""

import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


logger = logging.getLogger(__name__)

RUN_NAME = "random_forest_challenger_test_evaluation_v2"


def evaluate_challenger_model(
    trained_challenger_model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
    dates_test: pd.DataFrame | pd.Series,
    mlflow_parameters: dict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate the persisted Random Forest challenger on the final test set."""
    actual_target = _to_series(y_test, "y_test").reset_index(drop=True)
    test_dates = _to_series(dates_test, "dates_test").reset_index(drop=True)
    finite_X_test = X_test.replace([np.inf, -np.inf], 0.0)

    predicted_target = trained_challenger_model.predict(finite_X_test)
    probability_up = trained_challenger_model.predict_proba(finite_X_test)[:, 1]

    test_predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_dates, errors="raise"),
            "actual_target": actual_target.astype(int),
            "predicted_target": predicted_target.astype(int),
            "probability_up": probability_up,
        }
    )
    test_metrics = {
        "accuracy": float(accuracy_score(actual_target, predicted_target)),
        "precision": float(
            precision_score(actual_target, predicted_target, zero_division=0)
        ),
        "recall": float(recall_score(actual_target, predicted_target)),
        "f1_score": float(f1_score(actual_target, predicted_target)),
        "roc_auc": float(roc_auc_score(actual_target, probability_up)),
    }

    _log_test_evaluation(
        trained_challenger_model,
        finite_X_test,
        test_predictions,
        test_metrics,
        mlflow_parameters,
    )

    return test_predictions, test_metrics


def _log_test_evaluation(
    trained_challenger_model,
    X_test: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_metrics: dict[str, float],
    mlflow_parameters: dict,
) -> None:
    """Log final-test metrics and evaluation artefacts to MLflow."""
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

    input_example = X_test.head(5)
    signature = infer_signature(
        X_test,
        trained_challenger_model.predict(X_test),
    )

    mlflow.set_experiment(experiment_id=experiment_id)
    with redirect_stdout(io.StringIO()), mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_metrics(test_metrics)
        mlflow.set_tags(
            {
                "dataset": "sp500_feature_data",
                "split_strategy": "chronological_70_15_15",
                "model_family": "random_forest",
                "model_role": "challenger",
                "pipeline_stage": "final_test_evaluation",
                "evaluation_dataset": "test",
            }
        )
        mlflow.log_dict(
            test_metrics,
            "random_forest_test_metrics.json",
        )
        mlflow.log_text(
            test_predictions.to_csv(index=False),
            "random_forest_test_predictions.csv",
        )
        mlflow.log_text(
            input_example.to_csv(index=False),
            "input_example.csv",
        )
        mlflow.log_dict(
            signature.to_dict(),
            "signature.json",
        )
        logger.info(
            "MLflow test evaluation run created: experiment=%s, "
            "run_name=%s, run_id=%s",
            experiment_name,
            RUN_NAME,
            run.info.run_id,
        )


def _to_series(data: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    """Return a one-dimensional series."""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one column")
        return data.iloc[:, 0]

    return data
