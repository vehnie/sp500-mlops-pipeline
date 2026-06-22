"""Final baseline model evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_baseline_model(
    trained_baseline_model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
    dates_test: pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate the trained baseline on the chronological test set."""
    actual_target = _to_series(y_test, "y_test").reset_index(drop=True)
    test_dates = _to_series(dates_test, "dates_test").reset_index(drop=True)
    finite_X_test = X_test.replace([np.inf, -np.inf], 0.0)

    predicted_target = trained_baseline_model.predict(finite_X_test)
    probability_up = trained_baseline_model.predict_proba(finite_X_test)[:, 1]

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

    return test_predictions, test_metrics


def _to_series(data: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    """Return a one-dimensional series."""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one column")
        return data.iloc[:, 0]

    return data
