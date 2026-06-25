"""Nodes for chronological feature drift analysis with Evidently."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


DATE_COLUMN = "Date"
REFERENCE_FRACTION = 0.70
DRIFT_FEATURE_COLUMNS = [
    "simple_return",
    "log_return",
    "sma_10",
    "sma_20",
    "sma_ratio_10",
    "rsi_14",
    "volatility_10",
    "volume_change",
]


def split_reference_current_data(
    feature_data: pd.DataFrame,
    reference_fraction: float = REFERENCE_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort feature data by date and split it into oldest and newest periods."""
    if DATE_COLUMN not in feature_data.columns:
        raise ValueError(f"Missing required column: {DATE_COLUMN}")
    if feature_data.empty:
        raise ValueError("Feature dataset is empty")
    if not 0 < reference_fraction < 1:
        raise ValueError("Reference fraction must be between 0 and 1")

    sorted_data = feature_data.copy()
    sorted_data[DATE_COLUMN] = pd.to_datetime(
        sorted_data[DATE_COLUMN],
        errors="raise",
    )
    sorted_data = sorted_data.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = int(len(sorted_data) * reference_fraction)
    if split_index == 0 or split_index == len(sorted_data):
        raise ValueError("Reference and current data splits must both be non-empty")

    reference_data = sorted_data.iloc[:split_index].reset_index(drop=True)
    current_data = sorted_data.iloc[split_index:].reset_index(drop=True)
    return reference_data, current_data


def select_numeric_drift_features(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select and validate the numeric features used by Evidently."""
    missing_columns = [
        column
        for column in DRIFT_FEATURE_COLUMNS
        if column not in reference_data.columns or column not in current_data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required drift feature columns: {missing_columns}")

    non_numeric_columns = [
        column
        for column in DRIFT_FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(reference_data[column])
        or not pd.api.types.is_numeric_dtype(current_data[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            f"Drift feature columns must be numeric: {non_numeric_columns}"
        )

    reference_features = reference_data.loc[:, DRIFT_FEATURE_COLUMNS].copy()
    current_features = current_data.loc[:, DRIFT_FEATURE_COLUMNS].copy()
    return reference_features, current_features


def create_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> tuple[str, dict]:
    """Create standalone Evidently HTML and a compact JSON drift summary."""
    report = Report(
        metrics=[DataDriftPreset(columns=DRIFT_FEATURE_COLUMNS)],
        include_tests=True,
    )
    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )
    report_data = snapshot.dict()

    summary = _create_drift_summary(
        report_data=report_data,
        reference_rows=len(reference_data),
        current_rows=len(current_data),
    )
    return snapshot.get_html_str(as_iframe=False), summary


def _create_drift_summary(
    report_data: dict,
    reference_rows: int,
    current_rows: int,
) -> dict:
    drifted_columns_count = 0
    drifted_columns_share = 0.0
    column_drift: dict[str, dict] = {}

    for metric in report_data.get("metrics", []):
        config = metric.get("config", {})
        metric_type = str(config.get("type", ""))
        value = metric.get("value")

        if metric_type.endswith("DriftedColumnsCount") and isinstance(value, dict):
            drifted_columns_count = int(value.get("count", 0))
            drifted_columns_share = float(value.get("share", 0.0))
            continue

        if not metric_type.endswith("ValueDrift"):
            continue

        column = config.get("column")
        if column not in DRIFT_FEATURE_COLUMNS or not isinstance(value, (int, float)):
            continue

        threshold = float(config.get("threshold", 0.05))
        method = str(config.get("method", ""))
        drift_detected = (
            float(value) < threshold
            if "p_value" in method.lower()
            else float(value) > threshold
        )
        column_drift[column] = {
            "drift_detected": drift_detected,
            "drift_score": float(value),
            "method": method,
            "threshold": threshold,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_rows": reference_rows,
        "current_rows": current_rows,
        "features": DRIFT_FEATURE_COLUMNS,
        "number_of_features": len(DRIFT_FEATURE_COLUMNS),
        "dataset_drift": drifted_columns_share >= 0.5,
        "drifted_columns_count": drifted_columns_count,
        "drifted_columns_share": drifted_columns_share,
        "column_drift": column_drift,
    }
