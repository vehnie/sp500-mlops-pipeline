"""Evidently-based data drift evaluation for the engineered feature set."""

from datetime import datetime, timezone
import logging

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DatasetMissingValueCount
from evidently.presets import DataDriftPreset

from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    MARKET_FEATURE_COLUMNS,
)


logger = logging.getLogger(__name__)

DRIFT_FEATURE_COLUMNS = list(MARKET_FEATURE_COLUMNS)
OBSERVED_REPORT_NAME = "observed_feature_drift"
SYNTHETIC_REPORT_NAME = "synthetic_feature_drift"
OBSERVED_DATASET_NAME = "sp500_features_train_vs_test"
SYNTHETIC_DATASET_NAME = "sp500_features_synthetic_drift"
DRIFT_SHARE_THRESHOLD = 0.5
P_VALUE_THRESHOLD = 0.05

_VALUE_DRIFT_TYPE = "ValueDrift"
_MISSING_VALUE_TYPE = "DatasetMissingValueCount"


def generate_synthetic_current_data(
    reference: pd.DataFrame,
    drift_parameters: dict,
) -> pd.DataFrame:
    """Perturb the reference features to simulate covariate drift.

    Provides the synthetic monitoring scenario described in the project plan:
    a controlled distribution shift the drift report is expected to flag.
    """
    intensity = float(drift_parameters.get("synthetic_drift_intensity", 1.5))
    random_state = int(drift_parameters.get("random_state", 42))
    rng = np.random.default_rng(random_state)

    current = reference.reset_index(drop=True).copy()
    for column in _present_feature_columns(current):
        series = current[column].astype(float)
        spread = float(series.std(ddof=0)) or 1.0
        noise = rng.normal(0.0, spread * 0.1, size=len(series))
        current[column] = series + intensity * spread + noise

    return current


def evaluate_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_parameters: dict,
) -> dict:
    """Run the Evidently data-drift report over shared feature columns."""
    del drift_parameters  # report configuration is applied when summarizing
    columns = _shared_feature_columns(reference, current)
    if not columns:
        raise ValueError("No shared feature columns available for drift evaluation")

    data_definition = DataDefinition(numerical_columns=columns)
    reference_dataset = Dataset.from_pandas(
        reference[columns].reset_index(drop=True),
        data_definition=data_definition,
    )
    current_dataset = Dataset.from_pandas(
        current[columns].reset_index(drop=True),
        data_definition=data_definition,
    )

    report = Report(metrics=[DataDriftPreset(), DatasetMissingValueCount()])
    run = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset,
    )
    return run.dict()


def create_observed_drift_report(
    evidently_result: dict,
    drift_parameters: dict,
) -> dict:
    """Build the compact report for observed train-vs-test feature drift."""
    return create_drift_report(
        evidently_result=evidently_result,
        report_name=OBSERVED_REPORT_NAME,
        dataset_name=OBSERVED_DATASET_NAME,
        drift_parameters=drift_parameters,
    )


def create_synthetic_drift_report(
    evidently_result: dict,
    drift_parameters: dict,
) -> dict:
    """Build the compact report for the synthetic drift scenario."""
    return create_drift_report(
        evidently_result=evidently_result,
        report_name=SYNTHETIC_REPORT_NAME,
        dataset_name=SYNTHETIC_DATASET_NAME,
        drift_parameters=drift_parameters,
    )


def create_drift_report(
    evidently_result: dict,
    report_name: str,
    dataset_name: str,
    drift_parameters: dict,
) -> dict:
    """Transform an Evidently snapshot into a compact, serializable report."""
    p_value_threshold = float(
        drift_parameters.get("p_value_threshold", P_VALUE_THRESHOLD)
    )
    drift_share_threshold = float(
        drift_parameters.get("drift_share_threshold", DRIFT_SHARE_THRESHOLD)
    )

    metrics = evidently_result.get("metrics", [])
    feature_drift_details = _extract_value_drifts(metrics, p_value_threshold)
    drifted_features = sorted(
        column
        for column, detail in feature_drift_details.items()
        if detail["drift_detected"]
    )
    number_of_features = len(feature_drift_details)
    number_of_drifted = len(drifted_features)
    share_of_drifted = (
        number_of_drifted / number_of_features if number_of_features else 0.0
    )

    return {
        "report_name": report_name,
        "dataset_name": dataset_name,
        "dataset_drift_detected": bool(share_of_drifted >= drift_share_threshold),
        "number_of_features": number_of_features,
        "number_of_drifted_features": number_of_drifted,
        "share_of_drifted_features": share_of_drifted,
        "drift_share_threshold": drift_share_threshold,
        "p_value_threshold": p_value_threshold,
        "drifted_features": drifted_features,
        "feature_drift_details": feature_drift_details,
        "current_missing_value_share": _extract_missing_value_share(metrics),
        "drift_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def summarize_drift(
    observed_report: dict,
    synthetic_report: dict,
) -> dict:
    """Log and return a short comparison of the observed and synthetic reports."""
    summary = {
        report["dataset_name"]: {
            "dataset_drift_detected": report["dataset_drift_detected"],
            "number_of_drifted_features": report["number_of_drifted_features"],
            "share_of_drifted_features": report["share_of_drifted_features"],
        }
        for report in (observed_report, synthetic_report)
    }
    logger.info("Data drift summary: %s", summary)
    return summary


def _extract_value_drifts(metrics: list, p_value_threshold: float) -> dict:
    """Collect per-column drift decisions from Evidently ValueDrift metrics.

    Evidently adapts the statistical test to the sample size (e.g. a
    Kolmogorov-Smirnov p-value on small samples, a normed Wasserstein distance
    on large ones), so the drift decision is made in a method-aware way rather
    than assuming the value is always a p-value.
    """
    details: dict[str, dict] = {}
    for metric in metrics:
        config = metric.get("config", {})
        if not str(config.get("type", "")).endswith(_VALUE_DRIFT_TYPE):
            continue
        column = config.get("column")
        if column is None:
            continue
        method = str(config.get("method", ""))
        value = _as_float(metric.get("value"))
        is_p_value = "p_value" in method.lower()
        # For p-value tests the project threshold applies (drift when below it);
        # distance tests use the threshold Evidently calibrated for the sample.
        threshold = (
            p_value_threshold if is_p_value else _as_float(config.get("threshold"))
        )
        details[column] = {
            "method": method,
            "value": value,
            "threshold": threshold,
            "drift_detected": _is_drifted(value, threshold, is_p_value),
        }
    return details


def _is_drifted(value, threshold, is_p_value: bool) -> bool:
    """Decide drift for one column given its test method and threshold."""
    if value is None or threshold is None:
        return False
    if is_p_value:
        return value < threshold
    return value >= threshold


def _extract_missing_value_share(metrics: list) -> float:
    """Return the share of missing values in the current dataset."""
    for metric in metrics:
        config = metric.get("config", {})
        if str(config.get("type", "")).endswith(_MISSING_VALUE_TYPE):
            value = metric.get("value", {})
            if isinstance(value, dict):
                return float(value.get("share", 0.0))
    return 0.0


def _as_float(value) -> float | None:
    """Coerce an Evidently metric value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _present_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return drift feature columns present in a single frame."""
    return [column for column in DRIFT_FEATURE_COLUMNS if column in frame.columns]


def _shared_feature_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> list[str]:
    """Return drift feature columns present in both frames."""
    return [
        column
        for column in DRIFT_FEATURE_COLUMNS
        if column in reference.columns and column in current.columns
    ]
