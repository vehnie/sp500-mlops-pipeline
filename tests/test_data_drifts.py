from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_drifts.nodes import (
    DRIFT_FEATURE_COLUMNS,
    create_observed_drift_report,
    create_synthetic_drift_report,
    evaluate_feature_drift,
    generate_synthetic_current_data,
    summarize_drift,
)


DRIFT_PARAMETERS = {
    "drift_share_threshold": 0.5,
    "p_value_threshold": 0.05,
    "synthetic_drift_intensity": 1.5,
    "random_state": 42,
}

EXPECTED_REPORT_KEYS = {
    "report_name",
    "dataset_name",
    "dataset_drift_detected",
    "number_of_features",
    "number_of_drifted_features",
    "share_of_drifted_features",
    "drift_share_threshold",
    "p_value_threshold",
    "drifted_features",
    "feature_drift_details",
    "current_missing_value_share",
    "drift_timestamp",
}


def make_reference_features(rows: int = 250, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {column: rng.normal(0.0, 1.0, rows) for column in DRIFT_FEATURE_COLUMNS}
    )


def test_identical_current_data_reports_no_drift() -> None:
    reference = make_reference_features()

    result = evaluate_feature_drift(reference, reference.copy(), DRIFT_PARAMETERS)
    report = create_observed_drift_report(result, DRIFT_PARAMETERS)

    assert report["dataset_drift_detected"] is False
    assert report["number_of_drifted_features"] == 0
    assert report["number_of_features"] == len(DRIFT_FEATURE_COLUMNS)
    assert report["drifted_features"] == []


def test_synthetic_perturbation_shifts_feature_distributions() -> None:
    reference = make_reference_features()

    current = generate_synthetic_current_data(reference, DRIFT_PARAMETERS)

    assert list(current.columns) == list(reference.columns)
    assert len(current) == len(reference)
    for column in DRIFT_FEATURE_COLUMNS:
        assert current[column].mean() > reference[column].mean()


def test_synthetic_scenario_reports_dataset_drift() -> None:
    reference = make_reference_features()
    current = generate_synthetic_current_data(reference, DRIFT_PARAMETERS)

    result = evaluate_feature_drift(reference, current, DRIFT_PARAMETERS)
    report = create_synthetic_drift_report(result, DRIFT_PARAMETERS)

    assert report["dataset_drift_detected"] is True
    assert report["share_of_drifted_features"] >= report["drift_share_threshold"]
    assert set(report["drifted_features"]).issubset(set(DRIFT_FEATURE_COLUMNS))


def test_large_sample_synthetic_drift_is_detected() -> None:
    # Evidently switches from a K-S p-value to a Wasserstein distance on large
    # samples; the drift decision must stay method-aware and still fire here.
    reference = make_reference_features(rows=5000)
    current = generate_synthetic_current_data(reference, DRIFT_PARAMETERS)

    result = evaluate_feature_drift(reference, current, DRIFT_PARAMETERS)
    report = create_synthetic_drift_report(result, DRIFT_PARAMETERS)

    assert report["dataset_drift_detected"] is True
    assert report["number_of_drifted_features"] == len(DRIFT_FEATURE_COLUMNS)
    methods = {
        detail["method"] for detail in report["feature_drift_details"].values()
    }
    assert any("p_value" not in method.lower() for method in methods)


def test_synthetic_generation_is_deterministic() -> None:
    reference = make_reference_features()

    first = generate_synthetic_current_data(reference, DRIFT_PARAMETERS)
    second = generate_synthetic_current_data(reference, DRIFT_PARAMETERS)

    pd.testing.assert_frame_equal(first, second)


def test_drift_report_has_expected_keys() -> None:
    reference = make_reference_features()
    current = generate_synthetic_current_data(reference, DRIFT_PARAMETERS)

    result = evaluate_feature_drift(reference, current, DRIFT_PARAMETERS)
    report = create_synthetic_drift_report(result, DRIFT_PARAMETERS)

    assert set(report) == EXPECTED_REPORT_KEYS
    assert report["report_name"] == "synthetic_feature_drift"
    for detail in report["feature_drift_details"].values():
        assert set(detail) == {"method", "value", "threshold", "drift_detected"}


def test_evaluate_feature_drift_requires_shared_columns() -> None:
    reference = make_reference_features()
    current = pd.DataFrame({"unrelated_column": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="No shared feature columns"):
        evaluate_feature_drift(reference, current, DRIFT_PARAMETERS)


def test_summarize_drift_compares_both_datasets() -> None:
    reference = make_reference_features()
    current = generate_synthetic_current_data(reference, DRIFT_PARAMETERS)

    observed = create_observed_drift_report(
        evaluate_feature_drift(reference, reference.copy(), DRIFT_PARAMETERS),
        DRIFT_PARAMETERS,
    )
    synthetic = create_synthetic_drift_report(
        evaluate_feature_drift(reference, current, DRIFT_PARAMETERS),
        DRIFT_PARAMETERS,
    )

    summary = summarize_drift(observed, synthetic)

    assert summary[observed["dataset_name"]]["dataset_drift_detected"] is False
    assert summary[synthetic["dataset_name"]]["dataset_drift_detected"] is True
