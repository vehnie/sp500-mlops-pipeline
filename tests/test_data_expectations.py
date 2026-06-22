from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_expectations.nodes import (
    create_feature_validation_report,
    create_raw_validation_report,
    enforce_validation_contracts,
    validate_feature_data_expectations,
    validate_raw_market_data_expectations,
)
from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    MARKET_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


EXPECTED_REPORT_KEYS = {
    "validation_name",
    "success",
    "total_expectations",
    "successful_expectations",
    "failed_expectations",
    "failed_expectation_types",
    "failed_expectation_details",
    "validation_timestamp",
    "dataset_name",
}


def make_valid_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-01", periods=5),
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Adj Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Volume": [1_000_000, 1_100_000, 1_050_000, 1_200_000, 1_150_000],
        }
    )


def make_valid_feature_data() -> pd.DataFrame:
    rows = 5
    data = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-02-01", periods=rows),
            "simple_return": [0.01, -0.005, 0.003, 0.007, -0.002],
            "log_return": [0.009, -0.005, 0.003, 0.006, -0.002],
            "sma_10": [100.0, 100.2, 100.4, 100.7, 100.9],
            "sma_20": [99.5, 99.7, 99.9, 100.1, 100.3],
            "sma_ratio_10": [0.01, 0.008, 0.006, 0.009, 0.004],
            "rsi_14": [45.0, 48.0, 52.0, 55.0, 50.0],
            "volatility_10": [0.01, 0.012, 0.011, 0.013, 0.01],
            "volume_change": [0.02, -0.01, 0.03, 0.01, -0.02],
            TARGET_COLUMN: [1, 0, 1, 1, 0],
        }
    )
    assert set(MARKET_FEATURE_COLUMNS).issubset(data.columns)
    return data


def test_valid_raw_data_passes_validation() -> None:
    result = validate_raw_market_data_expectations(make_valid_raw_data())
    report = create_raw_validation_report(result)

    assert report["success"] is True


def test_raw_data_with_negative_price_fails() -> None:
    data = make_valid_raw_data()
    data.loc[0, "Close"] = -1.0

    result = validate_raw_market_data_expectations(data)
    report = create_raw_validation_report(result)

    with pytest.raises(ValueError, match="do not satisfy the quality contract"):
        enforce_validation_contracts(report, {"success": True})


def test_raw_data_with_duplicate_date_fails() -> None:
    data = make_valid_raw_data()
    data.loc[1, "Date"] = data.loc[0, "Date"]

    result = validate_raw_market_data_expectations(data)
    report = create_raw_validation_report(result)

    with pytest.raises(ValueError, match="do not satisfy the quality contract"):
        enforce_validation_contracts(report, {"success": True})


def test_valid_feature_data_passes_validation() -> None:
    result = validate_feature_data_expectations(make_valid_feature_data())
    report = create_feature_validation_report(result)

    assert report["success"] is True


def test_feature_data_with_invalid_target_fails() -> None:
    data = make_valid_feature_data()
    data.loc[0, TARGET_COLUMN] = 2

    result = validate_feature_data_expectations(data)
    report = create_feature_validation_report(result)

    with pytest.raises(ValueError, match="do not satisfy the quality contract"):
        enforce_validation_contracts({"success": True}, report)


def test_feature_data_with_null_feature_fails() -> None:
    data = make_valid_feature_data()
    data.loc[0, MARKET_FEATURE_COLUMNS[0]] = None

    result = validate_feature_data_expectations(data)
    report = create_feature_validation_report(result)

    with pytest.raises(ValueError, match="do not satisfy the quality contract"):
        enforce_validation_contracts({"success": True}, report)


def test_validation_reports_have_expected_keys() -> None:
    raw_report = create_raw_validation_report(
        validate_raw_market_data_expectations(make_valid_raw_data())
    )
    feature_report = create_feature_validation_report(
        validate_feature_data_expectations(make_valid_feature_data())
    )

    assert set(raw_report) == EXPECTED_REPORT_KEYS
    assert set(feature_report) == EXPECTED_REPORT_KEYS
