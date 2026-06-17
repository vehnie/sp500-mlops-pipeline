from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_quality.nodes import (
    REQUIRED_COLUMNS,
    validate_raw_market_data,
    validate_raw_market_data_csv,
)


RAW_SAMPLE_PATH = Path("data/01_raw/sp500_yahoo_finance_raw.csv")


def make_valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-05"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.5],
            "Close": [101.5, 102.5],
            "Adj Close": [101.5, 102.5],
            "Volume": [1_000_000, 1_100_000],
        }
    )


def test_validate_raw_market_data_accepts_valid_data() -> None:
    validated = validate_raw_market_data(make_valid_data())

    assert list(validated.columns) == REQUIRED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(validated["Date"])


def test_existing_raw_csv_sample_passes_quality_checks() -> None:
    validated = validate_raw_market_data_csv(RAW_SAMPLE_PATH)

    assert not validated.empty
    assert list(validated.columns) == REQUIRED_COLUMNS


def test_validate_raw_market_data_rejects_missing_required_columns() -> None:
    data = make_valid_data().drop(columns=["Adj Close"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_market_data(data)


def test_validate_raw_market_data_rejects_empty_dataset() -> None:
    data = make_valid_data().iloc[0:0]

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_raw_market_data(data)


def test_validate_raw_market_data_rejects_invalid_dates() -> None:
    data = make_valid_data()
    data.loc[0, "Date"] = "not-a-date"

    with pytest.raises(ValueError, match="Date column cannot be parsed"):
        validate_raw_market_data(data)


def test_validate_raw_market_data_rejects_missing_values() -> None:
    data = make_valid_data()
    data.loc[0, "Close"] = None

    with pytest.raises(ValueError, match="Missing values found"):
        validate_raw_market_data(data)


def test_validate_raw_market_data_rejects_duplicate_dates() -> None:
    data = make_valid_data()
    data.loc[1, "Date"] = data.loc[0, "Date"]

    with pytest.raises(ValueError, match="Duplicate dates found"):
        validate_raw_market_data(data)


def test_validate_raw_market_data_rejects_non_positive_prices() -> None:
    data = make_valid_data()
    data.loc[0, "Open"] = 0.0

    with pytest.raises(ValueError, match="Price columns must contain positive values"):
        validate_raw_market_data(data)


def test_validate_raw_market_data_rejects_negative_volume() -> None:
    data = make_valid_data()
    data.loc[0, "Volume"] = -1

    with pytest.raises(ValueError, match="Volume must contain non-negative values"):
        validate_raw_market_data(data)
