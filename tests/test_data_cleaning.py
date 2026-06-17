from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_cleaning.nodes import (
    NUMERIC_COLUMNS,
    clean_raw_market_data,
    load_and_clean_raw_market_data,
)
from sp500_mlops_pipeline.pipelines.data_quality.nodes import REQUIRED_COLUMNS


RAW_SAMPLE_PATH = Path("data/01_raw/sp500_yahoo_finance_raw.csv")


def make_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2026-01-05", "2026-01-02"],
            "Open": ["101.0", "100.0"],
            "High": ["103.0", "102.0"],
            "Low": ["100.5", "99.0"],
            "Close": ["102.5", "101.5"],
            "Adj Close": ["102.5", "101.5"],
            "Volume": ["1100000", "1000000"],
            "Unused Column": ["drop me", "drop me too"],
        }
    )


def test_clean_raw_market_data_cleans_valid_raw_data() -> None:
    cleaned = clean_raw_market_data(make_raw_data())

    assert isinstance(cleaned, pd.DataFrame)
    assert len(cleaned) == 2
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])


def test_clean_raw_market_data_sorts_rows_by_date() -> None:
    cleaned = clean_raw_market_data(make_raw_data())

    assert cleaned["Date"].is_monotonic_increasing
    assert cleaned.loc[0, "Date"] == pd.Timestamp("2026-01-02")


def test_clean_raw_market_data_resets_index() -> None:
    raw_data = make_raw_data()
    raw_data.index = [10, 20]

    cleaned = clean_raw_market_data(raw_data)

    assert list(cleaned.index) == [0, 1]


def test_clean_raw_market_data_keeps_only_required_columns() -> None:
    cleaned = clean_raw_market_data(make_raw_data())

    assert list(cleaned.columns) == REQUIRED_COLUMNS


def test_clean_raw_market_data_converts_numeric_columns() -> None:
    cleaned = clean_raw_market_data(make_raw_data())

    for column in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(cleaned[column])


def test_real_raw_csv_can_be_cleaned() -> None:
    cleaned = load_and_clean_raw_market_data(RAW_SAMPLE_PATH)

    assert not cleaned.empty
    assert list(cleaned.columns) == REQUIRED_COLUMNS
    assert cleaned["Date"].is_monotonic_increasing
