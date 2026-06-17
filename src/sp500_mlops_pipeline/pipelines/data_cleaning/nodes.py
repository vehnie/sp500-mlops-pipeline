"""Cleaning helpers for raw S&P 500 market data."""

from pathlib import Path

import pandas as pd

from sp500_mlops_pipeline.pipelines.data_quality.nodes import (
    PRICE_COLUMNS,
    REQUIRED_COLUMNS,
    validate_raw_market_data,
)


NUMERIC_COLUMNS = [*PRICE_COLUMNS, "Volume"]
RAW_SAMPLE_PATH = Path("data/01_raw/sp500_yahoo_finance_raw.csv")


def clean_raw_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw Yahoo Finance market data."""
    clean_data = data.loc[:, REQUIRED_COLUMNS].copy()
    clean_data["Date"] = pd.to_datetime(clean_data["Date"], errors="raise")

    for column in NUMERIC_COLUMNS:
        clean_data[column] = pd.to_numeric(clean_data[column], errors="raise")

    clean_data = validate_raw_market_data(clean_data)
    clean_data = clean_data.sort_values("Date").reset_index(drop=True)

    return clean_data


def load_and_clean_raw_market_data(
    csv_path: str | Path = RAW_SAMPLE_PATH,
) -> pd.DataFrame:
    """Load and clean the raw S&P 500 sample CSV."""
    data = pd.read_csv(csv_path)
    return clean_raw_market_data(data)
