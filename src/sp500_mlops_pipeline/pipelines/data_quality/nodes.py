"""Data quality checks for raw S&P 500 market data."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]


def validate_raw_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw Yahoo Finance market dataset.

    The function returns a copy of the input data with ``Date`` parsed as a
    datetime column when all checks pass. It raises ``ValueError`` with a clear
    message when a check fails.
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Dataset is empty")

    validated = data.copy()

    try:
        validated["Date"] = pd.to_datetime(validated["Date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Date column cannot be parsed as datetime") from exc

    columns_with_missing_values = [
        column for column in REQUIRED_COLUMNS if validated[column].isna().any()
    ]
    if columns_with_missing_values:
        raise ValueError(
            f"Missing values found in required columns: {columns_with_missing_values}"
        )

    if validated["Date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    non_positive_price_columns = [
        column for column in PRICE_COLUMNS if (validated[column] <= 0).any()
    ]
    if non_positive_price_columns:
        raise ValueError(
            f"Price columns must contain positive values: {non_positive_price_columns}"
        )

    if (validated["Volume"] < 0).any():
        raise ValueError("Volume must contain non-negative values")

    return validated


def validate_raw_market_data_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a raw market data CSV file."""
    data = pd.read_csv(csv_path)
    return validate_raw_market_data(data)
