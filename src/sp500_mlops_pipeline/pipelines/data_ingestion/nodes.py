"""Data ingestion helpers for local market data."""

from pathlib import Path

import pandas as pd


def load_market_data(csv_path: str | Path) -> pd.DataFrame:
    """Load market data from a local CSV file."""
    return pd.read_csv(csv_path)
