"""Data ingestion helpers for local and Yahoo Finance market data."""

from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKER = "^GSPC"
DEFAULT_START_DATE = "2001-01-01"
EXPECTED_MARKET_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def load_market_data(csv_path: str | Path) -> pd.DataFrame:
    """Load market data from a local CSV file."""
    return pd.read_csv(csv_path)


def download_sp500_market_data(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START_DATE,
    yfinance_cache_dir: str | None = None,
) -> pd.DataFrame:
    """Download and normalize a full S&P 500 Yahoo Finance raw-data snapshot.

    This node is intended for manual refreshes via ``kedro run --pipelines=data_ingestion``.
    Incremental updates can be added later; this first version deliberately performs a
    full refresh so the default Kedro pipeline can remain reproducible from the persisted
    local CSV snapshot.
    """
    if yfinance_cache_dir is not None:
        cache_dir = Path(yfinance_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))

    downloaded = yf.download(
        ticker,
        start=start_date,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
    )

    return normalize_yahoo_finance_data(downloaded, ticker=ticker)


def normalize_yahoo_finance_data(
    downloaded: pd.DataFrame,
    ticker: str = DEFAULT_TICKER,
) -> pd.DataFrame:
    """Normalize flat or MultiIndex Yahoo Finance OHLCV data into raw schema."""
    if downloaded.empty:
        raise ValueError(
            f"Yahoo Finance returned no rows for ticker {ticker!r}. "
            "Check internet access, ticker availability, and the configured start_date."
        )

    data = downloaded.copy()
    data = _flatten_yahoo_columns(data, ticker=ticker)
    data = data.reset_index()

    if "Datetime" in data.columns and "Date" not in data.columns:
        data = data.rename(columns={"Datetime": "Date"})

    if "Date" not in data.columns:
        raise ValueError("Yahoo Finance output does not contain a Date column.")

    missing_columns = [
        column for column in EXPECTED_MARKET_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Yahoo Finance output is missing required market columns: "
            f"{missing_columns}"
        )

    normalized = data.loc[:, EXPECTED_MARKET_COLUMNS].copy()
    normalized["Date"] = pd.to_datetime(
        normalized["Date"],
        errors="raise",
    ).dt.tz_localize(None)
    normalized = normalized.sort_values("Date").reset_index(drop=True)

    if normalized["Date"].duplicated().any():
        duplicate_dates = normalized.loc[
            normalized["Date"].duplicated(),
            "Date",
        ].dt.strftime("%Y-%m-%d")
        raise ValueError(
            "Yahoo Finance output contains duplicate dates after normalization: "
            f"{duplicate_dates.head(5).tolist()}"
        )

    return normalized


def _flatten_yahoo_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a DataFrame with one OHLCV column level for a single ticker."""
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    expected_without_date = set(EXPECTED_MARKET_COLUMNS) - {"Date"}
    level_values = [
        set(map(str, data.columns.get_level_values(level)))
        for level in range(data.columns.nlevels)
    ]

    if expected_without_date.intersection(level_values[0]):
        if data.columns.nlevels > 1 and ticker in level_values[1]:
            return data.xs(ticker, axis=1, level=1)

        flattened = data.copy()
        flattened.columns = flattened.columns.get_level_values(0)
        return flattened

    if data.columns.nlevels > 1 and expected_without_date.intersection(
        level_values[1]
    ):
        if ticker in level_values[0]:
            return data.xs(ticker, axis=1, level=0)

        flattened = data.copy()
        flattened.columns = flattened.columns.get_level_values(1)
        return flattened

    raise ValueError(
        "Yahoo Finance output has unsupported MultiIndex columns: "
        f"{data.columns.tolist()}"
    )
